"""Export ONE denoising step of a trained run to ONNX and check it against PyTorch.

Only the network is exported. The sampling loop stays outside (in Python
here, in TypeScript in the browser), which is what lets the web app pause,
scrub and edit between steps.

  python export/export_onnx.py --run base

Writes into runs/<run>/export/:
  chisel.onnx     inputs x_t [B,3,32,32] float32, t [B] int64, cls [B] int64 -> eps [B,3,32,32]
  schedule.json   alpha_bar table, class names, null class index (the browser never recomputes these)
  chisel.ref.json one model call on a formula input, for export/smoke.html
  parity.bin      starting noise + the Python sampler's 20-step result, for the JS sampler test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.stdout.reconfigure(encoding="utf-8")  # the exporter prints emoji, which the Windows console codec rejects
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from data import CLASSES  # noqa: E402
from diffusion import ddim_sample, make_alpha_bar, make_eps_fn  # noqa: E402
from unet import UNet  # noqa: E402

PARITY = dict(cls=0, guidance=2.0, steps=20, seed=0)


class EpsModel(torch.nn.Module):
    """What gets exported: always outputs predicted noise, whatever the network was trained to predict.

    For a v-prediction network the conversion eps = sqrt(a) * v + sqrt(1 - a) * x_t
    is baked into the graph, so the browser sampler has one contract to follow.
    """

    def __init__(self, model: UNet, alpha_bar: torch.Tensor, pred: str):
        super().__init__()
        self.model, self.pred = model, pred
        self.register_buffer("alpha_bar", alpha_bar.float())

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        out = self.model(x_t, t, cls)
        if self.pred == "v":
            a = self.alpha_bar[t].view(-1, 1, 1, 1)
            out = a.sqrt() * out + (1 - a).sqrt() * x_t
        return out


p = argparse.ArgumentParser()
p.add_argument("--run", default="base")
args = p.parse_args()

ckpt = torch.load(ROOT / "runs" / args.run / "ckpt.pt", map_location="cpu")
cfg = ckpt["args"]
unet = UNet(len(CLASSES), ch=cfg["ch"])
unet.load_state_dict(ckpt["ema"])  # the averaged weights are the ones that sample well
alpha_bar = make_alpha_bar(cfg["schedule"])
model = EpsModel(unet, alpha_bar, cfg["pred"]).eval()

out_dir = ROOT / "runs" / args.run / "export"
out_dir.mkdir(exist_ok=True)
onnx_path = out_dir / "chisel.onnx"

# The example input only fixes dtypes and the non-batch dimensions. The batch
# dimension is declared dynamic so the browser can send 1 image, or 2 at once
# for classifier-free guidance (conditional + unconditional).
example = (torch.randn(2, 3, 32, 32), torch.tensor([10, 500]), torch.tensor([0, unet.null_class]))
batch = torch.export.Dim("batch")
torch.onnx.export(
    model, example, str(onnx_path),
    input_names=["x_t", "t", "cls"], output_names=["eps"],
    dynamic_shapes=({0: batch}, {0: batch}, {0: batch}),
    external_data=False,  # weights inside the .onnx: the browser fetches one file
)

session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
print("inputs:", [(i.name, i.type, i.shape) for i in session.get_inputs()])
worst = 0.0
rng = np.random.default_rng(0)
for b in [1, 2, 5]:
    for t_value in [0, 250, 999]:
        x = rng.standard_normal((b, 3, 32, 32), dtype=np.float32)
        t = np.full((b,), t_value, dtype=np.int64)
        c = rng.integers(0, len(CLASSES) + 1, (b,), dtype=np.int64)
        got = session.run(None, {"x_t": x, "t": t, "cls": c})[0]
        with torch.no_grad():
            want = model(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(c)).numpy()
        worst = max(worst, float(np.abs(got - want).max()))
print(f"{onnx_path.name}: {onnx_path.stat().st_size / 2**20:.2f} MB, max abs diff vs PyTorch {worst:.2e}")
assert worst < 1e-3, "ONNX output does not match PyTorch"

(out_dir / "schedule.json").write_text(json.dumps(dict(
    T=len(alpha_bar), classes=CLASSES, null_class=unet.null_class,
    alpha_bar=[float(v) for v in alpha_bar.float()],  # float32 values, exactly what PyTorch sampling uses
)))

# Reference output for the browser smoke test (export/smoke.html). The input
# is a formula, not random numbers, so JavaScript can rebuild it exactly.
x = np.sin(np.arange(2 * 3 * 32 * 32, dtype=np.float64)).astype(np.float32).reshape(2, 3, 32, 32)
ref = session.run(None, {"x_t": x, "t": np.array([10, 500], dtype=np.int64),
                         "cls": np.array([0, unet.null_class], dtype=np.int64)})[0]
(out_dir / "chisel.ref.json").write_text(json.dumps([round(float(v), 6) for v in ref.ravel()]))

# Sampler parity fixture: fixed starting noise and what the Python DDIM sampler
# makes of it. The TypeScript sampler must turn the same noise into the same image.
noise = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(PARITY["seed"]))
with torch.no_grad():
    image = ddim_sample(make_eps_fn(unet, alpha_bar, torch.tensor([PARITY["cls"]]), PARITY["guidance"], cfg["pred"]),
                        noise, alpha_bar, steps=PARITY["steps"])
(out_dir / "parity.bin").write_bytes(noise.numpy().tobytes() + image.numpy().tobytes())
(out_dir / "parity.json").write_text(json.dumps(PARITY))
print(f"wrote schedule.json, chisel.ref.json, parity.bin to {out_dir}")
