"""Export ONE denoising step to ONNX and check it against PyTorch.

Only the network is exported. The sampling loop stays outside (in Python
here, in TypeScript in the browser), which is what lets the web app pause,
scrub and edit between steps.

  python export/export_onnx.py --ckpt runs/toy_smileys_cosine.pt --out runs/toy.onnx
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.stdout.reconfigure(encoding="utf-8")  # the exporter prints emoji, which the Windows console codec rejects
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from unet import TinyUNet  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--ch", type=int, default=32)
p.add_argument("--out", required=True)
args = p.parse_args()

model = TinyUNet(args.ch)
model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
model.eval()

# The example input only fixes dtypes and the non-batch dimensions. The batch
# dimension is declared dynamic so the browser can send 1 image, or 2 at once
# for classifier-free guidance (conditional + unconditional).
example = (torch.randn(2, 3, 32, 32), torch.tensor([10, 500]))
batch = torch.export.Dim("batch")
torch.onnx.export(
    model, example, args.out,
    input_names=["x_t", "t"], output_names=["eps"],
    dynamic_shapes=({0: batch}, {0: batch}),
    external_data=False,  # weights inside the .onnx: the browser fetches one file
)

session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
print("inputs:", [(i.name, i.type, i.shape) for i in session.get_inputs()])
worst = 0.0
rng = np.random.default_rng(0)
for b in [1, 2, 5]:
    for t_value in [0, 250, 999]:
        x = rng.standard_normal((b, 3, 32, 32), dtype=np.float32)
        t = np.full((b,), t_value, dtype=np.int64)
        got = session.run(None, {"x_t": x, "t": t})[0]
        with torch.no_grad():
            want = model(torch.from_numpy(x), torch.from_numpy(t)).numpy()
        worst = max(worst, float(np.abs(got - want).max()))

# Reference output for the browser smoke test (export/smoke.html). The input
# is a formula, not random numbers, so JavaScript can rebuild it exactly.
x = np.sin(np.arange(2 * 3 * 32 * 32, dtype=np.float64)).astype(np.float32).reshape(2, 3, 32, 32)
ref = session.run(None, {"x_t": x, "t": np.array([10, 500], dtype=np.int64)})[0]
Path(args.out).with_suffix(".ref.json").write_text(json.dumps([round(float(v), 6) for v in ref.ravel()]))

size_kb = Path(args.out).stat().st_size / 1024
print(f"{args.out}: {size_kb:.0f} KB, max abs diff vs PyTorch {worst:.2e}")
assert worst < 1e-3, "ONNX output does not match PyTorch"
