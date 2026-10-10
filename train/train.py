"""Train the class-conditional U-Net.

  python train/train.py --name base                      # full run
  python train/train.py --name overfit --n 16 --steps 3000 --sample-every 3000

Everything for a run lands in runs/<name>/: checkpoint, loss log, sample grids.
Re-running the same command resumes from the checkpoint.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from data import CLASSES, ROOT, EmojiDataset
from diffusion import T, ddim_sample, make_alpha_bar, make_eps_fn, q_sample, v_target
from unet import UNet, count_params

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--ch", type=int, default=24)
p.add_argument("--steps", type=int, default=40000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--p-uncond", type=float, default=0.1, help="chance of hiding the label, for classifier-free guidance")
p.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
p.add_argument("--pred", default="v", choices=["eps", "v"], help="what the network predicts")
p.add_argument("--n", type=int, default=0, help="use only the first n images (0 = all); for the overfit test")
p.add_argument("--sample-every", type=int, default=2500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--amp", action="store_true", help="mixed precision: faster on GPUs with tensor cores (Colab T4); leave off on GTX 16-series, which is prone to NaNs in fp16")
args = p.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
run = ROOT / "runs" / args.name
run.mkdir(parents=True, exist_ok=True)

data = EmojiDataset("train")
if args.n:
    data.x, data.y, data.flip = data.x[: args.n], data.y[: args.n], False
data.to(device)
alpha_bar = make_alpha_bar(args.schedule)

model = UNet(len(CLASSES), ch=args.ch).to(device)
# EMA: a second copy of the weights that is a running average of the first.
# Individual training steps jitter; the average is smoother and samples
# noticeably better. It is never trained directly, only updated by averaging.
ema = copy.deepcopy(model).requires_grad_(False).eval()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
# Warmup: ramp the learning rate from 0 over the first steps. Adam's estimates
# of gradient scale are unreliable at the very start, and a full-size step
# from random weights can do lasting damage.
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
# Mixed precision: run the forward pass in 16-bit floats where safe. The scaler
# multiplies the loss up before backward so tiny gradients do not round to zero.
scaler = torch.amp.GradScaler(device, enabled=args.amp)

step = 0
ckpt_path = run / "ckpt.pt"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"]); ema.load_state_dict(ckpt["ema"])
    opt.load_state_dict(ckpt["opt"]); sched.load_state_dict(ckpt["sched"])
    step = ckpt["step"]
    print(f"resumed from step {step}")
(run / "config.json").write_text(json.dumps(vars(args), indent=2))
print(f"{len(data)} images, {count_params(model):,} parameters, predicting {args.pred}, {args.schedule} schedule, {device}")


def save_samples(tag: str) -> None:
    """One row per class, 8 samples each, from the EMA weights with 20 DDIM steps."""
    cls = torch.arange(len(CLASSES), device=device).repeat_interleave(8)
    noise = torch.randn(len(cls), 3, 32, 32, device=device, generator=torch.Generator(device).manual_seed(0))
    samples = ddim_sample(make_eps_fn(ema, alpha_bar, cls, guidance=2.0, pred=args.pred), noise, alpha_bar, steps=20)
    save_image((samples + 1) / 2, run / f"samples_{tag}.png", nrow=8, padding=2)


log = open(run / "loss.csv", "a")
recent, start = [], time.time()
while step < args.steps:
    x0, cls = data.random_batch(args.batch)
    # Hide the label for a random 10% so the same network also learns the
    # unconditional prediction that guidance needs.
    cls = torch.where(torch.rand(len(cls), device=device) < args.p_uncond, model.null_class, cls)
    t = torch.randint(0, T, (len(x0),), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, alpha_bar)
    target = noise if args.pred == "eps" else v_target(x0, t, noise, alpha_bar)
    with torch.autocast(device, enabled=args.amp):
        loss = F.mse_loss(model(x_t, t, cls).float(), target)
    assert torch.isfinite(loss), f"loss became {loss.item()} at step {step}"

    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    # Clip rare huge gradients (a single unlucky batch) instead of letting them wreck the weights.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    step += 1

    # Early on the average would still be mostly random initial weights, so
    # the decay itself is warmed up: short memory first, long memory later.
    decay = min(args.ema, (1 + step) / (10 + step))
    with torch.no_grad():
        for p_ema, p_model in zip(ema.parameters(), model.parameters()):
            p_ema.lerp_(p_model, 1 - decay)

    recent.append(loss.item())
    if step % 100 == 0:
        mean = sum(recent) / len(recent)
        recent.clear()
        log.write(f"{step},{mean:.5f}\n"); log.flush()
        if step % 500 == 0:
            rate = 100 * 5 / (time.time() - start); start = time.time()
            print(f"step {step:>6}  loss {mean:.4f}  {rate:.1f} steps/s  about {(args.steps - step) / rate / 60:.0f} min left", flush=True)
    if step % args.sample_every == 0 or step == args.steps:
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), step=step, args=vars(args)), ckpt_path)
        save_samples(f"{step:06d}")
print(f"done: {run}")
