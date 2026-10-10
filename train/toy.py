"""Milestone 1: the smallest training loop that can work.

No EMA, no class labels, no data loader. It exists to prove the pieces fit.

  python train/toy.py --n 16                 # overfit 16 images: the correctness test
  python train/toy.py --cls 0 --steps 8000   # one class, first real samples
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from data import CLASSES, EmojiDataset
from diffusion import T, ddpm_sample, make_alpha_bar, q_sample
from unet import TinyUNet

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=0, help="use only the first n images (0 = all)")
p.add_argument("--cls", type=int, default=None, help="train on one class index")
p.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--ch", type=int, default=32)
p.add_argument("--out", default="runs/toy")
args = p.parse_args()

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
data = EmojiDataset("train", classes=None if args.cls is None else [args.cls])
x_all = data.x[: args.n or None].to(device)  # the whole training set lives on the GPU
print(f"{len(x_all)} images" + (f" of class '{CLASSES[args.cls]}'" if args.cls is not None else "") + f", {args.schedule} schedule, {device}")

alpha_bar = make_alpha_bar(args.schedule)
model = TinyUNet(args.ch).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(1, args.steps + 1):
    x0 = x_all[torch.randint(len(x_all), (64,), device=device)]    # a random batch of clean images
    t = torch.randint(0, T, (len(x0),), device=device)             # a random noise level for each
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, alpha_bar)                        # noise them
    loss = F.mse_loss(model(x_t, t), noise)                        # can the model tell what noise was added?
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0 or step == 1:
        print(f"step {step:>5}  loss {loss.item():.4f}")

model.eval()
samples = ddpm_sample(model, 16, alpha_bar, device)
# For each sample, squared distance to its closest training image. Near zero
# means it reproduced a training image, which is the goal when overfitting.
nearest = torch.cdist(samples.flatten(1), x_all.flatten(1)).pow(2).div(samples[0].numel()).min(dim=1).values
print(f"nearest-training-image MSE per sample: median {nearest.median():.4f}, worst {nearest.max():.4f}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), f"{args.out}.pt")
# Top: 16 samples. Bottom: 16 training images, for comparison.
save_image((torch.cat([samples, x_all[:16]]).clamp(-1, 1) + 1) / 2, f"{args.out}.png", nrow=16, padding=2)
print(f"wrote {args.out}.png and {args.out}.pt")
