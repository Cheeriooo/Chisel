"""Sample grids and GIFs from a trained run. Everything is written to runs/<run>/.

  python train/sample.py grid --run base      # one row per class
  python train/sample.py sdedit --run base    # rough sketches -> emoji at five strengths
  python train/sample.py inpaint --run base   # half-erased real emoji, refilled
  python train/sample.py gif --run base       # the denoising process, for the README
"""
import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid, save_image

from data import CLASSES, ROOT, EmojiDataset, to_tensor
from diffusion import ddim_sample, make_eps_fn
from eval import device, generate, load_run


def sketches() -> tuple[torch.Tensor, torch.Tensor]:
    """Three deliberately crude drawings, the kind a visitor makes with a fat brush, and their classes."""
    def canvas():
        img = Image.new("RGB", (32, 32), "white")
        return img, ImageDraw.Draw(img)

    face, d = canvas()
    d.ellipse([4, 4, 27, 27], fill="#ffcc4d")
    d.rectangle([10, 11, 12, 14], fill="#1c1a17"); d.rectangle([19, 11, 21, 14], fill="#1c1a17")
    d.line([10, 21, 15, 23, 21, 21], fill="#1c1a17", width=2)

    heart, d = canvas()
    d.ellipse([4, 6, 16, 18], fill="#dd2e44"); d.ellipse([15, 6, 27, 18], fill="#dd2e44")
    d.polygon([5, 15, 26, 15, 16, 28], fill="#dd2e44")

    tree, d = canvas()
    d.rectangle([14, 20, 17, 29], fill="#8a4b38")
    d.ellipse([6, 3, 25, 22], fill="#77b255")

    x = to_tensor(np.stack([np.asarray(i) for i in (face, heart, tree)]))
    return x.to(device), torch.tensor([0, 0, 2], device=device)  # Smileys, Smileys, Animals & Nature


def noise_like(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    return torch.randn(x.shape, device=device, generator=torch.Generator(device).manual_seed(seed))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["grid", "sdedit", "inpaint", "gif"])
    p.add_argument("--run", default="base")
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()
    model, alpha_bar, pred = load_run(args.run)
    out = ROOT / "runs" / args.run

    def eps_fn(cls):
        return make_eps_fn(model, alpha_bar, cls, args.guidance, pred)

    if args.what == "grid":
        x, _ = generate(model, alpha_bar, pred, per_class=12, steps=args.steps, guidance=args.guidance, seed=1)
        save_image((x + 1) / 2, out / "grid.png", nrow=12, padding=2)

    elif args.what == "sdedit":
        x, cls = sketches()
        strengths = [0.2, 0.4, 0.6, 0.8, 1.0]
        cols = [x] + [ddim_sample(eps_fn(cls), noise_like(x), alpha_bar, args.steps, init=x, strength=s) for s in strengths]
        # One row per sketch: the sketch, then strength 0.2 ... 1.0.
        save_image((torch.stack(cols, dim=1).flatten(0, 1) + 1) / 2, out / "sdedit.png", nrow=len(cols), padding=2)

    elif args.what == "inpaint":
        held = EmojiDataset("heldout")
        pick = torch.cat([(held.y == c).nonzero()[:2, 0] for c in range(len(CLASSES))])  # two real, unseen emoji per class
        x, cls = held.x[pick].to(device), held.y[pick].to(device)
        mask = torch.zeros(1, 1, 32, 32, device=device)
        mask[..., 16:] = 1  # the right half is the hole
        filled = ddim_sample(eps_fn(cls), noise_like(x), alpha_bar, args.steps, init=x, mask=mask)
        assert torch.allclose(filled[..., :16], x[..., :16], atol=1e-5), "pixels outside the mask changed"
        # Rows: original, what the model was given, what it filled in.
        save_image((torch.cat([x, x * (1 - mask) + mask * 0.6, filled]) + 1) / 2, out / "inpaint.png", nrow=len(x), padding=2)

    else:
        cls = torch.arange(len(CLASSES), device=device)
        noise = noise_like(torch.empty(len(cls), 3, 32, 32), seed=3)
        final, history = ddim_sample(eps_fn(cls), noise, alpha_bar, args.steps, return_all=True)
        frames = []
        for xt, x0 in history + [(final, final)]:
            # Top row: the noisy image the sampler holds. Bottom row: the model's current guess.
            grid = make_grid((torch.cat([xt, x0]).clamp(-1, 1) + 1) / 2, nrow=len(cls), padding=2, pad_value=1)
            frame = Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
            frames.append(frame.resize((frame.width * 4, frame.height * 4), Image.Resampling.NEAREST))
        frames[0].save(out / "denoise.gif", save_all=True, append_images=frames[1:], duration=[120] * (len(frames) - 1) + [2000], loop=0)

    print(f"wrote {out / {'grid': 'grid.png', 'sdedit': 'sdedit.png', 'inpaint': 'inpaint.png', 'gif': 'denoise.gif'}[args.what]}")
