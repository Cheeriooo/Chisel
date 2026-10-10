"""Evaluation: recognisability, step and guidance ablations, memorization check.

  python train/eval.py classifier           # train the judge, print its held-out accuracy
  python train/eval.py ablate --run base    # steps and guidance tables + grids
  python train/eval.py memorize --run base  # nearest training image for generated samples

Recognisability = the share of generated images that a classifier, trained
only on real emoji, assigns to the class that was requested. The classifier's
accuracy on real held-out emoji is the ceiling: the generator cannot be
expected to beat real data.
"""
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from data import CLASSES, ROOT, EmojiDataset
from diffusion import ddim_sample, make_alpha_bar, make_eps_fn
from unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
CLASSIFIER = ROOT / "runs" / "classifier.pt"


def make_classifier() -> nn.Module:
    def block(c_in, c_out):
        return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(128, len(CLASSES)))


def train_classifier(steps: int = 3000) -> None:
    torch.manual_seed(0)
    train, held = EmojiDataset("train").to(device), EmojiDataset("heldout").to(device)
    model = make_classifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    for step in range(1, steps + 1):
        x, y = train.random_batch(128)
        loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(held.x).argmax(1) == held.y).float().mean().item()
                train_acc = (model(train.x).argmax(1) == train.y).float().mean().item()
            model.train()
            print(f"step {step:>5}  loss {loss.item():.3f}  train acc {train_acc:.3f}  held-out acc {acc:.3f}")
    model.eval()
    with torch.no_grad():
        pred = model(held.x).argmax(1)
    print("held-out accuracy per class:")
    for c, name in enumerate(CLASSES):
        print(f"  {name:<20} {(pred[held.y == c] == c).float().mean().item():.3f}  (n={(held.y == c).sum().item()})")
    print(f"overall {(pred == held.y).float().mean().item():.3f}; chance would be {1 / len(CLASSES):.3f}")
    torch.save(model.state_dict(), CLASSIFIER)


def load_run(name: str):
    """EMA model, alpha_bar and prediction type of a finished training run."""
    ckpt = torch.load(ROOT / "runs" / name / "ckpt.pt", map_location=device)
    cfg = ckpt["args"]
    model = UNet(len(CLASSES), ch=cfg["ch"]).to(device)
    model.load_state_dict(ckpt["ema"])
    return model.eval(), make_alpha_bar(cfg["schedule"]), cfg["pred"]


@torch.no_grad()
def generate(model, alpha_bar, pred, per_class: int, steps: int, guidance: float, seed: int = 0):
    """per_class samples for every class. Returns (images, requested class)."""
    cls = torch.arange(len(CLASSES), device=device).repeat_interleave(per_class)
    noise = torch.randn(len(cls), 3, 32, 32, device=device, generator=torch.Generator(device).manual_seed(seed))
    out = [ddim_sample(make_eps_fn(model, alpha_bar, c, guidance, pred), n, alpha_bar, steps)
           for n, c in zip(noise.split(200), cls.split(200))]
    return torch.cat(out), cls


def ablate(run: str, per_class: int = 100) -> None:
    model, alpha_bar, pred = load_run(run)
    judge = make_classifier().to(device)
    judge.load_state_dict(torch.load(CLASSIFIER, map_location=device))
    judge.eval()
    out_dir = ROOT / "runs" / run
    results = []
    for label, settings in [("steps", [(s, 2.0) for s in (5, 10, 20, 50)]),
                            ("guidance", [(20, w) for w in (0.0, 1.0, 1.5, 2.0, 3.0, 5.0)])]:
        print(f"\n{label} ablation ({per_class} samples per class)")
        print(f"{'steps':>6}{'guidance':>10}{'recognised':>12}{'diversity':>11}")
        rows = []
        for steps, w in settings:
            x, cls = generate(model, alpha_bar, pred, per_class, steps, w)
            with torch.no_grad():
                recognised = (judge(x).argmax(1) == cls).float().mean().item()
            # Diversity: average pixel distance between two samples of the same class.
            # It falls when guidance pushes every sample towards the class "average".
            flat = x.flatten(1).view(len(CLASSES), per_class, -1)
            diversity = torch.cdist(flat, flat).sum().item() / (len(CLASSES) * per_class * (per_class - 1)) / flat.shape[-1] ** 0.5
            print(f"{steps:>6}{w:>10.1f}{recognised:>12.3f}{diversity:>11.3f}")
            results.append(dict(ablation=label, steps=steps, guidance=w, recognised=recognised, diversity=diversity))
            rows.append(x.view(len(CLASSES), per_class, 3, 32, 32)[:, :2].flatten(0, 1))  # 2 per class
        # One row per setting, two samples of each class.
        save_image((torch.cat(rows) + 1) / 2, out_dir / f"ablation_{label}.png", nrow=len(CLASSES) * 2, padding=2)
    (out_dir / "ablation.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir / 'ablation.json'} and ablation_*.png")


def memorize(run: str, per_class: int = 50) -> None:
    model, alpha_bar, pred = load_run(run)
    train, held = EmojiDataset("train", flip=False).to(device), EmojiDataset("heldout").to(device)
    x, _ = generate(model, alpha_bar, pred, per_class, steps=20, guidance=2.0)
    # Compare against the training set and its mirror image, since flips were used in training.
    bank = torch.cat([train.x, train.x.flip(-1)])

    def nearest(a):  # root-mean-square pixel distance to the closest training image, and its index
        d, i = (torch.cdist(a.flatten(1), bank.flatten(1)) / a[0].numel() ** 0.5).min(1)
        return d, i

    d_gen, i_gen = nearest(x)
    # Reference: how close is a real, never-trained-on emoji to the training set?
    # Generated samples sitting much closer than that are copies, not new images.
    d_held, _ = nearest(held.x)
    threshold = d_held.quantile(0.05).item()
    print(f"nearest-training-image distance (RMS per pixel, images in [-1, 1])")
    print(f"  generated: median {d_gen.median():.3f}  min {d_gen.min():.3f}")
    print(f"  held-out real emoji: median {d_held.median():.3f}  5th percentile {threshold:.3f}")
    print(f"  near-copies (closer than the 5th percentile of held-out): {(d_gen < threshold).float().mean().item():.1%} of {len(x)}")
    order = d_gen.argsort()[:16]
    # Top row: the 16 generated samples closest to a training image. Bottom row: that training image.
    save_image((torch.cat([x[order], bank[i_gen[order]]]) + 1) / 2, ROOT / "runs" / run / "memorization.png", nrow=16, padding=2)
    print(f"wrote runs/{run}/memorization.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["classifier", "ablate", "memorize"])
    p.add_argument("--run", default="base")
    args = p.parse_args()
    if args.what == "classifier":
        train_classifier()
    elif args.what == "ablate":
        ablate(args.run)
    else:
        memorize(args.run)
