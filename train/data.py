"""Emoji dataset: download, convert to 32x32 RGB tensors in [-1, 1].

Run `python train/data.py` to fetch one set and save a grid to runs/data_grid.png.
"""
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

RAW = Path("data/raw")
SIZE = 32

# name -> (repo url, directory inside the repo that holds the PNGs)
SOURCES = {
    "twemoji": ("https://github.com/jdecked/twemoji", "assets/72x72"),
}


def fetch(name: str) -> Path:
    """Download one emoji set and return the directory containing its PNGs.

    A sparse, blobless clone pulls only the one directory we need instead of
    the whole repo (which also carries SVGs, fonts and history).
    """
    url, subdir = SOURCES[name]
    dest = RAW / name
    if not (dest / subdir).exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", subdir], check=True)
    return dest / subdir


EMOJI_TEST = "https://unicode.org/Public/emoji/latest/emoji-test.txt"

# Class index = position in this list. Order is fixed; append, never reorder,
# or old checkpoints silently mean something else.
CLASSES = [
    "Smileys & Emotion",
    "People & Body",
    "Animals & Nature",
    "Food & Drink",
    "Travel & Places",
    "Activities",
    "Objects",
    "Symbols",
    # Flags (270 in Twemoji) are left out on purpose: see specs/SPEC.md D4.
]

# Codepoints that only restyle an emoji rather than make a new one.
SKIN_TONES = {f"{c:x}" for c in range(0x1F3FB, 0x1F400)}
INVISIBLE = {"fe0f"}  # variation selector: "draw as emoji", vendors disagree on keeping it in filenames
ZWJ_SUFFIXES = {"2640", "2642", "27a1"}  # joined on with 200d: female, male, facing right


def base_key(codepoints: list[str]) -> str:
    """Reduce a codepoint sequence to the emoji it is a variant of.

    ['1f3c3', '1f3fb', '200d', '2640', 'fe0f'] (woman running, light skin)
    -> '1f3c3' (person running)
    """
    out = []
    for cp in codepoints:
        cp = cp.lstrip("0")  # Unicode writes '0023', filenames write '23'
        if cp in SKIN_TONES or cp in INVISIBLE:
            continue
        if cp in ZWJ_SUFFIXES and out and out[-1] == "200d":
            out.pop()
            continue
        out.append(cp)
    return "-".join(out)


def load_groups() -> dict[str, str]:
    """base key -> Unicode group name, parsed from the official emoji list."""
    path = RAW / "emoji-test.txt"
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(EMOJI_TEST).read())
    groups, group = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# group:"):
            group = line.split(":", 1)[1].strip()
        elif ";" in line and not line.startswith("#"):
            key = base_key(line.split(";")[0].lower().split())
            if key:  # a bare skin-tone modifier reduces to nothing
                groups.setdefault(key, group)
    return groups


def index_set(name: str) -> dict[str, tuple[Path, int]]:
    """base key -> (png path, class index) for one emoji set, one image per base emoji."""
    groups = load_groups()
    out, unknown = {}, []
    # Sorting by stem puts '1f3c3' before '1f3c3-1f3fb-...', so the plain
    # version of each emoji is the one that gets kept.
    for path in sorted(fetch(name).glob("*.png"), key=lambda p: p.stem):
        key = base_key(path.stem.split("-"))
        group = groups.get(key)
        if group is None:
            unknown.append(path.stem)
        elif group in CLASSES and key not in out:
            out[key] = (path, CLASSES.index(group))
    if unknown:
        print(f"{name}: {len(unknown)} files not in the Unicode list, skipped: {unknown[:8]}")
    return out


def load_image(path: Path) -> torch.Tensor:
    """PNG with transparency -> float tensor [3, 32, 32] in [-1, 1]."""
    rgba = Image.open(path).convert("RGBA")
    # Composite BEFORE resizing. Fully transparent pixels still carry an RGB
    # value (often black); resizing first would blend that hidden colour into
    # the edges and leave a dark fringe around every emoji.
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(white, rgba).convert("RGB")
    rgb = rgb.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    x = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float()  # HWC uint8 -> CHW float
    # [0, 255] -> [-1, 1]. The noise we add later is centred on 0 with unit
    # variance, so the data has to live on the same scale and be centred too.
    return x / 127.5 - 1.0


if __name__ == "__main__":
    assert base_key("1f3c3 1f3fb 200d 2640 fe0f".split()) == "1f3c3"
    assert base_key(["2640", "fe0f"]) == "2640"  # the female sign on its own is a real emoji
    assert base_key("0023 fe0f 20e3".split()) == base_key("23-20e3".split("-"))  # keycap #

    index = index_set("twemoji")
    total = len(list(fetch("twemoji").glob("*.png")))
    print(f"{total} PNGs -> {len(index)} base emoji")
    Path("runs").mkdir(exist_ok=True)
    rows = []
    for c, name in enumerate(CLASSES):
        paths = [p for p, cls in index.values() if cls == c]
        print(f"  {c} {name:<20} {len(paths):>4}")
        rows += [load_image(p) for p in paths[:: max(1, len(paths) // 16)][:16]]
    x = torch.stack(rows)
    print(f"shape {tuple(x.shape)}  min {x.min():.3f}  max {x.max():.3f}  mean {x.mean():.3f}")
    assert -1 <= x.min() and x.max() <= 1
    # One row per class. save_image expects [0, 1], so undo the scaling for viewing only.
    save_image((x + 1) / 2, "runs/data_grid.png", nrow=16, padding=2)
    print("wrote runs/data_grid.png")
