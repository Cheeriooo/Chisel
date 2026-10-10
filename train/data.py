"""Emoji dataset: three vendors' emoji as 32x32 RGB, labelled by Unicode group.

Run `python train/data.py` to build the cache, print counts and save a grid
to runs/data_grid.png.
"""
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent.parent  # repo root, so scripts work from any directory
RAW = ROOT / "data/raw"
CACHE = ROOT / "data/emoji32.npz"
SIZE = 32
HELDOUT_FRACTION = 0.1
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
# Mirroring an arrow or a letter changes its meaning, so these never get flipped.
NO_FLIP = [CLASSES.index("Symbols")]


def _fluent_codepoints(path: Path) -> list[str]:
    # assets/<Name>/3D/x.png or assets/<Name>/Default/3D/x.png; metadata.json sits in <Name>/
    folder = path.parents[1] if (path.parents[1] / "metadata.json").exists() else path.parents[2]
    return json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["unicode"].split()


# name -> (repo url, sparse-checkout patterns, globs for the PNGs, path -> codepoints)
SOURCES = {
    "twemoji": (
        "https://github.com/jdecked/twemoji",
        ["/assets/72x72/"],
        ["assets/72x72/*.png"],
        lambda p: p.stem.split("-"),
    ),
    "noto": (
        "https://github.com/googlefonts/noto-emoji",
        ["/2D/png/128/"],
        ["2D/png/128/*.png"],
        lambda p: p.stem.removeprefix("emoji_u").split("_"),
    ),
    "fluent": (
        "https://github.com/microsoft/fluentui-emoji",
        # 'Default' is the neutral skin tone; the five toned folders are never downloaded.
        ["/assets/*/metadata.json", "/assets/*/3D/*.png", "/assets/*/Default/3D/*.png"],
        ["assets/*/3D/*.png", "assets/*/Default/3D/*.png"],
        _fluent_codepoints,
    ),
}
VENDORS = list(SOURCES)

# Codepoints that only restyle an emoji rather than make a new one.
SKIN_TONES = {f"{c:x}" for c in range(0x1F3FB, 0x1F400)}
INVISIBLE = {"fe0f"}  # variation selector: "draw as emoji", vendors disagree on keeping it in filenames
ZWJ_SUFFIXES = {"2640", "2642", "27a1"}  # joined on with 200d: female, male, facing right


def fetch(name: str) -> list[Path]:
    """Download one emoji set (once) and return its PNG paths.

    A sparse, blobless clone pulls only the files matching the patterns
    instead of the whole repo (SVGs, fonts, other sizes, history).
    """
    url, patterns, globs, _ = SOURCES[name]
    dest = RAW / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *patterns], check=True)
    return [p for g in globs for p in dest.glob(g)]


def base_key(codepoints: list[str]) -> str:
    """Reduce a codepoint sequence to the emoji it is a variant of.

    ['1f3c3', '1f3fb', '200d', '2640', 'fe0f'] (woman running, light skin)
    -> '1f3c3' (person running)
    """
    out = []
    for cp in codepoints:
        cp = cp.lower().lstrip("0")  # Unicode writes '0023', filenames write '23'
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
            key = base_key(line.split(";")[0].split())
            if key:  # a bare skin-tone modifier reduces to nothing
                groups.setdefault(key, group)
    return groups


def index_set(name: str) -> dict[str, tuple[Path, int]]:
    """base key -> (png path, class index) for one emoji set, one image per base emoji."""
    groups = load_groups()
    codepoints_of = SOURCES[name][3]
    files = [(codepoints_of(p), p) for p in fetch(name)]
    out, unknown = {}, 0
    # Fewest codepoints first, so the plain version of each emoji is the one kept.
    for cps, path in sorted(files, key=lambda f: (len(f[0]), f[0])):
        key = base_key(cps)
        group = groups.get(key)
        if group is None:
            unknown += 1
        elif group in CLASSES and key not in out:
            out[key] = (path, CLASSES.index(group))
    print(f"{name}: {len(files)} files -> {len(out)} base emoji ({unknown} not in the Unicode list)")
    return out


def load_image(path: Path) -> np.ndarray:
    """PNG with transparency -> uint8 array [32, 32, 3] on a white background."""
    rgba = Image.open(path).convert("RGBA")
    # Composite BEFORE resizing. Fully transparent pixels still carry an RGB
    # value (often black); resizing first would blend that hidden colour into
    # the edges and leave a dark fringe around every emoji.
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(white, rgba).convert("RGB")
    return np.asarray(rgb.resize((SIZE, SIZE), Image.Resampling.LANCZOS))


def build_cache() -> dict[str, np.ndarray]:
    """Convert every set once and save one .npz; later calls just load it."""
    if CACHE.exists():
        return dict(np.load(CACHE))
    x, y, vendor, keys = [], [], [], []
    for v, name in enumerate(VENDORS):
        for key, (path, cls) in index_set(name).items():
            x.append(load_image(path)); y.append(cls); vendor.append(v); keys.append(key)
    y, keys = np.array(y), np.array(keys)

    # Hold out whole emoji, not images. If the Twemoji cat were in training and
    # the Noto cat held out, "held-out" would not measure anything new.
    rng = random.Random(0)
    heldout_keys = set()
    for c in range(len(CLASSES)):
        class_keys = sorted(set(keys[y == c]))
        rng.shuffle(class_keys)
        heldout_keys.update(class_keys[: round(len(class_keys) * HELDOUT_FRACTION)])
    heldout = np.array([k in heldout_keys for k in keys])

    data = dict(x=np.stack(x), y=y, vendor=np.array(vendor), key=keys, heldout=heldout)
    np.savez_compressed(CACHE, **data)
    return data


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """uint8 [N, 32, 32, 3] -> float [N, 3, 32, 32] in [-1, 1].

    The noise added during diffusion is centred on 0 with unit variance, so
    the data has to live on the same scale.
    """
    return torch.from_numpy(x).permute(0, 3, 1, 2).float() / 127.5 - 1.0


class EmojiDataset(Dataset):
    """All images sit in memory as one tensor; the whole set is about 50 MB."""

    def __init__(self, split: str = "train", flip: bool = True, classes: list[int] | None = None):
        d = build_cache()
        keep = d["heldout"] == (split == "heldout")
        if classes is not None:
            keep &= np.isin(d["y"], classes)
        self.x = to_tensor(d["x"][keep])
        self.y = torch.from_numpy(d["y"][keep]).long()
        self.flip = flip and split == "train"

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.x[i], self.y[i]
        if self.flip and y.item() not in NO_FLIP and torch.rand(()) < 0.5:
            x = x.flip(-1)  # last dimension is width
        return x, y

    def to(self, device) -> "EmojiDataset":
        self.x, self.y = self.x.to(device), self.y.to(device)
        return self

    def random_batch(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """n random images with the same flip rule as __getitem__, done for the whole batch at once.

        With the data already on the GPU this replaces a DataLoader: no worker
        processes, no copying, and the dataset is small enough not to need them.
        """
        i = torch.randint(len(self.x), (n,), device=self.x.device)
        x, y = self.x[i], self.y[i]
        if self.flip:
            flip = (torch.rand(n, device=x.device) < 0.5) & ~torch.isin(y, torch.tensor(NO_FLIP, device=y.device))
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        return x, y


if __name__ == "__main__":
    assert base_key("1f3c3 1f3fb 200d 2640 fe0f".split()) == "1f3c3"
    assert base_key(["2640", "fe0f"]) == "2640"  # the female sign on its own is a real emoji
    assert base_key("0023 FE0F 20E3".split()) == base_key("23-20e3".split("-"))  # keycap #

    d = build_cache()
    x, y, vendor, key, heldout = d["x"], d["y"], d["vendor"], d["key"], d["heldout"]
    assert not set(key[heldout]) & set(key[~heldout]), "an emoji is on both sides of the split"
    print(f"{len(x)} images, {len(set(key))} distinct emoji, train {(~heldout).sum()}, held-out {heldout.sum()}")
    print(f"{'class':<20}" + "".join(f"{v:>9}" for v in VENDORS) + f"{'total':>9}{'held-out':>10}")
    for c, name in enumerate(CLASSES):
        per_vendor = [((y == c) & (vendor == v)).sum() for v in range(len(VENDORS))]
        print(f"{name:<20}" + "".join(f"{n:>9}" for n in per_vendor) + f"{(y == c).sum():>9}{((y == c) & heldout).sum():>10}")

    train = EmojiDataset("train")
    print(f"train tensor {tuple(train.x.shape)}  min {train.x.min():.3f}  max {train.x.max():.3f}  mean {train.x.mean():.3f}")
    assert -1 <= train.x.min() and train.x.max() <= 1
    symbol = next(i for i in range(len(train)) if train.y[i].item() in NO_FLIP)
    assert all(torch.equal(train[symbol][0], train.x[symbol]) for _ in range(1000)), "a symbol was flipped"
    bx, by = train.random_batch(2000)
    originals = train.x[train.y == NO_FLIP[0]]
    # Exact comparison: torch.cdist is only approximate and reports small non-zero distances for identical rows.
    assert all((originals == img).flatten(1).all(1).any() for img in bx[by == NO_FLIP[0]]), "random_batch flipped a symbol"
    assert not all((train.x == img).flatten(1).all(1).any() for img in bx[:200]), "random_batch never flips"
    face = next(i for i in range(len(train)) if train.y[i].item() not in NO_FLIP and not torch.equal(train.x[i], train.x[i].flip(-1)))
    assert any(not torch.equal(train[face][0], train.x[face]) for _ in range(100)), "flip never happens"

    # Two emoji per class that all three vendors drew: one row per vendor, same emoji in each column.
    shared = [k for k in sorted(set(key)) if (key == k).sum() == len(VENDORS)]
    cols = [k for c in range(len(CLASSES)) for k in [k for k in shared if y[key == k][0] == c][:2]]
    grid = np.stack([x[(key == k) & (vendor == v)][0] for v in range(len(VENDORS)) for k in cols])
    Path("runs").mkdir(exist_ok=True)
    # save_image expects [0, 1], so undo the scaling for viewing only.
    save_image((to_tensor(grid) + 1) / 2, "runs/data_grid.png", nrow=len(cols), padding=2)
    print("wrote runs/data_grid.png")
