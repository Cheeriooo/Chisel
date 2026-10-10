"""The denoising network: given a noisy image and how noisy it is, predict the noise."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Turn an integer step t into a vector of sines and cosines, shape [B, dim].

    A raw number like 743 is a poor network input: 742 and 743 look almost
    identical and the scale is huge. Sines at many frequencies give each step
    a distinct, smooth fingerprint. Same idea as transformer position encoding.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    """Two convolutions with a shortcut around them, told what step it is.

    The step embedding is added to the feature maps between the two convs, so
    every block can behave differently at high noise and at low noise.
    """

    def __init__(self, c_in: int, c_out: int, emb_dim: int):
        super().__init__()
        # GroupNorm rather than BatchNorm: each image in a batch has a
        # different noise level, so statistics shared across the batch are meaningless.
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        # The shortcut needs a 1x1 conv only when the channel count changes.
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]  # [B, C] -> [B, C, 1, 1], one offset per channel
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """Milestone 1 toy: two resolutions, no attention, no class label.

    Shape of the 'U': 32x32 -> 16x16 -> back to 32x32. The 32x32 features from
    the way down are concatenated on the way up (the skip connection), so fine
    detail does not have to squeeze through the 16x16 bottleneck.
    """

    def __init__(self, ch: int = 32):
        super().__init__()
        emb_dim = ch * 4
        self.ch = ch
        self.time_mlp = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        self.down1 = ResBlock(ch, ch, emb_dim)
        self.downsample = nn.Conv2d(ch, ch, 3, stride=2, padding=1)  # 32 -> 16
        self.down2 = ResBlock(ch, ch * 2, emb_dim)
        self.mid = ResBlock(ch * 2, ch * 2, emb_dim)
        self.up2 = ResBlock(ch * 2, ch, emb_dim)
        self.up1 = ResBlock(ch * 2, ch, emb_dim)  # input is upsampled features + skip, hence ch * 2
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.ch))
        h1 = self.down1(self.conv_in(x), emb)             # [B, ch, 32, 32]
        h = self.down2(self.downsample(h1), emb)          # [B, 2ch, 16, 16]
        h = self.mid(h, emb)
        h = self.up2(h, emb)                              # [B, ch, 16, 16]
        h = F.interpolate(h, scale_factor=2.0, mode="nearest")  # 16 -> 32
        h = self.up1(torch.cat([h, h1], dim=1), emb)      # [B, ch, 32, 32]
        return self.conv_out(F.silu(self.norm_out(h)))


class Attention(nn.Module):
    """Self-attention over the pixels of a feature map.

    A convolution only sees a 3x3 neighbourhood. Attention lets every position
    look at every other one, which is how the model keeps a face symmetric or
    a colour consistent across the image. Cost grows with (H*W)^2, so it is
    used only at 8x8, where there are just 64 positions.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)  # each [B, C, HW]
        # weights[b, i, j]: how much position i takes from position j.
        weights = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)  # [B, HW, HW]
        mixed = (v @ weights.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.out(mixed)


class UNet(nn.Module):
    """The full model: three resolutions (32, 16, 8), attention at 8x8, class-conditional.

    `cls` is a class index in [0, num_classes]. The extra index num_classes is
    the "null" class meaning "no label given". Training sometimes swaps the
    real label for it, so one network learns both the conditional and the
    unconditional prediction, which classifier-free guidance then combines.
    """

    def __init__(self, num_classes: int, ch: int = 24, mults: tuple[int, ...] = (1, 2, 4), blocks: int = 2):
        super().__init__()
        emb_dim = ch * 4
        self.ch, self.null_class = ch, num_classes
        self.time_mlp = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)

        # Way down. `skips` records the channel count of every feature map
        # that the way up will receive through a skip connection.
        self.down = nn.ModuleList()
        skips, c = [ch], ch
        for level, mult in enumerate(mults):
            last = level == len(mults) - 1
            for _ in range(blocks):
                self.down.append(ResBlock(c, ch * mult, emb_dim))
                c = ch * mult
                if last:
                    self.down.append(Attention(c))
                skips.append(c)
            if not last:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))  # halve the resolution
                skips.append(c)

        self.mid = nn.ModuleList([ResBlock(c, c, emb_dim), Attention(c), ResBlock(c, c, emb_dim)])

        # Way up: mirror image, each block also taking one skip connection.
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(mults))):
            for _ in range(blocks + 1):
                self.up.append(ResBlock(c + skips.pop(), ch * mult, emb_dim))
                c = ch * mult
                if level == len(mults) - 1:
                    self.up.append(Attention(c))
            if level > 0:
                self.up.append(nn.Upsample(scale_factor=2.0, mode="nearest"))

        self.norm_out = nn.GroupNorm(8, c)
        self.conv_out = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.ch)) + self.class_emb(cls)
        h = self.conv_in(x)
        skips = [h]
        for layer in self.down:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            if not isinstance(layer, Attention):  # attention refines the block before it; that pair gives one skip
                skips.append(h)
            else:
                skips[-1] = h
        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
            else:
                h = layer(h)
        return self.conv_out(F.silu(self.norm_out(h)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    x, t = torch.randn(4, 3, 32, 32), torch.randint(0, 1000, (4,))
    assert TinyUNet()(x, t).shape == x.shape
    print(f"TinyUNet: {count_params(TinyUNet()):,} parameters")
    cls = torch.tensor([0, 3, 7, 8])  # 8 is the null class
    for ch in [16, 24, 32, 48]:  # must be a multiple of 8 for GroupNorm
        model = UNet(num_classes=8, ch=ch)
        assert model(x, t, cls).shape == x.shape
        n = count_params(model)
        print(f"UNet ch={ch:<3} {n:>10,} parameters  {n * 4 / 2**20:5.1f} MB fp32  {n * 2 / 2**20:5.1f} MB fp16")
