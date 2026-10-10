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


if __name__ == "__main__":
    model = TinyUNet()
    x, t = torch.randn(4, 3, 32, 32), torch.randint(0, 1000, (4,))
    assert model(x, t).shape == x.shape
    print(f"TinyUNet: {sum(p.numel() for p in model.parameters()):,} parameters, output shape ok")
