"""Diffusion maths: noise schedules, the forward (noising) process, samplers.

Run `python train/diffusion.py` to save runs/forward.png: one emoji being
noised under each schedule.
"""
import math

import torch

T = 1000


def make_alpha_bar(schedule: str = "cosine", steps: int = T) -> torch.Tensor:
    """alpha_bar[t] = fraction of the original image's variance still left at step t.

    It starts near 1 (clean image) and falls to near 0 (pure noise). Everything
    else in diffusion is derived from this one table. Computed in float64
    because it is a product of a thousand numbers close to 1.
    """
    if schedule == "linear":
        # Ho et al. 2020: the noise added per step grows linearly.
        betas = torch.linspace(1e-4, 0.02, steps, dtype=torch.float64)
    elif schedule == "cosine":
        # Nichol & Dhariwal 2021: define alpha_bar directly as a cosine curve,
        # so the image fades steadily instead of being wiped out early.
        s = 0.008
        t = torch.arange(steps + 1, dtype=torch.float64) / steps
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)  # the last step would otherwise be beta = 1
    else:
        raise ValueError(schedule)
    return torch.cumprod(1 - betas, dim=0)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """Jump straight from a clean image to its noised version at step t.

    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

    No loop over steps is needed: adding Gaussian noise t times is the same as
    adding one bigger Gaussian once. This is what makes training cheap, since
    each batch can use a random t per image.
    """
    a = alpha_bar.to(x0.device, x0.dtype)[t].view(-1, 1, 1, 1)  # one scalar per image, broadcast over C, H, W
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


def predict_x0(x_t: torch.Tensor, eps: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Invert q_sample: if this is the noise, what clean image does it imply?

    Clamping to the valid pixel range matters most at the noisiest steps,
    where dividing by sqrt(alpha_bar), a number near zero, magnifies any error.
    """
    return ((x_t - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)


@torch.no_grad()
def ddpm_sample(model, n: int, alpha_bar: torch.Tensor, device, seed: int = 0) -> torch.Tensor:
    """Ancestral sampling: start from pure noise and undo one step at a time, all T of them."""
    g = torch.Generator(device).manual_seed(seed)
    ab = alpha_bar.to(device, torch.float32)
    x = torch.randn(n, 3, 32, 32, device=device, generator=g)
    for t in range(len(ab) - 1, -1, -1):
        a = ab[t]
        a_prev = ab[t - 1] if t > 0 else torch.ones((), device=device)
        beta = 1 - a / a_prev  # noise added by this one step
        x0 = predict_x0(x, model(x, torch.full((n,), t, device=device)), a)
        # Mean and variance of q(x_{t-1} | x_t, x0): the exact reverse step if
        # x0 were known. We plug in the model's guess of x0 (Ho et al., eq. 7).
        mean = (a_prev.sqrt() * beta * x0 + (1 - beta).sqrt() * (1 - a_prev) * x) / (1 - a)
        var = beta * (1 - a_prev) / (1 - a)
        x = mean + var.sqrt() * torch.randn(x.shape, device=device, generator=g) if t > 0 else mean
    return x


def v_target(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """Training target for v-prediction: v = sqrt(a) * noise - sqrt(1 - a) * x0.

    At low noise v is nearly the noise; at high noise it is nearly minus the
    clean image. So the network is always asked for the part it cannot just
    copy from its input, and the noisiest steps get a real training signal
    (see finding F1 in the spec). Salimans & Ho 2022.
    """
    a = alpha_bar.to(x0.device, x0.dtype)[t].view(-1, 1, 1, 1)
    return a.sqrt() * noise - (1 - a).sqrt() * x0


def make_eps_fn(model, alpha_bar: torch.Tensor, cls: torch.Tensor, guidance: float = 1.0, pred: str = "eps"):
    """Wrap a class-conditional model as eps_fn(x, t) -> predicted noise, for the samplers.

    Classifier-free guidance: run the model with the label and with the null
    label, then push past the conditional answer, away from the unconditional
    one:  eps = eps_uncond + guidance * (eps_cond - eps_uncond).
    guidance = 1 is plain conditional sampling, 0 ignores the label.
    """
    ab = alpha_bar.to(cls.device, torch.float32)
    null = torch.full_like(cls, model.null_class)

    def eps_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if guidance == 1.0:
            out = model(x, t, cls)
        else:
            # Both passes in one batch: half the calls, and the same trick the browser uses.
            cond, uncond = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cls, null])).chunk(2)
            out = uncond + guidance * (cond - uncond)
        if pred == "v":  # convert v to noise: eps = sqrt(a) * v + sqrt(1 - a) * x_t
            a = ab[t].view(-1, 1, 1, 1)
            out = a.sqrt() * out + (1 - a).sqrt() * x
        return out

    return eps_fn


def ddim_timesteps(steps: int, total: int = T) -> list[int]:
    """`steps` timesteps evenly spaced from total - 1 down to 0, e.g. 20 of the 1000."""
    return torch.linspace(total - 1, 0, steps).round().long().tolist()


@torch.no_grad()
def ddim_sample(eps_fn, x: torch.Tensor, alpha_bar: torch.Tensor, steps: int = 20, return_all: bool = False):
    """Deterministic sampling (DDIM, eta = 0) from starting noise `x` in a handful of steps.

    Each step: guess the clean image, then re-noise that guess to the next,
    lower noise level using the SAME predicted noise rather than fresh noise.
    Nothing random happens after x is chosen, so the starting noise fully
    determines the image, which is what makes seeds shareable.

    With return_all, also returns the x_t and the x0 guess at every step
    (what the web app caches for scrubbing).
    """
    ab = alpha_bar.to(x.device, torch.float32)
    ts = ddim_timesteps(steps, len(ab))
    history = []
    for i, t in enumerate(ts):
        a = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=x.device)
        x0 = predict_x0(x, eps_fn(x, torch.full((len(x),), t, device=x.device)), a)
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt()  # recompute so eps agrees with the clamped x0
        history.append((x, x0))
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return (x, history) if return_all else x


if __name__ == "__main__":
    from pathlib import Path

    from torchvision.utils import save_image

    from data import EmojiDataset

    x0 = EmojiDataset("train", flip=False).x[:1]
    ts = torch.tensor([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999])
    noise = torch.randn(len(ts), *x0.shape[1:], generator=torch.Generator().manual_seed(0))
    rows = []
    print("signal kept, sqrt(alpha_bar), at t =", ts.tolist())
    for schedule in ["linear", "cosine"]:
        ab = make_alpha_bar(schedule)
        assert ab[0] > 0.99 and ab[-1] < 1e-3 and (ab[1:] < ab[:-1]).all()
        print(f"  {schedule:<7}", " ".join(f"{v:.3f}" for v in ab[ts].sqrt().tolist()))
        rows.append(q_sample(x0.expand(len(ts), -1, -1, -1), ts, noise, ab))

    # Sampler check that needs no trained network. For a dataset of a few
    # images the perfect denoiser has a closed form: the average of the
    # training images, weighted by how well each one explains x_t. A correct
    # sampler driven by it must land exactly on training images.
    X = EmojiDataset("train", flip=False).x[:8]
    ab = make_alpha_bar("cosine").float()

    def perfect_denoiser(x, t):
        a = ab[t].view(-1, 1, 1, 1)
        sq_dist = ((x[:, None] - a.sqrt()[:, None] * X[None]) ** 2).flatten(2).sum(-1)  # [B, 8]
        weights = torch.softmax(-sq_dist / (2 * (1 - a.view(-1, 1))), dim=1)
        x0 = torch.einsum("bn,nchw->bchw", weights, X)
        return (x - a.sqrt() * x0) / (1 - a).sqrt()

    samples = ddpm_sample(perfect_denoiser, 32, ab, "cpu")
    dist = torch.cdist(samples.flatten(1), X.flatten(1)).pow(2) / X[0].numel()
    print(f"ddpm_sample with a perfect denoiser: worst distance to a training image {dist.min(1).values.max():.1e}, "
          f"{dist.argmin(1).unique().numel()} of 8 images reached")
    assert dist.min(1).values.max() < 1e-4 and dist.argmin(1).unique().numel() >= 6

    Path("runs").mkdir(exist_ok=True)
    # Noised images go outside [-1, 1]; clamp for viewing only.
    save_image((torch.cat(rows).clamp(-1, 1) + 1) / 2, "runs/forward.png", nrow=len(ts), padding=2)
    print("wrote runs/forward.png (top row linear, bottom row cosine)")
