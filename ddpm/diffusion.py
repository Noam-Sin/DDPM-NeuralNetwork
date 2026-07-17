"""
DDPM forward/reverse process, training loss, and sampler, following
Ho, Jain & Abbeel (2020) "Denoising Diffusion Probabilistic Models".

Paper-mandated defaults (Sec. 4 / Appendix B):
  T = 1000
  beta_start = 1e-4, beta_end = 0.02, linear schedule
  L_simple objective (Eq. 14): E_{t,x0,eps} || eps - eps_theta(x_t, t) ||^2

T, beta_start, beta_end and the schedule itself are exposed as constructor
parameters (not hardcoded) so the team can run experiments with different
values via TrainConfig (see config.py). The linear schedule above is what
the paper specifies; `schedule="cosine"` is an optional, documented
deviation (Nichol & Dhariwal, 2021) offered for experimentation only — it
is not used by any paper-faithful run. See METHODOLOGY.md.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Paper's noise schedule (Sec. 4): betas increasing linearly from
    beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal, "Improved Denoising Diffusion
    Probabilistic Models" (2021), Eq. 17. NOT part of Ho et al. (2020) —
    offered only as a configurable, documented deviation for experiments
    that explicitly want to compare schedules (see METHODOLOGY.md)."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-5, max=max_beta)


def make_beta_schedule(schedule: str, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if schedule == "linear":
        return linear_beta_schedule(timesteps, beta_start, beta_end)
    if schedule == "cosine":
        return cosine_beta_schedule(timesteps)
    raise ValueError(f"Unknown schedule '{schedule}', expected 'linear' or 'cosine'")


class GaussianDiffusion:
    """Holds the fixed (non-learned) noise schedule and implements:
      - q_sample: forward diffusion x_0 -> x_t in closed form (Eq. 4).
      - training_loss: the simplified epsilon-prediction MSE loss (Eq. 14).
      - p_sample_loop: the full T-step ancestral sampler (Algorithm 2).
      - sample: convenience wrapper around p_sample_loop for generating a
        batch of images from pure noise.

    T, beta_start, beta_end and schedule are all configurable so the team
    can sweep them across experiments without touching this file.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device="cpu",
    ):
        self.T = timesteps
        self.schedule = schedule
        betas = make_beta_schedule(schedule, timesteps, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        # sigma_t^2 = beta_t, the paper's simpler (and empirically
        # equally good) choice of reverse-process variance (Sec. 3.2).
        self.posterior_variance = betas

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps (Eq. 4)."""
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_omac * noise

    def training_loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """L_simple (Eq. 14): sample t ~ U{1..T}, eps ~ N(0,I), regress the
        model's noise prediction against the true noise."""
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = model(x_t, t)
        return torch.mean((noise - pred_noise) ** 2)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """One reverse step x_t -> x_{t-1} (Algorithm 2, line 4)."""
        b = x_t.shape[0]
        t_batch = torch.full((b,), t, device=x_t.device, dtype=torch.long)
        eps_theta = model(x_t, t_batch)

        beta_t = self.betas[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        sqrt_omac_t = self.sqrt_one_minus_alphas_cumprod[t]

        mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_omac_t) * eps_theta)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(self.posterior_variance[t])
            return mean + sigma_t * noise
        return mean

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape, device, return_trajectory: bool = False):
        """Full ancestral sampler: x_T ~ N(0,I), iterate down to x_0
        (Algorithm 2)."""
        x_t = torch.randn(shape, device=device)
        trajectory = [x_t.clone()] if return_trajectory else None
        for t in reversed(range(self.T)):
            x_t = self.p_sample(model, x_t, t)
            if return_trajectory and (t % (self.T // 10) == 0):
                trajectory.append(x_t.clone())
        if return_trajectory:
            return x_t, trajectory
        return x_t

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int = 16,
        image_size: int = None,
        channels: int = 3,
        device=None,
    ) -> torch.Tensor:
        """Convenience entry point for Algorithm 2: generate `batch_size`
        images from pure Gaussian noise. Returns a tensor of shape
        (batch_size, channels, image_size, image_size) in the model's
        native [-1, 1] range (use data.unnormalize for display/saving)."""
        image_size = image_size if image_size is not None else getattr(model, "image_size", None)
        if image_size is None:
            raise ValueError("image_size must be given (or the model must expose .image_size)")
        device = device if device is not None else next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            return self.p_sample_loop(model, (batch_size, channels, image_size, image_size), device)
        finally:
            model.train(was_training)


class EMA:
    """Exponential moving average of model weights, decay=0.9999 (paper,
    Sec. 4: "We also used EMA on model parameters with a decay factor of
    0.9999.").

    Note (see METHODOLOGY.md): with decay=0.9999 the EMA weights barely
    move away from their initialization over the few hundred/thousand
    optimizer steps used in this quick validation run, so EMA vs. raw
    weights will look nearly identical here. This is expected and is not a
    bug; EMA becomes meaningful once training runs for the tens/hundreds of
    thousands of steps used for the full paper-scale runs.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)
