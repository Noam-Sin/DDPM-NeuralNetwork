"""Quick validation run of the DDPM pipeline on a tiny Cubism subset.

This is the standalone script (formerly train.py) already executed to
produce the results documented in METHODOLOGY.md Section 5 and the
artifacts in outputs/ / train_log.txt -- kept as-is, self-contained, so
those results stay reproducible. It is NOT the production entry point:
that's train.py, the new config-driven pipeline (see config.py,
configs/config_exp1.py) Person 2 uses for the real experiments.

Goal (per assignment): verify the implementation is *correct* -- training
loss decreases, and samples visibly improve over training -- not to produce
high-quality samples. Fixed paper hyperparameters (T, beta schedule, Adam
lr, EMA decay, L_simple loss) are kept exactly as specified; a few
compute-only settings (network width, batch size, image resolution, number
of steps) are scaled down for a CPU-only quick check. See METHODOLOGY.md.
"""
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from unet import UNet
from diffusion import GaussianDiffusion, EMA
from data import CubismTiny, unnormalize

# ---- Fixed, paper-mandated hyperparameters (do not change) ----
T = 1000
BETA_1 = 1e-4
BETA_T = 0.02
LR = 2e-4
EMA_DECAY = 0.9999

# ---- Quick-validation-only settings (compute-driven, see METHODOLOGY.md) ----
IMAGE_SIZE = 32
N_IMAGES = 300
BATCH_SIZE = 8
BASE_CHANNELS = 64
TOTAL_STEPS = 800
SAMPLE_AT_STEPS = [0, 400, 800]
N_SAMPLE_IMAGES = 4
SEED = 0

DATA_ROOT = "/home/noam.sinay/Downloads/Cubism"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cpu"

    print("Loading & caching tiny Cubism subset...")
    t0 = time.time()
    dataset = CubismTiny(DATA_ROOT, image_size=IMAGE_SIZE, n_images=N_IMAGES, seed=SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    data_iter = cycle(loader)
    print(f"  {len(dataset)} images cached in {time.time()-t0:.1f}s")

    model = UNet(
        image_size=IMAGE_SIZE,
        base_channels=BASE_CHANNELS,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    diffusion = GaussianDiffusion(timesteps=T, beta_start=BETA_1, beta_end=BETA_T, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ema = EMA(model, decay=EMA_DECAY)

    losses = []
    step_losses = []  # (step, loss) for smoothed plotting

    def sample_and_save(step, use_ema):
        tag = "ema" if use_ema else "raw"
        eval_model = UNet(
            image_size=IMAGE_SIZE, base_channels=BASE_CHANNELS, channel_mult=(1, 2, 2, 2),
            num_res_blocks=2, attn_resolutions=(16,), dropout=0.1,
        ).to(device)
        eval_model.load_state_dict(ema.shadow if use_ema else model.state_dict())
        eval_model.eval()
        t0 = time.time()
        samples = diffusion.p_sample_loop(eval_model, (N_SAMPLE_IMAGES, 3, IMAGE_SIZE, IMAGE_SIZE), device)
        dt = time.time() - t0
        grid = vutils.make_grid(unnormalize(samples), nrow=N_SAMPLE_IMAGES, padding=2)
        path = os.path.join(OUT_DIR, f"samples_step{step:05d}_{tag}.png")
        vutils.save_image(grid, path)
        print(f"  [sample] step={step} ({tag}) saved to {path}  ({dt:.1f}s for {N_SAMPLE_IMAGES} imgs, {T} steps)")
        return path

    print(f"\nStarting training: {TOTAL_STEPS} steps, batch_size={BATCH_SIZE}, T={T}")
    if 0 in SAMPLE_AT_STEPS:
        print("Sampling at step 0 (untrained model) for baseline comparison...")
        sample_and_save(0, use_ema=False)

    t_train_start = time.time()
    for step in range(1, TOTAL_STEPS + 1):
        x0 = next(data_iter).to(device)
        loss = diffusion.training_loss(model, x0)

        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)

        losses.append(loss.item())
        step_losses.append((step, loss.item()))

        if step % 50 == 0 or step == 1:
            recent = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t_train_start
            print(f"  step {step:4d}/{TOTAL_STEPS}  loss={loss.item():.4f}  avg_last50={recent:.4f}  elapsed={elapsed:.0f}s")

        if step in SAMPLE_AT_STEPS:
            sample_and_save(step, use_ema=False)

    print(f"\nTraining done in {time.time()-t_train_start:.0f}s")

    # loss curve
    steps, vals = zip(*step_losses)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, vals, alpha=0.3, label="per-step loss")
    window = 20
    smoothed = [sum(vals[max(0, i - window):i + 1]) / len(vals[max(0, i - window):i + 1]) for i in range(len(vals))]
    ax.plot(steps, smoothed, label=f"moving avg (window={window})", linewidth=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("L_simple (MSE)")
    ax.set_title("DDPM quick validation: training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=120)
    print(f"Loss curve saved to {os.path.join(OUT_DIR, 'loss_curve.png')}")

    torch.save({"model": model.state_dict(), "ema": ema.shadow}, os.path.join(OUT_DIR, "checkpoint.pt"))
    print("Checkpoint saved.")


if __name__ == "__main__":
    main()
