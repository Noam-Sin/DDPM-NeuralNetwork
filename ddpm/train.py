"""Production training entry point.

    python train.py --config configs/config_exp1.py

Everything experiment-specific (T, beta schedule, lr, ema_decay, image_size,
batch_size, max_steps, checkpoint/sample cadence, run_name, data_root) comes
from the TrainConfig object the config file defines -- this file is not
touched between experiments, only the config file is swapped (see
config.py, configs/config_exp1.py).

Per experiment, this produces:
    runs/<run_name>/checkpoints/step_<N>.pt       (model.state_dict())
    runs/<run_name>/checkpoints/step_<N>_ema.pt    (EMA shadow state_dict())
    runs/<run_name>/samples/step_<N>.png           (sample grid, from EMA weights)
    runs/<run_name>/loss_log.csv                   (step, loss, elapsed_sec)

Checkpointing/sampling/logging are all step-based (not epoch-based): the
team doesn't fix a step count up front, they watch loss_log.csv / the
sample grids and stop the run when the loss has visibly stabilized.
"""
import argparse
import csv
import importlib.util
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils

from config import TrainConfig
from unet import UNet
from diffusion import GaussianDiffusion, EMA
from data import CubismDataset, unnormalize


def load_config(path: str) -> TrainConfig:
    """Import a config file (e.g. configs/config_exp1.py) and return the
    TrainConfig object it defines as a module-level `config` variable."""
    spec = importlib.util.spec_from_file_location("run_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "config"):
        raise AttributeError(f"{path} must define a module-level `config = TrainConfig(...)`")
    config = module.config
    if not isinstance(config, TrainConfig):
        raise TypeError(f"{path}: `config` must be a TrainConfig instance, got {type(config)}")
    return config


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def build_model(config: TrainConfig) -> UNet:
    model = UNet(
        image_size=config.image_size,
        image_channels=config.image_channels,
        base_channels=config.base_channels,
        channel_mult=tuple(config.channel_mult),
        num_res_blocks=config.num_res_blocks,
        attn_resolutions=tuple(config.attn_resolutions),
        dropout=config.dropout,
    )
    model.image_size = config.image_size
    return model


def run_training(config: TrainConfig, dataset: Dataset = None):
    """Runs training for config.max_steps steps. `dataset` can be injected
    directly (e.g. a tiny synthetic dataset in tests) to exercise the full
    loop without needing the real Cubism folder on disk."""
    torch.manual_seed(config.seed)
    device = config.resolved_device()

    run_dir = os.path.join(config.output_root, config.run_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    if dataset is None:
        dataset = CubismDataset(folder=config.data_root, image_size=config.image_size)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    data_iter = cycle(loader)

    model = build_model(config).to(device)
    diffusion = GaussianDiffusion(
        timesteps=config.T,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        schedule=config.schedule,
        device=device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    ema = EMA(model, decay=config.ema_decay)

    loss_log_path = os.path.join(run_dir, "loss_log.csv")
    log_file = open(loss_log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "loss", "elapsed_sec"])

    def save_checkpoint(step: int):
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"step_{step}.pt"))
        torch.save(ema.shadow, os.path.join(checkpoints_dir, f"step_{step}_ema.pt"))

    def save_samples(step: int):
        eval_model = build_model(config).to(device)
        eval_model.load_state_dict(ema.shadow)
        eval_model.eval()
        samples = diffusion.sample(eval_model, batch_size=config.n_sample_images, device=device)
        grid = vutils.make_grid(unnormalize(samples), nrow=int(config.n_sample_images ** 0.5) or 1, padding=2)
        vutils.save_image(grid, os.path.join(samples_dir, f"step_{step}.png"))

    t_start = time.time()
    try:
        for step in range(1, config.max_steps + 1):
            x0 = next(data_iter).to(device)
            loss = diffusion.training_loss(model, x0)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model)

            elapsed = time.time() - t_start
            log_writer.writerow([step, loss.item(), elapsed])
            log_file.flush()

            if step % config.checkpoint_every == 0 or step == config.max_steps:
                save_checkpoint(step)
            if step % config.sample_every == 0 or step == config.max_steps:
                save_samples(step)
    finally:
        log_file.close()

    return model, ema


def main():
    parser = argparse.ArgumentParser(description="Train a DDPM on the Cubism dataset.")
    parser.add_argument("--config", required=True, help="Path to a config file defining `config = TrainConfig(...)`")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config '{config.run_name}' from {args.config}")
    run_training(config)


if __name__ == "__main__":
    main()
