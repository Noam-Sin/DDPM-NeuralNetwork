"""Single source of truth for every knob a training run needs.

Person 2 runs her 3 experiments by copying one of the files in configs/
(e.g. configs/config_exp1.py) and editing the values below -- she never
needs to touch unet.py / diffusion.py / train.py to change T, the beta
schedule, lr, EMA decay, image size, batch size, step count, or checkpoint
cadence.

A config file is a plain Python module that defines a module-level
`config = TrainConfig(...)` object; train.py loads it with
`python train.py --config path/to/config_exp1.py`.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TrainConfig:
    # ---- run identity ----
    run_name: str = "baseline"
    data_root: str = "/home/noam.sinay/Downloads/Cubism"
    output_root: str = "runs"

    # ---- diffusion process (Ho, Jain, Abbeel 2020, Sec. 4) ----
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "linear"  # "linear" (paper) or "cosine" (documented deviation, see METHODOLOGY.md)

    # ---- data / model ----
    image_size: int = 64
    image_channels: int = 3
    base_channels: int = 128
    channel_mult: Tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.1

    # ---- optimization ----
    lr: float = 2e-4
    ema_decay: float = 0.9999
    batch_size: int = 32
    num_workers: int = 2

    # ---- schedule / duration (step-based, not epoch-based, per spec) ----
    max_steps: int = 100_000
    checkpoint_every: int = 500
    sample_every: int = 500
    n_sample_images: int = 16

    # ---- misc ----
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
