"""Example config: baseline run, paper-default hyperparameters.

Run with:
    python train.py --config configs/config_exp1.py

For experiment 2 / 3, copy this file (e.g. config_exp2.py), change
run_name plus whichever hyperparameters that experiment sweeps (schedule,
T, lr, ema_decay, ...), and point --config at the new file. train.py itself
never needs to change between experiments.

If running on a different account/machine, update `data_root` below to
wherever the 2235-image Cubism folder lives.
"""
from config import TrainConfig

config = TrainConfig(
    run_name="exp1_baseline",
    data_root="/home/noam.sinay/Downloads/Cubism",

    # Diffusion process -- paper defaults (Ho, Jain, Abbeel 2020, Sec. 4)
    T=1000,
    beta_start=1e-4,
    beta_end=0.02,
    schedule="linear",

    # Model / data
    image_size=64,
    base_channels=128,
    channel_mult=(1, 2, 2, 2),
    num_res_blocks=2,
    attn_resolutions=(16,),
    dropout=0.1,

    # Optimization
    lr=2e-4,
    ema_decay=0.9999,
    batch_size=32,

    # Step-based schedule: max_steps is an upper bound, not a target --
    # stop the run early once runs/exp1_baseline/loss_log.csv plateaus.
    max_steps=100_000,
    checkpoint_every=500,
    sample_every=500,
)
