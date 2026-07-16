exp1_config = {
# ----------------- the exact parameters from the article ------------
    "T": 1000,               
    "beta_start": 1e-4,      
    "beta_end": 0.02,
    "schedule": "linear",
    "lr": 2e-4,               
    "ema_decay": 0.9999,      
 # ----------------- the deviation from the article ------------
    "image_size": 64,         # soft deviation
    "batch_size": 128,       # compromise
    "max_steps": 5000,        # hard deviation
 # ----------------- running management (technical) ------------
    "checkpoint_every": 250,
    "sample_every": 250,
    "run_name": "exp1_baseline",
}