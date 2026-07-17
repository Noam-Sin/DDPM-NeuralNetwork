from config import TrainConfig

config = TrainConfig(
    run_name="exp3_t100",
    data_root="/projects/nn-bsc/shahar.girtler/DDPM-NeuralNetwork/data/cubism_64",

    T=100,                
    beta_start=1e-4,
    beta_end=0.02,
    schedule="linear",   

    lr=2e-4,
    ema_decay=0.9999,
    image_size=64,
    batch_size=32,
    max_steps=100_000,
    checkpoint_every=250,
    sample_every=250,
    device="cuda",
)