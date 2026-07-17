"""End-to-end smoke test for the production train.py loop: runs a handful
of steps against a tiny synthetic dataset (no real images, no real dataset
folder needed) and checks that checkpoints, EMA weights, sample grids, and
the loss CSV all land where the spec requires."""
import csv
import os
import tempfile
import unittest

import _pathfix  # noqa: F401
import torch
from torch.utils.data import Dataset

from config import TrainConfig
import train


class SyntheticImageDataset(Dataset):
    """Random tensors already in the [-1, 1] range CubismDataset produces,
    so run_training can be exercised without decoding real images."""

    def __init__(self, n=16, image_size=8, channels=3):
        self.data = torch.rand(n, channels, image_size, image_size) * 2 - 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class TestTrainSmoke(unittest.TestCase):
    def _tiny_config(self, output_root, run_name="smoke_run"):
        return TrainConfig(
            run_name=run_name,
            output_root=output_root,
            T=5,
            beta_start=1e-4,
            beta_end=0.02,
            schedule="linear",
            image_size=8,
            base_channels=4,
            channel_mult=(1,),
            num_res_blocks=1,
            attn_resolutions=(),
            dropout=0.0,
            lr=2e-4,
            ema_decay=0.9,
            batch_size=2,
            num_workers=0,
            max_steps=4,
            checkpoint_every=2,
            sample_every=4,
            n_sample_images=2,
            device="cpu",
        )

    def test_full_loop_produces_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._tiny_config(tmp)
            dataset = SyntheticImageDataset(n=8, image_size=config.image_size)

            model, ema = train.run_training(config, dataset=dataset)

            run_dir = os.path.join(tmp, config.run_name)
            checkpoints_dir = os.path.join(run_dir, "checkpoints")
            samples_dir = os.path.join(run_dir, "samples")
            loss_log_path = os.path.join(run_dir, "loss_log.csv")

            # checkpoints saved every checkpoint_every steps, standard state_dict format
            self.assertTrue(os.path.exists(os.path.join(checkpoints_dir, "step_2.pt")))
            self.assertTrue(os.path.exists(os.path.join(checkpoints_dir, "step_4.pt")))
            self.assertTrue(os.path.exists(os.path.join(checkpoints_dir, "step_2_ema.pt")))

            state_dict = torch.load(os.path.join(checkpoints_dir, "step_4.pt"), map_location="cpu")
            fresh_model = train.build_model(config)
            fresh_model.load_state_dict(state_dict)  # must not raise: standard state_dict format

            # sample grid saved at sample_every
            self.assertTrue(os.path.exists(os.path.join(samples_dir, "step_4.png")))

            # loss log: header + one row per step, columns step,loss,elapsed_sec
            with open(loss_log_path, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["step", "loss", "elapsed_sec"])
            self.assertEqual(len(rows) - 1, config.max_steps)
            for i, row in enumerate(rows[1:], start=1):
                self.assertEqual(int(row[0]), i)
                float(row[1])  # loss parses as float
                float(row[2])  # elapsed_sec parses as float

    def test_loss_is_finite_throughout(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._tiny_config(tmp, run_name="finite_check")
            dataset = SyntheticImageDataset(n=8, image_size=config.image_size)
            train.run_training(config, dataset=dataset)

            loss_log_path = os.path.join(tmp, config.run_name, "loss_log.csv")
            with open(loss_log_path, newline="") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                loss = float(row["loss"])
                self.assertTrue(loss == loss)  # not NaN
                self.assertNotEqual(loss, float("inf"))

    def test_cosine_schedule_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._tiny_config(tmp, run_name="cosine_check")
            config.schedule = "cosine"
            dataset = SyntheticImageDataset(n=8, image_size=config.image_size)
            train.run_training(config, dataset=dataset)
            self.assertTrue(os.path.exists(os.path.join(tmp, config.run_name, "loss_log.csv")))


class TestLoadConfig(unittest.TestCase):
    def test_load_config_from_file(self):
        ddpm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(ddpm_dir, "configs", "config_exp1.py")
        config = train.load_config(config_path)
        self.assertIsInstance(config, TrainConfig)
        self.assertEqual(config.T, 1000)
        self.assertEqual(config.schedule, "linear")

    def test_load_config_rejects_file_without_config_var(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = os.path.join(tmp, "bad_config.py")
            with open(bad_path, "w") as f:
                f.write("x = 1\n")
            with self.assertRaises(AttributeError):
                train.load_config(bad_path)


if __name__ == "__main__":
    unittest.main()
