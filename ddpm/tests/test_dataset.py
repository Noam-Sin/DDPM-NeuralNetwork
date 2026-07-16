import os
import tempfile
import unittest

import _pathfix  # noqa: F401
import torch
from PIL import Image

from data import CubismDataset, CubismTiny, unnormalize


def _make_dummy_images(folder, n=5, size=(40, 30)):
    for i in range(n):
        color = (i * 20 % 256, (i * 50) % 256, (i * 90) % 256)
        Image.new("RGB", size, color=color).save(os.path.join(folder, f"img_{i}.jpg"))


class TestCubismDataset(unittest.TestCase):
    def test_loads_all_images_at_requested_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_dummy_images(tmp, n=5)
            ds = CubismDataset(folder=tmp, image_size=16)
            self.assertEqual(len(ds), 5)
            item = ds[0]
            self.assertEqual(item.shape, (3, 16, 16))

    def test_scaled_to_minus_one_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_dummy_images(tmp, n=2)
            ds = CubismDataset(folder=tmp, image_size=8)
            item = ds[0]
            self.assertGreaterEqual(item.min().item(), -1.0 - 1e-5)
            self.assertLessEqual(item.max().item(), 1.0 + 1e-5)

    def test_non_image_files_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_dummy_images(tmp, n=3)
            with open(os.path.join(tmp, "notes.txt"), "w") as f:
                f.write("not an image")
            ds = CubismDataset(folder=tmp, image_size=8)
            self.assertEqual(len(ds), 3)

    def test_expanduser_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_dummy_images(tmp, n=1)
            ds = CubismDataset(folder=tmp, image_size=8)
            self.assertEqual(len(ds), 1)


class TestCubismTiny(unittest.TestCase):
    def test_caches_requested_subset_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_dummy_images(tmp, n=10)
            ds = CubismTiny(tmp, image_size=8, n_images=4, seed=0)
            self.assertEqual(len(ds), 4)
            self.assertEqual(ds[0].shape, (3, 8, 8))

    def test_seed_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_dummy_images(tmp, n=10)
            ds1 = CubismTiny(tmp, image_size=8, n_images=4, seed=42)
            ds2 = CubismTiny(tmp, image_size=8, n_images=4, seed=42)
            self.assertTrue(torch.equal(ds1.data, ds2.data))


class TestUnnormalize(unittest.TestCase):
    def test_round_trip_range(self):
        x = torch.linspace(-1, 1, steps=10)
        out = unnormalize(x)
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_clamps_out_of_range_values(self):
        x = torch.tensor([-2.0, 2.0])
        out = unnormalize(x)
        self.assertTrue(torch.allclose(out, torch.tensor([0.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
