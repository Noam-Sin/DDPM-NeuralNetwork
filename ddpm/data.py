"""Cubism (WikiArt) dataset loaders.

Two loaders live here:
  - CubismDataset: the full, on-the-fly loader over all 2235 images, used by
    the real training runs (train.py). This is Person 2's original class,
    taken as-is and integrated directly (not an interface built for future
    use) -- see METHODOLOGY.md for the one change made (default path).
  - CubismTiny: an in-memory-cached random subset used only for the quick
    CPU correctness check (quick_validation.py) and for unit tests, so the
    pipeline can be verified without decoding the full dataset every epoch.
"""
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CubismDataset(Dataset):
    """Full Cubism dataset loader (Person 2's class), decoding images from
    disk on every access. Used for real training runs, where the whole
    2235-image set is iterated many times and caching everything in memory
    up front isn't necessary (DataLoader workers parallelize the decode).

    Note: the original default path (~/MyWork/data/cubism_64) pointed at
    Person 2's JupyterHub account; the only change made here is the default
    folder, updated to this machine's copy of the dataset. Pass `folder`
    explicitly (as TrainConfig.data_root does) to override it either way.
    """

    def __init__(self, folder: str = "/home/noam.sinay/Downloads/Cubism", image_size: int = 64):
        folder = os.path.expanduser(folder)
        self.paths = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Lambda(lambda t: t * 2 - 1),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


class CubismTiny(Dataset):
    """Loads a random n_images subset and decodes/resizes them once into an
    in-memory tensor cache. The full-size dataset (2235 images, up to
    ~2000x2000px each) is far too slow to decode on the fly every epoch on
    CPU; since the validation subset + resolution is tiny, caching the
    already-resized tensors (a few MB total) removes that bottleneck."""

    def __init__(self, root: str, image_size: int = 32, n_images: int = 256, seed: int = 0):
        all_files = sorted(f for f in os.listdir(root) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        rng = random.Random(seed)
        rng.shuffle(all_files)
        files = [os.path.join(root, f) for f in all_files[:n_images]]

        # Images -> [-1, 1], matching the paper's data scaling (Sec. 3.3:
        # "assume that image data consists of integers in {0,...,255}
        # scaled linearly to [-1, 1]").
        transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1),
        ])

        tensors = []
        for f in files:
            img = Image.open(f).convert("RGB")
            tensors.append(transform(img))
        self.data = torch.stack(tensors)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for visualization."""
    return (x.clamp(-1, 1) + 1) / 2
