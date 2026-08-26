"""PyTorch datasets for the three sources feeding the multi-task model.

- FER2013  -> emotion (7-way classification)
- UTKFace  -> age (regression) + gender (binary classification)
- CelebA   -> facial hair + hair attributes (multi-label)

Ethical constraint (CLAUDE.md #2): UTKFace filenames embed an ethnicity field
([age]_[gender]_[race]_[timestamp].jpg). That field is discarded during
filename parsing and never stored on the sample, the dataset, or anywhere else.

Splits are deterministic (SPLIT_SEED) so train/val/test never shift between
runs. CelebA uses its official partition file; FER2013 ships train/test only,
so val is carved out of train; UTKFace has no official split, so we shuffle
once with the fixed seed.
"""

import random
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from app.config import (
    CELEBA_ATTRS,
    CELEBA_DIR,
    EMOTION_CLASSES,
    FER2013_DIR,
    SPLIT_RATIOS,
    SPLIT_SEED,
    UTKFACE_DIR,
)

Transform = Callable[[Image.Image], torch.Tensor]


def _deterministic_split(items: list, split: str) -> list:
    """80/10/10 split of a sorted item list, stable across runs and machines."""
    items = sorted(items)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    if split == "train":
        return items[:n_train]
    if split == "val":
        return items[n_train : n_train + n_val]
    if split == "test":
        return items[n_train + n_val :]
    raise ValueError(f"unknown split: {split!r}")


# ---------------------------------------------------------------------------
# FER2013 — emotion
# ---------------------------------------------------------------------------
class FER2013Dataset(Dataset):
    """Kaggle msambare/fer2013 layout: {train,test}/<class>/<img>.jpg.

    The Kaggle release has no val folder, so val is a deterministic 1/9 slice
    of train (keeping overall ratios close to 80/10/10). Images are 48x48
    grayscale; they are converted to RGB so every task shares one backbone.
    """

    def __init__(
        self,
        split: str = "train",
        root: Path = FER2013_DIR,
        transform: Optional[Transform] = None,
    ):
        self.transform = transform
        folder = root / ("test" if split == "test" else "train")
        if not folder.exists():
            raise FileNotFoundError(
                f"FER2013 not found at {folder} — run app.training.download_datasets"
            )

        samples: list[tuple[str, int]] = []
        for idx, cls in enumerate(EMOTION_CLASSES):
            for img in sorted((folder / cls).glob("*.jpg")):
                samples.append((str(img), idx))

        if split in ("train", "val"):
            rng = random.Random(SPLIT_SEED)
            rng.shuffle(samples)
            n_val = len(samples) // 9
            samples = samples[:n_val] if split == "val" else samples[n_val:]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, {"emotion": torch.tensor(label, dtype=torch.long)}


# ---------------------------------------------------------------------------
# UTKFace — age + gender
# ---------------------------------------------------------------------------
class UTKFaceDataset(Dataset):
    """Kaggle jangedoo/utkface-new layout: UTKFace/<age>_<gender>_<...>.jpg."""

    def __init__(
        self,
        split: str = "train",
        root: Path = UTKFACE_DIR,
        transform: Optional[Transform] = None,
    ):
        self.transform = transform
        folder = root / "UTKFace"
        if not folder.exists():
            raise FileNotFoundError(
                f"UTKFace not found at {folder} — run app.training.download_datasets"
            )

        samples: list[tuple[str, int, int]] = []
        for img in folder.glob("*.jpg"):
            parsed = self._parse_filename(img.name)
            if parsed is not None:
                age, gender = parsed
                samples.append((str(img), age, gender))
        self.samples = _deterministic_split(samples, split)

    @staticmethod
    def _parse_filename(name: str) -> Optional[tuple[int, int]]:
        """Return (age, gender) or None for malformed names.

        The third underscore-separated field (ethnicity) is intentionally
        never read — cf. module docstring.
        """
        parts = name.split("_")
        if len(parts) < 4:  # a handful of UTKFace files are known to be malformed
            return None
        try:
            age, gender = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        if gender not in (0, 1):
            return None
        return age, gender

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, age, gender = self.samples[i]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, {
            "age": torch.tensor(float(age), dtype=torch.float32),
            "gender": torch.tensor(gender, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# CelebA — facial hair + hair (multi-label)
# ---------------------------------------------------------------------------
class CelebADataset(Dataset):
    """Kaggle jessicali9530/celeba-dataset with official partitions.

    Only the columns in CELEBA_ATTRS are loaded; every other CelebA attribute
    is dropped at read time. Values are remapped -1/1 -> 0.0/1.0 for BCE.
    """

    _PARTITIONS = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        split: str = "train",
        root: Path = CELEBA_DIR,
        transform: Optional[Transform] = None,
    ):
        self.transform = transform
        attr_path = root / "list_attr_celeba.csv"
        part_path = root / "list_eval_partition.csv"
        if not attr_path.exists():
            raise FileNotFoundError(
                f"CelebA not found at {root} — run app.training.download_datasets"
            )

        attrs = pd.read_csv(attr_path, usecols=["image_id", *CELEBA_ATTRS])
        parts = pd.read_csv(part_path)
        df = attrs.merge(parts, on="image_id")
        df = df[df["partition"] == self._PARTITIONS[split]]

        self.image_dir = root / "img_align_celeba" / "img_align_celeba"
        if not self.image_dir.exists():  # some extractions are single-level
            self.image_dir = root / "img_align_celeba"

        self.image_ids = df["image_id"].tolist()
        self.labels = torch.tensor(
            (df[CELEBA_ATTRS].to_numpy() > 0).astype("float32")
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, i: int):
        image = Image.open(self.image_dir / self.image_ids[i]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, {"celeba_attrs": self.labels[i]}
