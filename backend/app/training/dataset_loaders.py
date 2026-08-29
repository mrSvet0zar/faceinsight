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
# FER+ — emotion with crowd-sourced relabels (10 annotators per image)
# ---------------------------------------------------------------------------
# Inference feeds the model a MediaPipe-aligned crop widened by CROP_MARGIN
# (35%) around the face box; FER images are tight face crops. Padding them
# with replicated borders at load time closes that train/inference geometry
# gap for the emotion head. ~0.175 of the image per side ≈ the margin the
# aligned crop adds around the detection box.
FER_CONTEXT_PAD_FRACTION = 0.175

# FER+ vote columns, in fer2013new.csv order. unknown/NF must take part in
# the vote so that a mostly-unlabelable image is dropped, not misread as the
# first emotion column.
_FERPLUS_COLS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt", "unknown", "NF",
]
# FER+ emotion name -> our EMOTION_CLASSES index (contempt intentionally
# absent: images whose majority vote is contempt/unknown/NF are dropped to
# keep the 7-class head and API unchanged)
_FERPLUS_TO_CLASS = {
    "neutral": EMOTION_CLASSES.index("neutral"),
    "happiness": EMOTION_CLASSES.index("happy"),
    "surprise": EMOTION_CLASSES.index("surprise"),
    "sadness": EMOTION_CLASSES.index("sad"),
    "anger": EMOTION_CLASSES.index("angry"),
    "disgust": EMOTION_CLASSES.index("disgust"),
    "fear": EMOTION_CLASSES.index("fear"),
}


def _pad_context(image: Image.Image) -> Image.Image:
    """Replicate-pad a tight face crop to match the inference crop geometry."""
    import numpy as np

    pad = int(image.size[0] * FER_CONTEXT_PAD_FRACTION)
    array = np.array(image)
    array = np.pad(array, [(pad, pad), (pad, pad)] + [(0, 0)] * (array.ndim - 2), mode="edge")
    return Image.fromarray(array)


class FERPlusDataset(Dataset):
    """FER2013 images with FER+ majority-vote labels (Barsoum et al. 2016).

    Requires data/fer2013/fer2013.csv (original pixel CSV) and
    data/fer2013/fer2013new.csv (FER+ votes) — see download_datasets.py.
    Usage column gives the official split: Training / PublicTest (val) /
    PrivateTest (test).
    """

    _USAGE = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}

    def __init__(
        self,
        split: str = "train",
        root: Path = FER2013_DIR,
        transform: Optional[Transform] = None,
        pad_context: bool = True,
    ):
        self.transform = transform
        self.pad_context = pad_context
        pixels_csv = root / "fer2013.csv"
        votes_csv = root / "fer2013new.csv"
        if not pixels_csv.exists() or not votes_csv.exists():
            raise FileNotFoundError(
                f"FER+ needs {pixels_csv.name} and {votes_csv.name} in {root} — "
                "run app.training.download_datasets ferplus"
            )

        pixels = pd.read_csv(pixels_csv)
        votes = pd.read_csv(votes_csv)
        assert len(pixels) == len(votes), "fer2013.csv / fer2013new.csv misaligned"

        usage = self._USAGE[split]
        mask = pixels["Usage"] == usage
        vote_matrix = votes.loc[mask, _FERPLUS_COLS].to_numpy()
        winner = vote_matrix.argmax(axis=1)
        winner_name = [_FERPLUS_COLS[i] for i in winner]

        self.samples: list[tuple[str, int]] = []
        for pixel_str, name in zip(pixels.loc[mask, "pixels"], winner_name):
            label = _FERPLUS_TO_CLASS.get(name)
            if label is not None:  # drops contempt-majority images
                self.samples.append((pixel_str, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        import numpy as np

        pixel_str, label = self.samples[i]
        array = np.fromstring(pixel_str, dtype=np.uint8, sep=" ").reshape(48, 48)
        image = Image.fromarray(array).convert("RGB")
        if self.pad_context:
            image = _pad_context(image)
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
