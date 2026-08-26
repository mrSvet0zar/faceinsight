"""Unit tests for dataset loaders — use tiny synthetic dataset trees in tmp_path,
so they run without downloading anything.
"""

import csv
import random

import pytest
from PIL import Image

from app.config import CELEBA_ATTRS, EMOTION_CLASSES
from app.training.dataset_loaders import (
    CelebADataset,
    FER2013Dataset,
    UTKFaceDataset,
    _deterministic_split,
)


def make_jpg(path, size=(48, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=random.randint(0, 255)).save(path)


# ---------------------------------------------------------------------------
# Deterministic split
# ---------------------------------------------------------------------------
def test_deterministic_split_is_stable_and_disjoint():
    items = [f"img_{i}.jpg" for i in range(100)]
    train = _deterministic_split(items, "train")
    val = _deterministic_split(items, "val")
    test = _deterministic_split(items, "test")
    assert len(train) == 80 and len(val) == 10 and len(test) == 10
    assert set(train) | set(val) | set(test) == set(items)
    assert not (set(train) & set(val)) and not (set(val) & set(test))
    # Same call again -> identical result (no split drift between runs)
    assert _deterministic_split(items, "train") == train


# ---------------------------------------------------------------------------
# FER2013
# ---------------------------------------------------------------------------
@pytest.fixture
def fer_root(tmp_path):
    for folder in ("train", "test"):
        for cls in EMOTION_CLASSES:
            for i in range(3):
                make_jpg(tmp_path / folder / cls / f"{cls}_{i}.jpg")
    return tmp_path


def test_fer2013_loads_and_labels(fer_root):
    ds = FER2013Dataset("test", root=fer_root)
    assert len(ds) == 3 * len(EMOTION_CLASSES)
    image, target = ds[0]
    assert image.mode == "RGB"
    assert 0 <= target["emotion"].item() < len(EMOTION_CLASSES)


def test_fer2013_train_val_disjoint(fer_root):
    train = FER2013Dataset("train", root=fer_root)
    val = FER2013Dataset("val", root=fer_root)
    train_paths = {p for p, _ in train.samples}
    val_paths = {p for p, _ in val.samples}
    assert not train_paths & val_paths
    assert len(train) + len(val) == 3 * len(EMOTION_CLASSES)


# ---------------------------------------------------------------------------
# UTKFace
# ---------------------------------------------------------------------------
@pytest.fixture
def utk_root(tmp_path):
    folder = tmp_path / "UTKFace"
    # Valid files: age_gender_race_timestamp.jpg (race field must be ignored)
    for i, (age, gender) in enumerate([(25, 0), (60, 1), (3, 1), (40, 0)] * 5):
        make_jpg(folder / f"{age}_{gender}_2_2017010{i % 10}.jpg.chip.jpg")
    # Malformed files that must be skipped
    make_jpg(folder / "61_3_20170109150557335.jpg.chip.jpg")  # missing field
    make_jpg(folder / "notaface.jpg")
    return tmp_path


def test_utkface_parses_and_skips_malformed(utk_root):
    sizes = sum(len(UTKFaceDataset(s, root=utk_root)) for s in ("train", "val", "test"))
    assert sizes == 20  # the 2 malformed files are excluded


def test_utkface_targets_have_no_ethnicity(utk_root):
    ds = UTKFaceDataset("train", root=utk_root)
    image, target = ds[0]
    # Only age and gender ever leave the loader (ethical constraint #2)
    assert set(target.keys()) == {"age", "gender"}
    assert target["gender"].item() in (0, 1)
    assert target["age"].item() >= 0
    # Nothing else is stored on the sample tuples either
    assert all(len(s) == 3 for s in ds.samples)  # (path, age, gender)


# ---------------------------------------------------------------------------
# CelebA
# ---------------------------------------------------------------------------
@pytest.fixture
def celeba_root(tmp_path):
    ids = [f"{i:06d}.jpg" for i in range(1, 10)]
    img_dir = tmp_path / "img_align_celeba" / "img_align_celeba"
    for image_id in ids:
        make_jpg(img_dir / image_id, size=(178, 218))

    with open(tmp_path / "list_attr_celeba.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Extra column checks that only CELEBA_ATTRS are loaded
        writer.writerow(["image_id", *CELEBA_ATTRS, "Smiling"])
        for i, image_id in enumerate(ids):
            values = [1 if (i + j) % 2 == 0 else -1 for j in range(len(CELEBA_ATTRS))]
            writer.writerow([image_id, *values, 1])

    with open(tmp_path / "list_eval_partition.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "partition"])
        for i, image_id in enumerate(ids):
            writer.writerow([image_id, 0 if i < 5 else (1 if i < 7 else 2)])
    return tmp_path


def test_celeba_partitions_and_labels(celeba_root):
    train = CelebADataset("train", root=celeba_root)
    val = CelebADataset("val", root=celeba_root)
    test = CelebADataset("test", root=celeba_root)
    assert (len(train), len(val), len(test)) == (5, 2, 2)

    image, target = train[0]
    labels = target["celeba_attrs"]
    assert labels.shape == (len(CELEBA_ATTRS),)
    assert set(labels.unique().tolist()) <= {0.0, 1.0}
