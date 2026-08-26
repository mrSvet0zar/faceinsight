"""Print per-dataset statistics: split sizes, class balance, label prevalence.

Run after download_datasets.py. Output feeds the README (known limits,
class-imbalance discussion) and sanity-checks the loaders.

Usage:
    python -m app.training.explore_datasets
"""

from collections import Counter

from app.config import CELEBA_ATTRS, EMOTION_CLASSES
from app.training.dataset_loaders import (
    CelebADataset,
    FER2013Dataset,
    UTKFaceDataset,
)

SPLITS = ["train", "val", "test"]


def explore_fer2013() -> None:
    print("\n=== FER2013 (emotion) ===")
    for split in SPLITS:
        ds = FER2013Dataset(split)
        counts = Counter(label for _, label in ds.samples)
        dist = ", ".join(
            f"{EMOTION_CLASSES[i]}: {counts.get(i, 0)}" for i in range(len(EMOTION_CLASSES))
        )
        print(f"{split:>5}: {len(ds):6d} images | {dist}")


def explore_utkface() -> None:
    print("\n=== UTKFace (age + gender) ===")
    for split in SPLITS:
        ds = UTKFaceDataset(split)
        ages = [age for _, age, _ in ds.samples]
        genders = Counter(g for _, _, g in ds.samples)
        print(
            f"{split:>5}: {len(ds):6d} images | age mean {sum(ages)/len(ages):5.1f}, "
            f"min {min(ages)}, max {max(ages)} | male {genders[0]}, female {genders[1]}"
        )


def explore_celeba() -> None:
    print("\n=== CelebA (facial hair + hair, multi-label) ===")
    for split in SPLITS:
        ds = CelebADataset(split)
        prevalence = ds.labels.mean(dim=0)
        print(f"{split:>5}: {len(ds):6d} images")
        if split == "train":
            for attr, p in zip(CELEBA_ATTRS, prevalence):
                print(f"        {attr:20s} {p:.1%}")


def main() -> None:
    for fn in (explore_fer2013, explore_utkface, explore_celeba):
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"\n[skipped] {exc}")


if __name__ == "__main__":
    main()
