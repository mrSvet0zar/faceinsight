"""Download the three training datasets into data/ via the Kaggle API.

Requires Kaggle credentials: place kaggle.json in ~/.kaggle/ (or set
KAGGLE_USERNAME / KAGGLE_KEY). See https://www.kaggle.com/docs/api

Usage:
    python -m app.training.download_datasets            # all datasets
    python -m app.training.download_datasets fer2013    # single dataset
"""

import argparse
import sys
import zipfile
from pathlib import Path

from app.config import CELEBA_DIR, DATA_DIR, FER2013_DIR, UTKFACE_DIR

# Kaggle dataset slugs
DATASETS = {
    "fer2013": {
        "slug": "msambare/fer2013",
        "target": FER2013_DIR,
        "expect": "train",  # folder present after extraction if already done
    },
    "utkface": {
        "slug": "jangedoo/utkface-new",
        "target": UTKFACE_DIR,
        "expect": "UTKFace",
    },
    "celeba": {
        "slug": "jessicali9530/celeba-dataset",
        "target": CELEBA_DIR,
        "expect": "img_align_celeba",
    },
}


def download(name: str, force: bool = False) -> None:
    spec = DATASETS[name]
    target: Path = spec["target"]

    if not force and (target / spec["expect"]).exists():
        print(f"[{name}] already present at {target}, skipping (use --force to redo)")
        return

    # Imported lazily: kaggle raises at import time if credentials are missing,
    # and we want the 'already present' fast path to work without them.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    target.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] downloading {spec['slug']} -> {target} ...")
    api.dataset_download_files(spec["slug"], path=str(target), quiet=False)

    for zip_path in target.glob("*.zip"):
        print(f"[{name}] extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        zip_path.unlink()

    print(f"[{name}] done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=[*DATASETS, []],
        help="datasets to download (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    names = args.datasets or list(DATASETS)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        try:
            download(name, force=args.force)
        except Exception as exc:  # noqa: BLE001 — surface which dataset failed, keep going
            print(f"[{name}] FAILED: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
