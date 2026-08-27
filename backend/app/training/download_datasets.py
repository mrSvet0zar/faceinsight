"""Download the three training datasets into data/ via the Kaggle API.

Credentials are read from backend/.env (KAGGLE_USERNAME + KAGGLE_KEY, with
KAGGLE_API_TOKEN accepted as an alias for the key), or from ~/.kaggle/
kaggle.json. See https://www.kaggle.com/docs/api

Usage:
    python -m app.training.download_datasets            # all datasets
    python -m app.training.download_datasets fer2013    # single dataset
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from app.config import BACKEND_DIR, CELEBA_DIR, DATA_DIR, FER2013_DIR, UTKFACE_DIR


def load_kaggle_credentials() -> None:
    """Populate KAGGLE_USERNAME/KAGGLE_KEY from backend/.env if present."""
    load_dotenv(BACKEND_DIR / ".env")
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token and not os.environ.get("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = token

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
    load_kaggle_credentials()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        try:
            download(name, force=args.force)
        except Exception as exc:  # noqa: BLE001 — surface which dataset failed, keep going
            print(f"[{name}] FAILED: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
