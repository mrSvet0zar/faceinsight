"""Upload the validated checkpoint (+ eval report) to Hugging Face Hub.

The production backend then loads it at startup via FACEINSIGHT_HF_REPO
(cf. inference/predictor.resolve_checkpoint). Authentication: HF_TOKEN in
backend/.env (needs write scope), or `huggingface-cli login`.

Usage:
    python -m app.training.upload_checkpoint \
        --checkpoint path/to/best.pth \
        --repo <user>/faceinsight-weights \
        --eval-report ../reports/eval_baseline.json
"""

import argparse
from pathlib import Path

import torch

from app.config import BACKEND_DIR  # noqa: F401 — imports load backend/.env


def slim_checkpoint(path: Path) -> Path:
    """Strip optimizer/scheduler state: inference only needs the weights.

    Roughly halves the uploaded size; written next to the source file.
    """
    ckpt = torch.load(path, map_location="cpu")
    slim = {k: ckpt[k] for k in ("model", "epoch", "loss_weights") if k in ckpt}
    out = path.with_name(path.stem + "-slim.pth")
    torch.save(slim, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo", required=True, help="e.g. <user>/faceinsight-weights")
    parser.add_argument("--eval-report", type=Path, default=None,
                        help="JSON report to upload alongside the weights")
    parser.add_argument("--public", action="store_true",
                        help="create the repo public (default: private)")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, private=not args.public, exist_ok=True)

    slim = slim_checkpoint(args.checkpoint)
    print(f"uploading {slim.name} ({slim.stat().st_size / 1e6:.1f} MB) -> {args.repo}")
    api.upload_file(
        path_or_fileobj=str(slim), path_in_repo="best.pth", repo_id=args.repo
    )
    if args.eval_report:
        api.upload_file(
            path_or_fileobj=str(args.eval_report),
            path_in_repo="eval_report.json",
            repo_id=args.repo,
        )
    calibration = args.checkpoint.parent / "calibration.json"
    if calibration.exists():
        api.upload_file(
            path_or_fileobj=str(calibration),
            path_in_repo="calibration.json",
            repo_id=args.repo,
        )
        print("calibration.json uploaded")
    print(f"done — set FACEINSIGHT_HF_REPO={args.repo} on the backend")


if __name__ == "__main__":
    main()
