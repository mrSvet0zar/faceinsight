"""Post-hoc confidence calibration (temperature scaling, Guo et al. 2017).

Fits one temperature per classification head (emotion, gender) on the
validation split, so that a displayed "82%" actually means ~82% empirical
accuracy. Writes calibration.json next to the checkpoint; the predictor
applies it at inference, and upload_checkpoint ships it to HF Hub.

Usage (CPU is fine, ~5 min):
    python -m app.training.calibrate --checkpoint app/models/checkpoints/best.pth
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.config import CHECKPOINTS_DIR
from app.models.multitask_model import MultiTaskFaceModel
from app.training.dataset_loaders import FER2013Dataset, UTKFaceDataset
from app.training.transforms import eval_transform


def expected_calibration_error(
    logits: torch.Tensor, labels: torch.Tensor, bins: int = 15
) -> float:
    """ECE: mean |confidence - accuracy| over confidence bins."""
    probs = logits.softmax(dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()
    ece = torch.zeros(1)
    for lo in torch.linspace(0, 1, bins + 1)[:-1]:
        mask = (conf > lo) & (conf <= lo + 1 / bins)
        if mask.any():
            ece += mask.float().mean() * (correct[mask].mean() - conf[mask].mean()).abs()
    return float(ece)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Scalar T minimizing NLL of softmax(logits / T) on the val split."""
    log_t = torch.zeros(1, requires_grad=True)  # optimize log T for positivity
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp())


@torch.no_grad()
def collect_logits(model, dataset, head: str, label_key: str, batch_size: int):
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    logits, labels = [], []
    for images, targets in dl:
        logits.append(model(images)[head])
        labels.append(targets[label_key])
    return torch.cat(logits), torch.cat(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "best.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = MultiTaskFaceModel(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    calibration: dict[str, float] = {}
    for head, dataset, label_key in (
        ("emotion", FER2013Dataset("val", transform=eval_transform), "emotion"),
        ("gender", UTKFaceDataset("val", transform=eval_transform), "gender"),
    ):
        print(f"[{head}] collecting val logits…")
        logits, labels = collect_logits(model, dataset, head, label_key, args.batch_size)
        before = expected_calibration_error(logits, labels)
        t = fit_temperature(logits, labels)
        after = expected_calibration_error(logits / t, labels)
        calibration[head] = round(t, 4)
        print(f"[{head}] T = {t:.3f} | ECE {before:.4f} -> {after:.4f}")

    out = args.checkpoint.parent / "calibration.json"
    out.write_text(json.dumps(calibration, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
