"""Final evaluation on the held-out test splits + bias analysis.

Per task: accuracy (emotion, gender), MAE (age), macro/per-label F1 (facial
hair, hair), emotion confusion matrix. Bias analysis (cf. CLAUDE.md): gender
accuracy and age MAE broken down by age bucket, and age MAE by gender, to
surface any disproportionate degradation on a subgroup.

Usage:
    python -m app.training.evaluate --checkpoint path/to/best.pth [--json report.json]
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.config import (
    CHECKPOINTS_DIR,
    EMOTION_CLASSES,
    FACIAL_HAIR_ATTRS,
    GENDER_CLASSES,
    HAIR_ATTRS,
)
from app.models.multitask_model import MultiTaskFaceModel
from app.training.dataset_loaders import CelebADataset, FER2013Dataset, UTKFaceDataset
from app.training.metrics import accuracy, confusion_matrix, mae, multilabel_f1
from app.training.train import N_FACIAL_HAIR
from app.training.transforms import eval_transform

AGE_BUCKETS = [(0, 18), (19, 35), (36, 60), (61, 200)]


@torch.no_grad()
def collect(model, dataset, device, batch_size, num_workers, tasks):
    """Run the test split through the model; return {task: logits}, {key: targets}."""
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    outs: dict[str, list] = {}
    tgts: dict[str, list] = {}
    for images, targets in dl:
        outputs = model(images.to(device, non_blocking=True))
        for task in tasks:
            outs.setdefault(task, []).append(outputs[task].cpu())
        for key, value in targets.items():
            tgts.setdefault(key, []).append(value)
    return (
        {k: torch.cat(v) for k, v in outs.items()},
        {k: torch.cat(v) for k, v in tgts.items()},
    )


def evaluate_emotion(model, device, args, report):
    outs, tgts = collect(
        model, FER2013Dataset("test", transform=eval_transform),
        device, args.batch_size, args.num_workers, ("emotion",),
    )
    logits, labels = outs["emotion"], tgts["emotion"]
    report["emotion"] = {"accuracy": accuracy(logits, labels)}
    matrix = confusion_matrix(logits, labels, len(EMOTION_CLASSES))
    report["emotion"]["confusion_matrix"] = matrix.tolist()

    print(f"\n=== Emotion (FER2013 test) === accuracy: {report['emotion']['accuracy']:.4f}")
    header = " ".join(f"{c[:5]:>6}" for c in EMOTION_CLASSES)
    print(f"{'true\\pred':>10} {header}")
    for i, cls in enumerate(EMOTION_CLASSES):
        row = " ".join(f"{n:6d}" for n in matrix[i].tolist())
        print(f"{cls:>10} {row}")


def evaluate_age_gender(model, device, args, report):
    outs, tgts = collect(
        model, UTKFaceDataset("test", transform=eval_transform),
        device, args.batch_size, args.num_workers, ("age", "gender"),
    )
    age_pred = outs["age"].squeeze(-1)
    age_true = tgts["age"]
    gender_logits, gender_true = outs["gender"], tgts["gender"]

    report["age"] = {"mae": mae(age_pred, age_true)}
    report["gender"] = {"accuracy": accuracy(gender_logits, gender_true)}
    print(f"\n=== Age / gender (UTKFace test) ===")
    print(f"age MAE: {report['age']['mae']:.2f} years | "
          f"gender accuracy: {report['gender']['accuracy']:.4f}")

    # --- Bias analysis: metrics per visible subgroup ---
    print("\n--- Bias analysis (per subgroup) ---")
    by_bucket = {}
    for lo, hi in AGE_BUCKETS:
        mask = (age_true >= lo) & (age_true <= hi)
        if mask.sum() == 0:
            continue
        by_bucket[f"{lo}-{hi}"] = {
            "n": int(mask.sum()),
            "age_mae": mae(age_pred[mask], age_true[mask]),
            "gender_acc": accuracy(gender_logits[mask], gender_true[mask]),
        }
        b = by_bucket[f"{lo}-{hi}"]
        print(f"age {lo:3d}-{hi:<3d} (n={b['n']:5d}): "
              f"age MAE {b['age_mae']:5.2f} | gender acc {b['gender_acc']:.4f}")
    by_gender = {}
    for idx, name in enumerate(GENDER_CLASSES):
        mask = gender_true == idx
        by_gender[name] = {
            "n": int(mask.sum()),
            "age_mae": mae(age_pred[mask], age_true[mask]),
            "gender_acc": accuracy(gender_logits[mask], gender_true[mask]),
        }
        g = by_gender[name]
        print(f"{name:>10} (n={g['n']:5d}): "
              f"age MAE {g['age_mae']:5.2f} | gender acc {g['gender_acc']:.4f}")
    report["bias_analysis"] = {"by_age_bucket": by_bucket, "by_gender": by_gender}


def evaluate_celeba(model, device, args, report):
    outs, tgts = collect(
        model, CelebADataset("test", transform=eval_transform),
        device, args.batch_size, args.num_workers, ("facial_hair", "hair"),
    )
    attrs = tgts["celeba_attrs"]
    print("\n=== Facial hair / hair (CelebA test) ===")
    for task, logits, names, cols in (
        ("facial_hair", outs["facial_hair"], FACIAL_HAIR_ATTRS, slice(None, N_FACIAL_HAIR)),
        ("hair", outs["hair"], HAIR_ATTRS, slice(N_FACIAL_HAIR, None)),
    ):
        macro, per_label = multilabel_f1(logits, attrs[:, cols])
        report[task] = {"macro_f1": macro, "per_label_f1": dict(zip(names, per_label))}
        print(f"{task}: macro F1 {macro:.4f}")
        for attr, f1 in zip(names, per_label):
            print(f"    {attr:20s} F1 {f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "best.pth")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--json", type=Path, default=None, help="write full report as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = MultiTaskFaceModel(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"checkpoint: {args.checkpoint} (epoch {ckpt['epoch']}, "
          f"loss weights {ckpt.get('loss_weights')})")

    report: dict = {"checkpoint": str(args.checkpoint), "epoch": ckpt["epoch"]}
    evaluate_emotion(model, device, args, report)
    evaluate_age_gender(model, device, args, report)
    evaluate_celeba(model, device, args, report)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.json}")


if __name__ == "__main__":
    main()
