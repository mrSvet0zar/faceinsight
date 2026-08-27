"""Multi-task training: shared ResNet-18 backbone + 5 heads over 3 datasets.

Each dataset labels a subset of tasks (FER2013 -> emotion, UTKFace -> age +
gender, CelebA -> facial hair + hair). Every batch comes from a single
dataset; the loss is computed only on that dataset's heads, weighted by the
per-task loss weights, and batches from the three datasets are interleaved in
random order within an epoch.

Designed for Colab/RunPod sessions that can die mid-run (cf. CLAUDE.md):
- checkpoint written every epoch to --out-dir (point it at Google Drive)
- --resume restarts from the last checkpoint (model, optimizer, scheduler,
  epoch, best metric all restored)
- W&B logging in parallel unless --no-wandb

Local CPU smoke test (a few minutes, no W&B):
    python -m app.training.train --epochs 1 --limit-batches 5 --no-wandb

Real run (Colab GPU):
    python -m app.training.train --epochs 20 --out-dir /content/drive/MyDrive/faceinsight
"""

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from app.config import CHECKPOINTS_DIR, FACIAL_HAIR_ATTRS
from app.models.multitask_model import MultiTaskFaceModel
from app.training.dataset_loaders import CelebADataset, FER2013Dataset, UTKFaceDataset
from app.training.metrics import accuracy, mae, multilabel_f1
from app.training.transforms import eval_transform, train_transform

# Tasks carried by each dataset's batches
DATASET_TASKS = {
    "fer2013": ("emotion",),
    "utkface": ("age", "gender"),
    "celeba": ("facial_hair", "hair"),
}
N_FACIAL_HAIR = len(FACIAL_HAIR_ATTRS)  # split index inside celeba_attrs

# Default per-task loss weights (w1..w5 in CLAUDE.md). Age uses SmoothL1 in
# years (typical early value ~15-30), hence the smaller weight to keep tasks
# on comparable scales. Tune via CLI, compare runs in W&B.
DEFAULT_LOSS_WEIGHTS = {
    "emotion": 1.0,
    "age": 0.1,
    "gender": 1.0,
    "facial_hair": 1.0,
    "hair": 1.0,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataloaders(args) -> tuple[dict, dict]:
    """Return ({name: train_loader}, {name: val_loader})."""
    train_sets = {
        "fer2013": FER2013Dataset("train", transform=train_transform),
        "utkface": UTKFaceDataset("train", transform=train_transform),
        "celeba": CelebADataset("train", transform=train_transform),
    }
    val_sets = {
        "fer2013": FER2013Dataset("val", transform=eval_transform),
        "utkface": UTKFaceDataset("val", transform=eval_transform),
        "celeba": CelebADataset("val", transform=eval_transform),
    }

    def loader(ds, shuffle, sampler=None):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    # CelebA is ~6x larger than the others: subsample a fresh random fraction
    # each epoch so one epoch isn't dominated by hair/facial-hair batches.
    celeba_sampler = RandomSampler(
        train_sets["celeba"],
        replacement=False,
        num_samples=max(args.batch_size, int(len(train_sets["celeba"]) * args.celeba_frac)),
    )
    train_loaders = {
        "fer2013": loader(train_sets["fer2013"], shuffle=True),
        "utkface": loader(train_sets["utkface"], shuffle=True),
        "celeba": loader(train_sets["celeba"], shuffle=False, sampler=celeba_sampler),
    }
    val_loaders = {name: loader(ds, shuffle=False) for name, ds in val_sets.items()}
    return train_loaders, val_loaders


# ---------------------------------------------------------------------------
# Losses (class-imbalance aware — cf. dataset exploration stats)
# ---------------------------------------------------------------------------
def build_losses(train_loaders, device) -> dict[str, nn.Module]:
    # Emotion: inverse-frequency class weights (disgust is ~1.6% of FER2013)
    fer = train_loaders["fer2013"].dataset
    counts = torch.zeros(7)
    for _, label in fer.samples:
        counts[label] += 1
    emotion_weights = (counts.sum() / (len(counts) * counts.clamp(min=1))).to(device)

    # CelebA: pos_weight = neg/pos per label, clamped so ultra-rare attributes
    # (Bald 2.3%) don't blow up the gradient scale.
    prevalence = train_loaders["celeba"].dataset.labels.mean(dim=0)
    pos_weight = ((1 - prevalence) / prevalence.clamp(min=1e-4)).clamp(max=20.0).to(device)

    return {
        # Label smoothing: the run-1 val emotion curve oscillated hard
        # (0.47-0.63) — smoothing damps the overconfident CE spikes.
        "emotion": nn.CrossEntropyLoss(weight=emotion_weights, label_smoothing=0.1),
        "age": nn.SmoothL1Loss(),
        "gender": nn.CrossEntropyLoss(),
        "facial_hair": nn.BCEWithLogitsLoss(pos_weight=pos_weight[:N_FACIAL_HAIR]),
        "hair": nn.BCEWithLogitsLoss(pos_weight=pos_weight[N_FACIAL_HAIR:]),
    }


def batch_losses(outputs, targets, dataset_name, losses) -> dict[str, torch.Tensor]:
    """Losses for the tasks this dataset labels; keys are task names."""
    out = {}
    if dataset_name == "fer2013":
        out["emotion"] = losses["emotion"](outputs["emotion"], targets["emotion"])
    elif dataset_name == "utkface":
        out["age"] = losses["age"](outputs["age"].squeeze(-1), targets["age"])
        out["gender"] = losses["gender"](outputs["gender"], targets["gender"])
    elif dataset_name == "celeba":
        attrs = targets["celeba_attrs"]
        out["facial_hair"] = losses["facial_hair"](
            outputs["facial_hair"], attrs[:, :N_FACIAL_HAIR]
        )
        out["hair"] = losses["hair"](outputs["hair"], attrs[:, N_FACIAL_HAIR:])
    return out


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def train_one_epoch(model, train_loaders, losses, loss_weights, optimizer, device, args, log):
    model.train()
    iterators = {name: iter(dl) for name, dl in train_loaders.items()}
    schedule = [name for name, dl in train_loaders.items() for _ in range(len(dl))]
    random.shuffle(schedule)
    if args.limit_batches:
        schedule = schedule[: args.limit_batches]

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for step, name in enumerate(schedule):
        try:
            images, targets = next(iterators[name])
        except StopIteration:  # only possible with limit_batches reshuffles
            iterators[name] = iter(train_loaders[name])
            images, targets = next(iterators[name])

        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        outputs = model(images)
        per_task = batch_losses(outputs, targets, name, losses)
        loss = sum(loss_weights[t] * l for t, l in per_task.items())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        for t, l in per_task.items():
            totals[t] = totals.get(t, 0.0) + l.item()
            counts[t] = counts.get(t, 0) + 1
        if (step + 1) % args.log_every == 0:
            log({f"train/loss_{t}": totals[t] / counts[t] for t in totals})

    return {f"train/loss_{t}": totals[t] / counts[t] for t in totals}


@torch.no_grad()
def validate(model, val_loaders, losses, loss_weights, device, args):
    model.eval()
    metrics: dict[str, float] = {}
    weighted_loss = 0.0

    for name, dl in val_loaders.items():
        outs: dict[str, list] = {}
        tgts: dict[str, list] = {}
        for i, (images, targets) in enumerate(dl):
            if args.limit_batches and i >= args.limit_batches:
                break
            outputs = model(images.to(device, non_blocking=True))
            for task in DATASET_TASKS[name]:
                outs.setdefault(task, []).append(outputs[task].cpu())
            for key, value in targets.items():
                tgts.setdefault(key, []).append(value)

        if name == "fer2013":
            logits = torch.cat(outs["emotion"])
            labels = torch.cat(tgts["emotion"])
            metrics["val/emotion_acc"] = accuracy(logits, labels)
            weighted_loss += loss_weights["emotion"] * losses["emotion"](
                logits.to(device), labels.to(device)
            ).item()
        elif name == "utkface":
            age_pred = torch.cat(outs["age"])
            gender_logits = torch.cat(outs["gender"])
            age_t = torch.cat(tgts["age"])
            gender_t = torch.cat(tgts["gender"])
            metrics["val/age_mae"] = mae(age_pred, age_t)
            metrics["val/gender_acc"] = accuracy(gender_logits, gender_t)
            weighted_loss += loss_weights["age"] * losses["age"](
                age_pred.squeeze(-1).to(device), age_t.to(device)
            ).item()
            weighted_loss += loss_weights["gender"] * losses["gender"](
                gender_logits.to(device), gender_t.to(device)
            ).item()
        elif name == "celeba":
            attrs = torch.cat(tgts["celeba_attrs"])
            fh_logits = torch.cat(outs["facial_hair"])
            hair_logits = torch.cat(outs["hair"])
            metrics["val/facial_hair_f1"], _ = multilabel_f1(fh_logits, attrs[:, :N_FACIAL_HAIR])
            metrics["val/hair_f1"], _ = multilabel_f1(hair_logits, attrs[:, N_FACIAL_HAIR:])
            weighted_loss += loss_weights["facial_hair"] * losses["facial_hair"](
                fh_logits.to(device), attrs[:, :N_FACIAL_HAIR].to(device)
            ).item()
            weighted_loss += loss_weights["hair"] * losses["hair"](
                hair_logits.to(device), attrs[:, N_FACIAL_HAIR:].to(device)
            ).item()

    metrics["val/weighted_loss"] = weighted_loss
    # Composite selection score (higher is better). Run 1 showed weighted val
    # loss rising from overconfidence while every task metric held or improved,
    # which froze best.pth at epoch 6 — so best is now picked on the metrics
    # we actually report, not on the loss. MAE is scaled to a ~[0,1] range.
    metrics["val/score"] = (
        metrics.get("val/emotion_acc", 0.0)
        + metrics.get("val/gender_acc", 0.0)
        + metrics.get("val/facial_hair_f1", 0.0)
        + metrics.get("val/hair_f1", 0.0)
        - metrics.get("val/age_mae", 0.0) / 20.0
    )
    return metrics


# ---------------------------------------------------------------------------
# Checkpointing (Colab sessions die: save every epoch, resume anytime)
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, best, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best,
            "loss_weights": loss_weights_from_args(args),
        },
        path,
    )


def loss_weights_from_args(args) -> dict[str, float]:
    return {
        "emotion": args.w_emotion,
        "age": args.w_age,
        "gender": args.w_gender,
        "facial_hair": args.w_facial_hair,
        "hair": args.w_hair,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="epochs with frozen backbone (heads-only warmup)")
    parser.add_argument("--celeba-frac", type=float, default=0.25,
                        help="fraction of CelebA sampled per epoch")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=CHECKPOINTS_DIR,
                        help="checkpoint dir (point at Google Drive on Colab)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from <out-dir>/last.pth")
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="cap batches per epoch (CPU smoke test)")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="faceinsight")
    parser.add_argument("--run-name", default=None)
    # Per-task loss weights (w1..w5)
    parser.add_argument("--w-emotion", type=float, default=DEFAULT_LOSS_WEIGHTS["emotion"])
    parser.add_argument("--w-age", type=float, default=DEFAULT_LOSS_WEIGHTS["age"])
    parser.add_argument("--w-gender", type=float, default=DEFAULT_LOSS_WEIGHTS["gender"])
    parser.add_argument("--w-facial-hair", type=float,
                        default=DEFAULT_LOSS_WEIGHTS["facial_hair"])
    parser.add_argument("--w-hair", type=float, default=DEFAULT_LOSS_WEIGHTS["hair"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_weights = loss_weights_from_args(args)
    print(f"device: {device} | loss weights: {loss_weights}")

    train_loaders, val_loaders = build_dataloaders(args)
    losses = build_losses(train_loaders, device)

    model = MultiTaskFaceModel(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.heads.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_score = 0, float("-inf")
    last_path = args.out_dir / "last.pth"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", float("-inf"))
        print(f"resumed from {last_path} at epoch {start_epoch}")

    if args.no_wandb:
        log = lambda d: None  # noqa: E731
    else:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={**vars(args), "out_dir": str(args.out_dir)},
            resume="allow",
        )
        log = wandb.log

    for epoch in range(start_epoch, args.epochs):
        model.freeze_backbone(epoch < args.warmup_epochs)
        train_metrics = train_one_epoch(
            model, train_loaders, losses, loss_weights, optimizer, device, args, log
        )
        val_metrics = validate(model, val_loaders, losses, loss_weights, device, args)
        scheduler.step()

        epoch_log = {"epoch": epoch, **train_metrics, **val_metrics}
        log(epoch_log)
        print(
            f"epoch {epoch:3d} | "
            + " | ".join(f"{k.split('/')[-1]} {v:.4f}" for k, v in val_metrics.items())
        )

        # Update best BEFORE writing last.pth, so a resumed run restores the
        # true best and cannot overwrite best.pth with a worse epoch.
        is_best = val_metrics["val/score"] > best_score
        if is_best:
            best_score = val_metrics["val/score"]
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_score, args)
        if is_best:
            save_checkpoint(
                args.out_dir / "best.pth", model, optimizer, scheduler, epoch, best_score, args
            )
            print(f"  -> new best (val score {best_score:.4f})")

    if not args.no_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
