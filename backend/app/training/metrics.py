"""Per-task metric helpers (pure torch, no sklearn dependency)."""

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for classification logits (N, C) vs targets (N,)."""
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean absolute error for regression (both (N,) or (N, 1))."""
    return (preds.squeeze(-1) - targets.squeeze(-1)).abs().mean().item()


def multilabel_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """Macro F1 over labels for multi-label logits (N, L) vs 0/1 targets.

    Returns (macro_f1, per_label_f1) — per-label scores feed the README's
    known-limits section (rare attributes score lower).
    """
    preds = (logits.sigmoid() >= threshold).float()
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-8)
    return f1.mean().item(), f1.tolist()


def confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Row = true class, column = predicted class."""
    preds = logits.argmax(dim=-1)
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.tolist(), preds.tolist()):
        matrix[t, p] += 1
    return matrix
