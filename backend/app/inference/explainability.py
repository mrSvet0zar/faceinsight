"""Grad-CAM explainability on the shared backbone (pytorch-grad-cam wrapper).

For a requested head (emotion, gender, facial_hair, hair) — age regression is
excluded — computes which face regions drove the prediction, and returns the
heatmap blended over the aligned face crop. Computed on demand only (separate
endpoint): Grad-CAM costs a backward pass, too heavy for every frame.
"""

import base64
from typing import Optional

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from app.config import (
    EMOTION_CLASSES,
    EMOTION_LABELS_FR,
    FACIAL_HAIR_ATTRS,
    GENDER_CLASSES,
    GENDER_LABELS_FR,
    HAIR_ATTRS,
)
from app.inference.preprocessing import to_model_tensor
from app.models.multitask_model import MultiTaskFaceModel

EXPLAINABLE_TASKS = ("emotion", "gender", "facial_hair", "hair")

EXPLANATION_NOTE = "Zones du visage ayant le plus influencé cette prédiction"


class _HeadModel(nn.Module):
    """Adapter: GradCAM expects a model returning a single logits tensor."""

    def __init__(self, model: MultiTaskFaceModel, task: str):
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[self.task]


def _predicted_label(task: str, logits: torch.Tensor, target_idx: int) -> str:
    if task == "emotion":
        return EMOTION_LABELS_FR[EMOTION_CLASSES[target_idx]]
    if task == "gender":
        return GENDER_LABELS_FR[GENDER_CLASSES[target_idx]]
    if task == "facial_hair":
        return FACIAL_HAIR_ATTRS[target_idx]
    return HAIR_ATTRS[target_idx]


class Explainer:
    """Grad-CAM on the last conv block of the shared backbone."""

    def __init__(self, model: MultiTaskFaceModel):
        self.model = model

    def explain(
        self,
        aligned_rgb: np.ndarray,
        task: str,
        target_label: Optional[str] = None,
    ) -> dict:
        """Return predicted label + base64 PNG of the heatmap overlay.

        aligned_rgb: the 224x224 aligned face crop (uint8).
        target_label: optional attribute name (facial_hair/hair heads) or
        class name to explain instead of the head's top prediction.
        """
        if task not in EXPLAINABLE_TASKS:
            raise ValueError(
                f"attribute must be one of {EXPLAINABLE_TASKS} (got {task!r})"
            )

        head_model = _HeadModel(self.model, task)
        tensor = to_model_tensor(aligned_rgb)

        with torch.no_grad():
            logits = head_model(tensor)[0]
        if target_label is not None:
            names = {
                "emotion": EMOTION_CLASSES,
                "gender": GENDER_CLASSES,
                "facial_hair": FACIAL_HAIR_ATTRS,
                "hair": HAIR_ATTRS,
            }[task]
            if target_label not in names:
                raise ValueError(f"unknown label {target_label!r} for task {task!r}")
            target_idx = names.index(target_label)
        else:
            target_idx = int(logits.argmax())

        # GradCAM context re-enables gradients for the backward pass
        with GradCAM(
            model=head_model, target_layers=[self.model.backbone.layer4[-1]]
        ) as cam:
            grayscale = cam(
                input_tensor=tensor, targets=[ClassifierOutputTarget(target_idx)]
            )[0]

        overlay = show_cam_on_image(
            aligned_rgb.astype(np.float32) / 255.0, grayscale, use_rgb=True
        )
        ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("PNG encoding failed")

        return {
            "attribute": task,
            "predicted_label": _predicted_label(task, logits, target_idx),
            "heatmap_overlay_base64": base64.b64encode(png.tobytes()).decode(),
            "explanation_note": EXPLANATION_NOTE,
        }
