"""Multi-task model: shared ResNet-18 backbone + one head per attribute task.

One forward pass yields every prediction (emotion, age, gender, facial hair,
hair). Eye color is NOT part of this model — it uses a classical heuristic
(see eye_color_heuristic.py, Phase 2).

Heads:
- emotion:     7-way softmax (cross-entropy)
- age:         scalar regression (MSE / L1 on years)
- gender:      2-way softmax (cross-entropy) — presented as a statistical
               estimate with confidence, never as a fact (constraint #5)
- facial_hair: 5 independent sigmoids (BCE with logits)
- hair:        8 independent sigmoids (BCE with logits)
"""

import torch
from torch import nn
from torchvision import models

from app.config import (
    EMOTION_CLASSES,
    FACIAL_HAIR_ATTRS,
    GENDER_CLASSES,
    HAIR_ATTRS,
)

# Task name -> output dimension
TASK_DIMS = {
    "emotion": len(EMOTION_CLASSES),      # 7
    "age": 1,
    "gender": len(GENDER_CLASSES),        # 2
    "facial_hair": len(FACIAL_HAIR_ATTRS),  # 5
    "hair": len(HAIR_ATTRS),              # 8
}


class MultiTaskFaceModel(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        def head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, out_dim),
            )

        self.heads = nn.ModuleDict({task: head(dim) for task, dim in TASK_DIMS.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, 3, 224, 224) -> {task: logits} (age: raw scalar in years)."""
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze/unfreeze the backbone for two-stage fine-tuning
        (heads-only warmup first, then full fine-tune at low LR)."""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze


def predictions_from_logits(outputs: dict[str, torch.Tensor]) -> dict:
    """Turn raw logits into human-readable predictions with confidences.

    Used by the inference path (Phase 3); kept next to the model so the
    logit semantics and their decoding never drift apart.
    """
    emotion_probs = outputs["emotion"].softmax(dim=-1)
    emotion_conf, emotion_idx = emotion_probs.max(dim=-1)
    gender_probs = outputs["gender"].softmax(dim=-1)
    gender_conf, gender_idx = gender_probs.max(dim=-1)

    return {
        "emotion": {
            "label": [EMOTION_CLASSES[i] for i in emotion_idx.tolist()],
            "confidence": emotion_conf.tolist(),
        },
        "age": {"value": outputs["age"].squeeze(-1).round().int().tolist()},
        "gender": {
            "label": [GENDER_CLASSES[i] for i in gender_idx.tolist()],
            "confidence": gender_conf.tolist(),
        },
        "facial_hair": {
            "attrs": FACIAL_HAIR_ATTRS,
            "probs": outputs["facial_hair"].sigmoid().tolist(),
        },
        "hair": {
            "attrs": HAIR_ATTRS,
            "probs": outputs["hair"].sigmoid().tolist(),
        },
    }
