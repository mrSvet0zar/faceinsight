"""Smoke tests for the multi-task architecture — random tensors, no data, no
pretrained download (pretrained=False keeps CI/offline runs fast).
"""

import torch

from app.models.multitask_model import TASK_DIMS, MultiTaskFaceModel, predictions_from_logits


def test_forward_output_shapes():
    model = MultiTaskFaceModel(pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert set(out.keys()) == set(TASK_DIMS.keys())
    for task, dim in TASK_DIMS.items():
        assert out[task].shape == (2, dim), task


def test_freeze_backbone_only_heads_trainable():
    model = MultiTaskFaceModel(pretrained=False)
    model.freeze_backbone(True)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.heads.parameters())
    model.freeze_backbone(False)
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_predictions_from_logits_decoding():
    model = MultiTaskFaceModel(pretrained=False)
    model.eval()
    with torch.no_grad():
        preds = predictions_from_logits(model(torch.randn(3, 3, 224, 224)))

    assert len(preds["emotion"]["label"]) == 3
    assert all(0.0 <= c <= 1.0 for c in preds["emotion"]["confidence"])
    assert all(g in ("male", "female") for g in preds["gender"]["label"])
    assert len(preds["facial_hair"]["probs"][0]) == TASK_DIMS["facial_hair"]
    assert all(0.0 <= p <= 1.0 for p in preds["hair"]["probs"][0])
