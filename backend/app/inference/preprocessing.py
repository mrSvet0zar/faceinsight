"""Face alignment and normalization ahead of the multi-task model.

Pipeline: rotate so the eyes are horizontal, crop around the face with a
margin, resize to IMAGE_SIZE, then (for the model path) convert to a
normalized tensor. Everything stays in memory — no intermediate files.
"""

import math

import cv2
import numpy as np
import torch
from torchvision import transforms

from app.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from app.inference.face_detector import DetectedFace

# Margin added around the detected box before cropping: MediaPipe boxes are
# tight around the face, and the hair/beard heads need context beyond it.
CROP_MARGIN = 0.35

_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def align_face(image_rgb: np.ndarray, face: DetectedFace) -> np.ndarray:
    """Return an eye-aligned IMAGE_SIZE x IMAGE_SIZE RGB crop of the face."""
    (lx, ly), (rx, ry) = face.left_eye, face.right_eye
    angle = math.degrees(math.atan2(ry - ly, rx - lx))

    # Rotate the full frame around the eye midpoint so eyes end up horizontal.
    center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    h, w = image_rgb.shape[:2]
    rotated = cv2.warpAffine(
        image_rgb, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    # Rotate the box corners with the same matrix, then take their new bounds.
    corners = np.array(
        [
            [face.x, face.y, 1],
            [face.x + face.width, face.y, 1],
            [face.x, face.y + face.height, 1],
            [face.x + face.width, face.y + face.height, 1],
        ],
        dtype=np.float64,
    )
    moved = corners @ rot.T
    x0, y0 = moved.min(axis=0)
    x1, y1 = moved.max(axis=0)

    # Expand to a square with margin, clamped to the frame.
    side = max(x1 - x0, y1 - y0) * (1 + CROP_MARGIN)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(w, cx + side / 2))
    y1 = int(min(h, cy + side / 2))

    crop = rotated[y0:y1, x0:x1]
    if crop.size == 0:  # degenerate box (face on the very edge of the frame)
        crop = rotated
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def to_model_tensor(aligned_rgb: np.ndarray) -> torch.Tensor:
    """Aligned RGB uint8 crop -> normalized (1, 3, H, W) float tensor."""
    tensor = _to_tensor(aligned_rgb.copy())
    return tensor.unsqueeze(0)


def preprocess_face(image_rgb: np.ndarray, face: DetectedFace) -> torch.Tensor:
    """Full path: align + normalize, ready for the shared backbone."""
    return to_model_tensor(align_face(image_rgb, face))
