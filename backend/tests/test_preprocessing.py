"""Unit tests for alignment/normalization — run on synthetic images, no dataset needed."""

import numpy as np
import pytest

from app.config import IMAGE_SIZE
from app.inference.face_detector import DetectedFace
from app.inference.preprocessing import align_face, preprocess_face, to_model_tensor


def make_face(x=100, y=80, w=200, h=200, tilt=0.0) -> DetectedFace:
    """Synthetic detection; tilt shifts the right eye vertically (pixels)."""
    eye_y = y + h * 0.35
    return DetectedFace(
        x=x, y=y, width=w, height=h, confidence=0.9,
        left_eye=(x + w * 0.3, eye_y),
        right_eye=(x + w * 0.7, eye_y + tilt),
    )


@pytest.fixture
def frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


def test_align_face_output_shape(frame):
    aligned = align_face(frame, make_face())
    assert aligned.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert aligned.dtype == np.uint8


def test_align_face_with_tilt(frame):
    aligned = align_face(frame, make_face(tilt=40.0))
    assert aligned.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_align_face_edge_of_frame(frame):
    face = make_face(x=0, y=0, w=100, h=100)
    aligned = align_face(frame, face)
    assert aligned.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_to_model_tensor_shape_and_normalization(frame):
    aligned = align_face(frame, make_face())
    tensor = to_model_tensor(aligned)
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    # ImageNet-normalized values must leave [0, 1]
    assert tensor.min() < 0 or tensor.max() > 1


def test_preprocess_face_end_to_end(frame):
    tensor = preprocess_face(frame, make_face(tilt=15.0))
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype.is_floating_point
