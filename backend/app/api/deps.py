"""Shared singletons: the model stack is loaded once per process."""

from functools import lru_cache

import cv2
import numpy as np

from app.inference.explainability import Explainer
from app.inference.predictor import Predictor


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


@lru_cache(maxsize=1)
def get_explainer() -> Explainer:
    return Explainer(get_predictor().model)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> RGB uint8 array, fully in memory (never touches disk).

    Raises ValueError on undecodable input.
    """
    buffer = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("image could not be decoded (expected JPEG/PNG/WebP)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
