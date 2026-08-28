"""Shared singletons: the model stack is loaded once per process."""

import io
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

from app.inference.explainability import Explainer
from app.inference.predictor import Predictor

# Lets PIL open HEIC/HEIF — the default format of smartphone photos
register_heif_opener()


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


@lru_cache(maxsize=1)
def get_explainer() -> Explainer:
    return Explainer(get_predictor().model)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Image bytes -> RGB uint8 array, fully in memory (never touches disk).

    cv2 covers JPEG/PNG/WebP/BMP; PIL (+pillow-heif) covers HEIC/HEIF from
    phones. Raises ValueError on undecodable input.
    """
    buffer = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    except UnidentifiedImageError as exc:
        raise ValueError(
            "image could not be decoded (expected JPEG/PNG/WebP/HEIC)"
        ) from exc
