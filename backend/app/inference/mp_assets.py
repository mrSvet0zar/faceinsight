"""Download-once cache for MediaPipe Tasks model assets (.tflite / .task)."""

import urllib.request
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "mediapipe"

MODELS = {
    # BlazeFace short-range: the only face-detector model published for the
    # Tasks API. Tuned for faces within ~2m — fine for webcam and datasets.
    "face_detector": (
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    ),
    # Face Landmarker: 478 landmarks including the 10 iris points needed by
    # the eye-color heuristic.
    "face_landmarker": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
    ),
}


def ensure_model(name: str) -> Path:
    """Download the asset on first use (a few MB max) and cache it locally."""
    filename, url = MODELS[name]
    path = _MODEL_DIR / filename
    if not path.exists():
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path
