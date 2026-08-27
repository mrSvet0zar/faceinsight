"""Eye color estimation — classical vision heuristic, deliberately NOT ML.

No large, reliably-labeled eye-color dataset exists, so instead of training an
unreliable model we extract the iris region via MediaPipe Face Landmarker
(478 landmarks, indices 468-477 are the two irises) and classify the dominant
HSV color: brown / blue / green / gray / hazel. Documented in the README as
"indicative accuracy, not a learned model" — an explicit architecture decision
(knowing when NOT to use deep learning).

Everything runs in memory on the provided frame; nothing is persisted.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from app.inference.mp_assets import ensure_model

# Face Landmarker iris indices: center + 4 boundary points per eye
_RIGHT_IRIS = [468, 469, 470, 471, 472]  # subject's right (viewer's left)
_LEFT_IRIS = [473, 474, 475, 476, 477]

EYE_COLORS_FR = {
    "brown": "marron",
    "blue": "bleu",
    "green": "vert",
    "gray": "gris",
    "hazel": "noisette",
}


@dataclass
class EyeColorEstimate:
    label: str          # English key, cf. EYE_COLORS_FR for display
    label_fr: str
    method: str = "heuristique_hsv"
    confidence: str = "indicative"  # never a numeric score: this is a heuristic

    def to_dict(self) -> dict:
        return {
            "label": self.label_fr,
            "method": self.method,
            "confidence": self.confidence,
        }


def classify_iris_hsv(h: float, s: float, v: float) -> str:
    """Map a median iris color (OpenCV HSV: H 0-179, S/V 0-255) to a label.

    Thresholds were set empirically on sample portraits; this is intentionally
    a coarse, explainable rule — not a trained decision boundary.
    """
    if s < 45:
        # Desaturated iris: gray if reasonably bright, dark brown otherwise
        return "gray" if v > 90 else "brown"
    if 85 <= h <= 135:
        return "blue"
    if 40 <= h < 85:
        return "green"
    if 20 <= h < 40 and s < 130:
        return "hazel"  # between brown and green, moderately saturated
    return "brown"


class EyeColorEstimator:
    def __init__(self):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(ensure_model("face_landmarker"))
            ),
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def estimate(self, face_crop_rgb: np.ndarray) -> Optional[EyeColorEstimate]:
        """Estimate eye color from an RGB uint8 face crop.

        Returns None when landmarks or usable iris pixels can't be found
        (closed eyes, extreme low light, tiny/blurred crops).
        """
        h, w = face_crop_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        pixels = []
        for iris in (_RIGHT_IRIS, _LEFT_IRIS):
            pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in iris])
            center, boundary = pts[0], pts[1:]
            radius = float(np.linalg.norm(boundary - center, axis=1).mean())
            pixels.append(self._iris_pixels(face_crop_rgb, center, radius))
        pixels = np.concatenate([p for p in pixels if p is not None]) if any(
            p is not None for p in pixels
        ) else None
        if pixels is None or len(pixels) < 10:
            return None

        hsv = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        med_h, med_s, med_v = np.median(hsv, axis=0)
        label = classify_iris_hsv(float(med_h), float(med_s), float(med_v))
        return EyeColorEstimate(label=label, label_fr=EYE_COLORS_FR[label])

    @staticmethod
    def _iris_pixels(
        image_rgb: np.ndarray, center: np.ndarray, radius: float
    ) -> Optional[np.ndarray]:
        """Sample iris pixels inside 0.8*radius, dropping pupil (too dark)
        and specular highlights (too bright)."""
        if radius < 2:
            return None
        h, w = image_rgb.shape[:2]
        y, x = np.ogrid[:h, :w]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= (0.8 * radius) ** 2
        pixels = image_rgb[mask]
        if len(pixels) == 0:
            return None
        brightness = pixels.mean(axis=1)
        keep = (brightness > 35) & (brightness < 220)
        return pixels[keep] if keep.any() else None

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "EyeColorEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
