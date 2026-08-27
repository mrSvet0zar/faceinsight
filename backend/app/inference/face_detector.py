"""MediaPipe face detection wrapper (pre-trained, not re-trained).

Uses the MediaPipe Tasks API (mediapipe >= 1.0 — the legacy mp.solutions API
was removed in 1.0). The BlazeFace short-range .tflite model is downloaded
once into a local cache on first use.

Returns bounding boxes plus the two eye keypoints needed downstream for
alignment. Works on RGB numpy frames; input frames are never written to disk
(privacy constraint CLAUDE.md #3).
"""

from dataclasses import dataclass

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

from app.inference.mp_assets import ensure_model


@dataclass
class DetectedFace:
    """One detected face, coordinates in pixels of the input image."""

    x: int
    y: int
    width: int
    height: int
    confidence: float
    left_eye: tuple[float, float]  # (x, y) — left eye from viewer's perspective
    right_eye: tuple[float, float]

    @property
    def bounding_box(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


class FaceDetector:
    """Thin wrapper around the MediaPipe Tasks FaceDetector."""

    def __init__(self, min_confidence: float = 0.5):
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(ensure_model("face_detector"))
            ),
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image_rgb: np.ndarray) -> list[DetectedFace]:
        """Detect faces in an RGB uint8 image (H, W, 3)."""
        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)

        faces: list[DetectedFace] = []
        for det in result.detections:
            box = det.bounding_box  # already in pixels in the Tasks API
            # Keypoint order: right eye (0), left eye (1), nose tip, mouth,
            # right ear tragion, left ear tragion — "right/left" are the
            # subject's, so keypoint 0 is on the viewer's left. Keypoints are
            # normalized to [0, 1].
            kps = det.keypoints
            faces.append(
                DetectedFace(
                    x=max(0, box.origin_x),
                    y=max(0, box.origin_y),
                    width=min(box.width, w),
                    height=min(box.height, h),
                    confidence=float(det.categories[0].score),
                    left_eye=(kps[0].x * w, kps[0].y * h),
                    right_eye=(kps[1].x * w, kps[1].y * h),
                )
            )
        return faces

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
