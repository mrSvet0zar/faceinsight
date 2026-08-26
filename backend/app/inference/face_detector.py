"""MediaPipe face detection wrapper (pre-trained, not re-trained).

Returns bounding boxes plus the two eye keypoints needed downstream for
alignment. Works on BGR/RGB numpy frames; nothing is ever written to disk
(privacy constraint CLAUDE.md #3).
"""

from dataclasses import dataclass

import mediapipe as mp
import numpy as np


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
    """Thin wrapper around MediaPipe Face Detection.

    model_selection=1 uses the full-range model (better for webcam frames where
    the face may be further from the camera); 0 is the short-range model.
    """

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 1):
        self._detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_confidence,
            model_selection=model_selection,
        )

    def detect(self, image_rgb: np.ndarray) -> list[DetectedFace]:
        """Detect faces in an RGB uint8 image (H, W, 3)."""
        h, w = image_rgb.shape[:2]
        results = self._detector.process(image_rgb)
        faces: list[DetectedFace] = []
        if not results.detections:
            return faces

        for det in results.detections:
            box = det.location_data.relative_bounding_box
            kps = det.location_data.relative_keypoints
            # MediaPipe keypoint order: right eye (0), left eye (1), nose, mouth,
            # right ear, left ear — "right/left" are the subject's, so keypoint 0
            # is on the viewer's left.
            faces.append(
                DetectedFace(
                    x=max(0, int(box.xmin * w)),
                    y=max(0, int(box.ymin * h)),
                    width=min(int(box.width * w), w),
                    height=min(int(box.height * h), h),
                    confidence=float(det.score[0]),
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
