"""Full inference orchestration: detect -> align -> multi-task forward ->
eye-color heuristic -> unified French-labeled response (cf. CLAUDE.md format).

Privacy (constraint #3): frames are numpy arrays in memory from start to
finish; nothing is ever written to disk and no reference outlives the call.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.config import (
    CHECKPOINTS_DIR,
    EMOTION_CLASSES,
    EMOTION_LABELS_FR,
    FACIAL_HAIR_ATTRS,
    GENDER_CLASSES,
    GENDER_LABELS_FR,
    HAIR_ATTRS,
)
from app.inference.face_detector import DetectedFace, FaceDetector
from app.inference.preprocessing import align_face, to_model_tensor
from app.models.eye_color_heuristic import EyeColorEstimator
from app.models.multitask_model import MultiTaskFaceModel

logger = logging.getLogger(__name__)

DISCLAIMER = "Estimations statistiques du modèle, à but démonstratif uniquement."
AGE_RANGE_MARGIN = 4  # value 28 -> "24-32"


def resolve_checkpoint() -> Optional[Path]:
    """Locate the production checkpoint, in priority order:

    1. FACEINSIGHT_CHECKPOINT — explicit local path
    2. FACEINSIGHT_HF_REPO (+ FACEINSIGHT_HF_FILE, default best.pth) —
       downloaded from Hugging Face Hub at startup, cached locally.
       Private repos authenticate via HF_TOKEN (read from backend/.env).
    3. app/models/checkpoints/best.pth — local default

    Returns None when nothing is available (dev mode, untrained heads).
    """
    import os

    explicit = os.environ.get("FACEINSIGHT_CHECKPOINT")
    if explicit:
        return Path(explicit)

    hf_repo = os.environ.get("FACEINSIGHT_HF_REPO")
    if hf_repo:
        from huggingface_hub import hf_hub_download

        filename = os.environ.get("FACEINSIGHT_HF_FILE", "best.pth")
        return Path(hf_hub_download(repo_id=hf_repo, filename=filename))

    default = CHECKPOINTS_DIR / "best.pth"
    return default if default.exists() else None

_HAIR_IDX = {name: i for i, name in enumerate(HAIR_ATTRS)}
_FH_IDX = {name: i for i, name in enumerate(FACIAL_HAIR_ATTRS)}
_HAIR_COLORS_FR = {
    "Black_Hair": "noir",
    "Blond_Hair": "blond",
    "Brown_Hair": "brun",
    "Gray_Hair": "gris",
}


class Predictor:
    """Loads the model once; analyze() is called per image/frame."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint_path = checkpoint_path or resolve_checkpoint()

        self.trained = checkpoint_path is not None and checkpoint_path.exists()
        # Untrained fallback keeps the whole API testable before the first
        # real checkpoint lands; responses are flagged via model_trained.
        self.model = MultiTaskFaceModel(pretrained=not self.trained)
        if self.trained:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint %s (epoch %s)", checkpoint_path, ckpt.get("epoch"))
        else:
            logger.warning(
                "no checkpoint found (looked at %s) — running with UNTRAINED heads (dev mode)",
                checkpoint_path or CHECKPOINTS_DIR / "best.pth",
            )
        self.model.to(self.device).eval()

        self.detector = FaceDetector()
        self.eye_estimator = EyeColorEstimator()

    # ------------------------------------------------------------------
    def analyze(self, image_rgb: np.ndarray) -> dict:
        """Analyze one RGB uint8 image; returns the unified response dict."""
        faces = self.detector.detect(image_rgb)
        results = [self._analyze_face(image_rgb, face) for face in faces]
        return {
            "faces": results,
            "disclaimer": DISCLAIMER,
            "explainability_available": True,
            "model_trained": self.trained,
        }

    @torch.no_grad()
    def _analyze_face(self, image_rgb: np.ndarray, face: DetectedFace) -> dict:
        aligned = align_face(image_rgb, face)
        outputs = self.model(to_model_tensor(aligned).to(self.device))

        emotion_probs = outputs["emotion"].softmax(dim=-1)[0]
        emotion_idx = int(emotion_probs.argmax())
        gender_probs = outputs["gender"].softmax(dim=-1)[0]
        gender_idx = int(gender_probs.argmax())
        age = max(0, int(round(float(outputs["age"][0, 0]))))
        fh = outputs["facial_hair"].sigmoid()[0]
        hair = outputs["hair"].sigmoid()[0]

        eye = self.eye_estimator.estimate(aligned)

        return {
            "bounding_box": face.bounding_box,
            "detection_confidence": round(face.confidence, 4),
            "emotion": {
                "label": EMOTION_LABELS_FR[EMOTION_CLASSES[emotion_idx]],
                "confidence": round(float(emotion_probs[emotion_idx]), 4),
            },
            "age_estimate": {
                "value": age,
                "range": f"{max(0, age - AGE_RANGE_MARGIN)}-{age + AGE_RANGE_MARGIN}",
            },
            "gender": {
                "label": GENDER_LABELS_FR[GENDER_CLASSES[gender_idx]],
                "confidence": round(float(gender_probs[gender_idx]), 4),
            },
            "facial_hair": self._decode_facial_hair(fh),
            "hair": self._decode_hair(hair),
            "eye_color": eye.to_dict() if eye else None,
        }

    @staticmethod
    def _decode_facial_hair(probs: torch.Tensor) -> dict:
        p_beard = 1.0 - float(probs[_FH_IDX["No_Beard"]])
        p_mustache = float(probs[_FH_IDX["Mustache"]])
        # Confidence = how far the decisive probabilities sit from 0.5
        confidence = float(np.mean([abs(p - 0.5) * 2 for p in (p_beard, p_mustache)]))
        return {
            "barbe": p_beard >= 0.5,
            "moustache": p_mustache >= 0.5,
            "confidence": round(confidence, 4),
        }

    @staticmethod
    def _decode_hair(probs: torch.Tensor) -> dict:
        p_bald = float(probs[_HAIR_IDX["Bald"]])
        if p_bald >= 0.5:
            return {"couleur": "chauve", "longueur_estimee": "chauve",
                    "confidence": round(p_bald, 4)}

        color_attr = max(_HAIR_COLORS_FR, key=lambda a: float(probs[_HAIR_IDX[a]]))
        color_conf = float(probs[_HAIR_IDX[color_attr]])

        # CelebA has no direct short/long label (documented limitation): use
        # visible texture attributes as a weak proxy for length.
        p_texture = max(
            float(probs[_HAIR_IDX["Straight_Hair"]]),
            float(probs[_HAIR_IDX["Wavy_Hair"]]),
        )
        length = "mi-long/long" if p_texture >= 0.6 else "court/indéterminé"
        return {
            "couleur": _HAIR_COLORS_FR[color_attr],
            "longueur_estimee": length,
            "confidence": round(color_conf, 4),
        }

    def close(self) -> None:
        self.detector.close()
        self.eye_estimator.close()
