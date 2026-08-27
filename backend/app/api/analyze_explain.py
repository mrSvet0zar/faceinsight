"""POST /api/analyze/explain — on-demand Grad-CAM heatmap for one attribute.

Separate from the standard analysis endpoint: Grad-CAM needs a backward pass,
so it runs only when the user clicks « Pourquoi ? » on an attribute.
"""

import base64
import binascii
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.deps import decode_image_bytes, get_explainer, get_predictor
from app.inference.explainability import EXPLAINABLE_TASKS
from app.inference.preprocessing import align_face

router = APIRouter()


class ExplainRequest(BaseModel):
    image: str  # base64-encoded JPEG/PNG
    attribute: str  # emotion | gender | facial_hair | hair
    target_label: Optional[str] = None  # optional specific class/attribute


@router.post("/api/analyze/explain")
def analyze_explain(req: ExplainRequest) -> dict:
    if req.attribute not in EXPLAINABLE_TASKS:
        raise HTTPException(
            400, f"attribute must be one of {list(EXPLAINABLE_TASKS)}"
        )
    try:
        image_rgb = decode_image_bytes(base64.b64decode(req.image, validate=True))
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(400, f"invalid image payload: {exc}") from exc

    predictor = get_predictor()
    faces = predictor.detector.detect(image_rgb)
    if not faces:
        raise HTTPException(422, "no face detected in the image")

    # Explain the most confident detection (front-end sends one face at a time)
    face = max(faces, key=lambda f: f.confidence)
    aligned = align_face(image_rgb, face)
    try:
        result = get_explainer().explain(aligned, req.attribute, req.target_label)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    del image_rgb, aligned  # privacy: nothing outlives the request
    return {**result, "model_trained": predictor.trained}
