"""POST /api/analyze/image — analyze one uploaded photo."""

from fastapi import APIRouter, HTTPException, UploadFile

from app.api.deps import decode_image_bytes, get_predictor

router = APIRouter()

MAX_IMAGE_BYTES = 15 * 1024 * 1024


@router.post("/api/analyze/image")
async def analyze_image(file: UploadFile) -> dict:
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image too large (max 15 MB)")
    try:
        image_rgb = decode_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    # Analysis is synchronous CPU work; FastAPI runs sync-heavy code fine here
    # because inference is short (single forward pass on one image).
    result = get_predictor().analyze(image_rgb)
    del image_rgb, data  # privacy: no reference outlives the request
    return result
