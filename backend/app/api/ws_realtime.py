"""WS /ws/realtime — webcam frames in, analysis JSON out.

The frontend sends a frame every ~500ms-1s (not every frame — cf. CLAUDE.md).
Protocol: client sends {"frame": "<base64 JPEG>"}; server replies with the
same JSON as /api/analyze/image, or {"error": ...} for a bad payload.
"""

import base64
import binascii
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.deps import decode_image_bytes, get_predictor

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/realtime")
async def ws_realtime(websocket: WebSocket) -> None:
    await websocket.accept()
    predictor = get_predictor()
    try:
        while True:
            message = await websocket.receive_text()
            try:
                payload = json.loads(message)
                image_rgb = decode_image_bytes(
                    base64.b64decode(payload["frame"], validate=True)
                )
            except (json.JSONDecodeError, KeyError, ValueError, binascii.Error) as exc:
                await websocket.send_json({"error": f"invalid frame payload: {exc}"})
                continue

            result = predictor.analyze(image_rgb)
            del image_rgb  # privacy: frame dropped as soon as analyzed
            await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.debug("realtime client disconnected")
