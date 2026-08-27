"""API tests: schema validity, error handling, and the privacy guarantee
(no image ever persisted to disk during a request).

Synthetic inputs keep most tests dataset-free; tests needing a real face use
CelebA and are skipped when data/ is absent.
"""

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import CELEBA_DIR, PROJECT_ROOT
from app.main import app

client = TestClient(app)

CELEBA_IMG_DIR = CELEBA_DIR / "img_align_celeba" / "img_align_celeba"
needs_celeba = pytest.mark.skipif(
    not CELEBA_IMG_DIR.exists(), reason="CelebA not downloaded"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".avi", ".webm", ".gif"}


def noise_jpeg(size=(320, 240)) -> bytes:
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def celeba_jpeg() -> bytes:
    return sorted(CELEBA_IMG_DIR.glob("*.jpg"))[0].read_bytes()


# ---------------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------------
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_image_no_face_valid_schema():
    r = client.post("/api/analyze/image", files={"file": ("x.jpg", noise_jpeg())})
    assert r.status_code == 200
    body = r.json()
    assert body["faces"] == []
    assert "disclaimer" in body and "estimations" in body["disclaimer"].lower()
    assert body["explainability_available"] is True


def test_analyze_image_rejects_garbage():
    r = client.post("/api/analyze/image", files={"file": ("x.jpg", b"not an image")})
    assert r.status_code == 400


@needs_celeba
def test_analyze_image_with_face_full_schema():
    r = client.post("/api/analyze/image", files={"file": ("f.jpg", celeba_jpeg())})
    assert r.status_code == 200
    faces = r.json()["faces"]
    assert len(faces) == 1
    face = faces[0]
    assert set(face["bounding_box"]) == {"x", "y", "width", "height"}
    assert 0.0 <= face["emotion"]["confidence"] <= 1.0
    assert "-" in face["age_estimate"]["range"]
    assert face["gender"]["label"] in ("homme", "femme")
    assert isinstance(face["facial_hair"]["barbe"], bool)
    assert face["hair"]["couleur"] in ("noir", "blond", "brun", "gris", "chauve")
    if face["eye_color"] is not None:
        assert face["eye_color"]["method"] == "heuristique_hsv"
        assert face["eye_color"]["confidence"] == "indicative"


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
@needs_celeba
def test_explain_returns_heatmap():
    payload = {
        "image": base64.b64encode(celeba_jpeg()).decode(),
        "attribute": "emotion",
    }
    r = client.post("/api/analyze/explain", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["attribute"] == "emotion"
    png = base64.b64decode(body["heatmap_overlay_base64"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_explain_rejects_unknown_attribute():
    payload = {"image": base64.b64encode(noise_jpeg()).decode(), "attribute": "age"}
    r = client.post("/api/analyze/explain", json=payload)
    assert r.status_code == 400


def test_explain_no_face_is_422():
    payload = {"image": base64.b64encode(noise_jpeg()).decode(), "attribute": "emotion"}
    r = client.post("/api/analyze/explain", json=payload)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
def synthetic_mp4(seconds: int = 3, fps: int = 8) -> bytes:
    """Tiny in-memory MP4 of noise frames."""
    import av

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height = 128, 96
        stream.pix_fmt = "yuv420p"
        for i in range(seconds * fps):
            # Solid gray frames: random noise can trigger spurious detections
            frame = av.VideoFrame.from_ndarray(
                np.full((96, 128, 3), 60 + i, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


def test_analyze_video_samples_and_aggregates():
    r = client.post("/api/analyze/video", files={"file": ("v.mp4", synthetic_mp4())})
    assert r.status_code == 200
    body = r.json()
    # ~1 frame/s over 3s -> around 3 sampled frames
    assert 2 <= body["aggregate"]["frames_sampled"] <= 4
    assert body["aggregate"]["frames_with_faces"] == 0
    assert body["aggregate"]["dominant_emotion"] is None
    assert "disclaimer" in body


def test_analyze_video_rejects_garbage():
    r = client.post("/api/analyze/video", files={"file": ("v.mp4", b"not a video")})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# WebSocket realtime
# ---------------------------------------------------------------------------
def test_ws_realtime_roundtrip():
    with client.websocket_connect("/ws/realtime") as ws:
        ws.send_text(
            '{"frame": "' + base64.b64encode(noise_jpeg()).decode() + '"}'
        )
        body = ws.receive_json()
        assert body["faces"] == []
        ws.send_text("not json")
        assert "error" in ws.receive_json()


# ---------------------------------------------------------------------------
# Privacy: no image file may persist anywhere after a request (constraint #3)
# ---------------------------------------------------------------------------
def _snapshot_files() -> set[Path]:
    """Image/video files under the project tree and the system temp dir."""
    found: set[Path] = set()
    skip = {".git", ".venv", "node_modules", "data", "wandb", "__pycache__"}
    for base in (PROJECT_ROOT, Path(tempfile.gettempdir())):
        for path in base.rglob("*"):
            if any(part in skip for part in path.parts):
                continue
            if path.suffix.lower() in IMAGE_EXTS:
                found.add(path)
    return found


@needs_celeba
def test_no_image_persisted_after_requests():
    before = _snapshot_files()

    client.post("/api/analyze/image", files={"file": ("f.jpg", celeba_jpeg())})
    client.post(
        "/api/analyze/explain",
        json={"image": base64.b64encode(celeba_jpeg()).decode(), "attribute": "hair"},
    )

    leaked = _snapshot_files() - before
    assert not leaked, f"image files persisted after request: {leaked}"
