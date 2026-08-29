"""FaceInsight API — FastAPI entry point.

Run locally:
    uvicorn app.main:app --reload

The model stack (multi-task checkpoint, MediaPipe detector, eye-color
landmarker) is loaded once at startup. Without a checkpoint in
app/models/checkpoints/best.pth the API still runs, flagged
model_trained=false in every response (dev mode).
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.api import analyze_explain, analyze_image, analyze_video, ws_realtime
from app.api.deps import get_predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faceinsight.access")

# Per-IP rate limiting, in-memory and transient — nothing is logged or
# persisted per client (privacy constraint #6). WS realtime is throttled
# client-side (~1 frame/700ms).
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = get_predictor()  # eager load: fail fast, no first-request lag
    yield
    predictor.close()


app = FastAPI(
    title="FaceInsight API",
    description=(
        "Analyse faciale multi-attributs (émotion, âge, genre, pilosité, "
        "cheveux, couleur des yeux). Aucune identification de personnes, "
        "aucune image conservée après traitement."
    ),
    version="0.2.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def access_metrics(request: Request, call_next):
    """Anonymous latency logging: endpoint + status + duration, never who."""
    start = time.perf_counter()
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        logger.info(
            "%s %s %s %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
    return response

# CORS: open in dev; set ALLOWED_ORIGINS=https://<frontend>.vercel.app in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_image.router)
app.include_router(analyze_video.router)
app.include_router(analyze_explain.router)
app.include_router(ws_realtime.router)


@app.get("/api/health")
@limiter.exempt
def health() -> dict:
    predictor = get_predictor()
    return {
        "status": "ok",
        "model_trained": predictor.trained,
        "model_epoch": predictor.epoch,
        "calibrated": bool(predictor.calibration),
        "api_version": app.version,
    }
