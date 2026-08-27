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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analyze_explain, analyze_image, analyze_video, ws_realtime
from app.api.deps import get_predictor

logging.basicConfig(level=logging.INFO)


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
    version="0.1.0",
    lifespan=lifespan,
)

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
def health() -> dict:
    return {"status": "ok", "model_trained": get_predictor().trained}
