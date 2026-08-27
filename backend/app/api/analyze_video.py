"""POST /api/analyze/video — sample ~1 frame/second, analyze each, aggregate.

Decoding uses PyAV on an in-memory buffer: the uploaded video is never
written to disk (privacy constraint #3).
"""

import io
from collections import Counter

import av
from fastapi import APIRouter, HTTPException, UploadFile

from app.api.deps import get_predictor
from app.inference.predictor import DISCLAIMER

router = APIRouter()

MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_SAMPLED_FRAMES = 120  # ~2 minutes at 1 fps
SAMPLE_INTERVAL_S = 1.0


@router.post("/api/analyze/video")
async def analyze_video(file: UploadFile) -> dict:
    data = await file.read()
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(413, "video too large (max 100 MB)")

    predictor = get_predictor()
    frames_out = []
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.video[0]
            next_sample_t = 0.0
            for frame in container.decode(stream):
                t = float(frame.time) if frame.time is not None else 0.0
                if t < next_sample_t:
                    continue
                next_sample_t = t + SAMPLE_INTERVAL_S

                analysis = predictor.analyze(frame.to_ndarray(format="rgb24"))
                frames_out.append({"timestamp_s": round(t, 2), "faces": analysis["faces"]})
                if len(frames_out) >= MAX_SAMPLED_FRAMES:
                    break
    except av.FFmpegError as exc:
        raise HTTPException(400, f"video could not be decoded: {exc}") from exc
    finally:
        del data

    return {
        "frames": frames_out,
        "aggregate": _aggregate(frames_out),
        "disclaimer": DISCLAIMER,
        "explainability_available": False,  # per-frame heatmaps not offered on video
        "model_trained": predictor.trained,
    }


def _aggregate(frames: list[dict]) -> dict:
    """Video-level summary: dominant emotion, mean age, frames with faces."""
    emotions = Counter()
    ages = []
    for f in frames:
        for face in f["faces"]:
            emotions[face["emotion"]["label"]] += 1
            ages.append(face["age_estimate"]["value"])
    return {
        "frames_sampled": len(frames),
        "frames_with_faces": sum(1 for f in frames if f["faces"]),
        "dominant_emotion": emotions.most_common(1)[0][0] if emotions else None,
        "emotion_distribution": dict(emotions),
        "mean_age_estimate": round(sum(ages) / len(ages), 1) if ages else None,
    }
