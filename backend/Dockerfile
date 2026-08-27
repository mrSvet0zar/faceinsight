# FaceInsight backend — CPU inference (Railway/Fly.io)
FROM python:3.11-slim

# opencv/mediapipe runtime libs (PyAV wheels bundle their own ffmpeg);
# mediapipe >=1.0 dlopens libEGL/libGLES even for CPU inference
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# CPU-only torch first (much smaller than the default CUDA wheels), then the rest
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Runtime env expected in production:
#   FACEINSIGHT_HF_REPO  — HF Hub repo holding best.pth (downloaded at startup)
#   HF_TOKEN             — only if that repo is private
#   ALLOWED_ORIGINS      — comma-separated frontend origins for CORS
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
