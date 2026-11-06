# Experimental Linux x86-64 container: core app and remote/cloud providers.
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/tts \
    XDG_CACHE_HOME=/home/tts/.cache \
    HF_HOME=/home/tts/.cache/huggingface \
    TTS_STORY_SKIP_BROWSER=1 \
    TTS_STORY_PRODUCTION=1 \
    TTS_STORY_CONFIG_FILE=/app/data/config.json

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg rubberband-cli sox libsox-fmt-all \
      espeak-ng libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 tts \
    && useradd --uid 10001 --gid 10001 --home-dir /home/tts --create-home tts

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir 'waitress>=3.0,<4'

COPY --chown=tts:tts . .
RUN mkdir -p data static/audio static/samples models /home/tts/.cache \
    && chown -R tts:tts data static/audio static/samples models /home/tts

USER tts
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/', timeout=4)" || exit 1

CMD ["python", "app.py"]

# Optional NVIDIA CUDA runtime. This supplies CUDA-enabled PyTorch in the
# application environment; individual isolated TTS engines must still be
# installed and verified separately.
FROM runtime AS cuda
USER root
RUN python -m pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
USER tts

# Keep CPU as the default target for a plain 'docker build'.
FROM runtime AS cpu
