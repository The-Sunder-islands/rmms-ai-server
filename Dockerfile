FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY rmms_ai_server/ ./rmms_ai_server/

RUN pip install --no-cache-dir .

EXPOSE 8420

ENV AI_SERVER_HOST=0.0.0.0
ENV AI_SERVER_PORT=8420
ENV AI_SERVER_LOG_LEVEL=info

VOLUME ["/data/uploads", "/data/output", "/data/models"]

ENV AI_SERVER_UPLOAD_DIR=/data/uploads
ENV AI_SERVER_OUTPUT_DIR=/data/output
ENV AI_SERVER_MODEL_CACHE_DIR=/data/models

CMD ["rmms-ai-server"]
