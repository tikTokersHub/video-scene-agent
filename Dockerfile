FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -e .

COPY static ./static

RUN mkdir -p data/api_uploads data/uploads chroma_db models

FROM base AS serve

EXPOSE 8000

CMD ["uvicorn", "video_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS demo

EXPOSE 7860

CMD ["python", "-m", "video_agent.ui"]
