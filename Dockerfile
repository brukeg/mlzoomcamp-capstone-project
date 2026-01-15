# Dockerfile
FROM python:3.12-slim-bookworm

# Keep Python quieter + deterministic-ish
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=-1 \
    # Put the uv venv on PATH
    PATH="/app/.venv/bin:$PATH" \
    # FastAPI app defaults
    MODEL_PATH="models/cats_dogs.keras" \
    IMG_SIZE="160"

WORKDIR /app

# System deps:
# - libgl1 + libglib2.0-0: often needed for Pillow image decoding on slim images
# - curl: optional but handy for debugging inside container
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     libgl1 \
     libglib2.0-0 \
     curl \
  && rm -rf /var/lib/apt/lists/*

# Install uv (preferred dependency manager per project)
RUN pip install --no-cache-dir uv

# Copy dependency files first to maximize Docker layer caching
COPY pyproject.toml uv.lock ./

# Create venv and install deps exactly from lockfile
# --frozen ensures uv.lock is respected (reproducible)
RUN uv sync --frozen --no-dev

# Copy app code + model artifact
COPY predict.py ./predict.py
COPY models ./models

EXPOSE 8000

# Start the API
CMD ["uvicorn", "predict:app", "--host=0.0.0.0", "--port=8000"]
