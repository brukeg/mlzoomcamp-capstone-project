FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=-1 \
    PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH="models/cats_dogs.keras" \
    IMG_SIZE="160"

WORKDIR /app

# Deps: libgl1 and libglib2.0-0: for Pillow image decoding on slim images
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     libgl1 \
     libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

# Create venv install deps from lockfile
RUN uv sync --frozen --no-dev

COPY predict.py ./predict.py
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "predict:app", "--host=0.0.0.0", "--port=8000"]
