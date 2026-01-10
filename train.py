# predict.py
import io
import os
from typing import Optional

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Silence TF noise + force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf  # noqa: E402


APP_TITLE = "Cats vs Dogs Classifier"
MODEL_PATH = os.getenv("MODEL_PATH", "models/cats_dogs.keras")
IMG_SIZE = int(os.getenv("IMG_SIZE", "160"))  # keep in sync with training


app = FastAPI(title=APP_TITLE)


_model: Optional[tf.keras.Model] = None


def get_model() -> tf.keras.Model:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. Train it first (train.py) or set MODEL_PATH."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def load_image_from_bytes(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)
    return x


def predict_proba(x: np.ndarray) -> float:
    model = get_model()
    y = model.predict(x, verbose=0)
    # Expecting sigmoid output shape (1, 1) or (1,)
    p = float(np.ravel(y)[0])
    return p


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(None), url: Optional[str] = None):
    if file is None and not url:
        raise HTTPException(status_code=400, detail="Provide either a file upload or a url parameter.")

    if file is not None and url:
        raise HTTPException(status_code=400, detail="Provide only one: file OR url, not both.")

    if url:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; mlzoomcamp-capstone/1.0; +https://github.com/)"
            }
        try:
            r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            r.raise_for_status()
            data = r.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download url: {e}")
    else:
        data = await file.read()

    img = load_image_from_bytes(data)
    x = preprocess(img)
    p = predict_proba(x)

    label = "dog" if p >= 0.5 else "cat"
    return {"label": label, "probability": p}
