"""
FastAPI inference service for a binary Cats vs Dogs image classifier.

Notes:
- TensorFlow is forced to run on CPU only.
- CUDA and verbose TF logging are explicitly disabled.
"""
import io
import os
from typing import Optional

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Silence TensorFlow noise and force CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


APP_TITLE = "Cats vs Dogs Classifier"
MODEL_PATH = os.getenv("MODEL_PATH", "models/cats_dogs.keras")
IMG_SIZE = int(os.getenv("IMG_SIZE", "160"))

app = FastAPI(title=APP_TITLE)

_model: Optional[tf.keras.Model] = None


def get_model() -> tf.keras.Model:
    """
    Load and cache the TensorFlow Keras model.

    Returns:
        tf.keras.Model: Loaded model instance.

    Raises:
        FileNotFoundError: If the model file does not exist at MODEL_PATH.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. Train it first (train.py) or set MODEL_PATH."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Load an image from raw byte content.

    Args:
        data (bytes): Raw image bytes.

    Returns:
        PIL.Image.Image: Loaded RGB image.

    Raises:
        HTTPException: If the input bytes do not represent a valid image.
    """
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


def preprocess(img: Image.Image) -> np.ndarray:
    """
    Preprocess an image for model inference.

    Steps:
    - Resize to IMG_SIZE x IMG_SIZE
    - Convert to float32
    - Normalize pixel values to [0, 1]
    - Add batch dimension

    Args:
        img (PIL.Image.Image): Input image.

    Returns:
        np.ndarray: Preprocessed image tensor of shape (1, H, W, 3).
    """
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def predict_proba(x: np.ndarray) -> float:
    """
    Run model inference and return the predicted probability.

    Args:
        x (np.ndarray): Preprocessed image tensor.

    Returns:
        float: Probability of the positive class ("dog").

    Notes:
        Assumes a sigmoid output with shape (1, 1) or (1,).
    """
    model = get_model()
    y = model.predict(x, verbose=0)
    p = float(np.ravel(y)[0])
    return p


@app.get("/health")
def health():
    """
    Simple health check endpoint.

    Returns:
        dict: status indicator.
    """
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(None), url: Optional[str] = None):
    """
    Predict whether an image contains a cat or a dog!

    The image can be provided either as:
    - A multipart file upload, OR
    - A publicly accessible image URL

    Exactly one input method must be used.

    Args:
        file (UploadFile, optional): Uploaded image file.
        url (str, optional): URL pointing to an image.

    Returns:
        dict: Prediction result containing:
            - label: "cat" or "dog"
            - probability: Model confidence for "dog"

    Raises:
        HTTPException: For invalid input, download failures, or image errors.
    """
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
