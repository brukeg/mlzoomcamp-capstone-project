# ML Zoomcamp Capstone – Cats vs Dogs Classifier

## Project Overview

This project implements an end-to-end image classification service that predicts whether an image contains a cat or a dog. It is built to demonstrate core concepts from the second half of the Machine Learning Zoomcamp, including:

- Convolutional neural networks with transfer learning
- Reproducible training using a scripted pipeline
- Serving a trained model via a web API
- Containerization with Docker
- Deployment to a local Kubernetes cluster

The project intentionally focuses on clarity and correctness over scope: a well-understood dataset, a strong baseline model, and a working inference API.

---

## Problem Description

Given an input image (of either a cat or a dog), the system predicts one of two classes:
- cat
- dog

The output includes both the predicted label and the model’s estimated probability for the positive class (`dog`).

This is a **closed-set binary classifier**: every input image is forced into one of the two classes, even if the image is outside the training distribution (e.g., a human or an object). This behavior is expected and documented.

---

## Dataset

- Source: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs?utm_source=chatgpt.com) (`cats_vs_dogs`)

**Notes:**
- Corrupted images are automatically skipped by TFDS
- The dataset is downloaded on first run and cached locally

No manual dataset download is required.

---

## Exploratory Data Analysis (EDA)

EDA and model experiments are documented in `notebooks/EDA.ipynb`.

The analysis includes:
- Visualization of sample images per class
- Class balance inspection
- Image size and aspect-ratio distribution
- Examples of preprocessing and data augmentation
- Comparison of baseline CNN vs. transfer learning performance

### Data Splits

The dataset is split deterministically into:
- 80% training
- 10% validation
- 10% test

All splits are derived from the TFDS `train` split to ensure reproducibility.

---

## Models

### Baseline CNN (from scratch)
- Architecture: Custom convolutional neural network
- Regularization: Dropout (0.3)
- Input size: 160 × 160 RGB images
- Output: Single sigmoid unit (P(dog))
- Training: Trained from scratch (no pretrained weights)

### MobileNetV2
- Architecture: MobileNetV2 (transfer learning)
- Pretrained on: ImageNet
- Input size: 160 × 160 RGB images
- Output: Single sigmoid unit (P(dog))

---

## Training Setup

Model development and selection were performed in a Jupyter notebook (`notebooks/EDA.ipynb`) and then consolidated into a reproducible training script (`train.py`).

Two models were trained and evaluated:

1. **Baseline CNN**
   - Small custom convolutional neural network trained from scratch
   - Used as a performance baseline on the dataset

2. **MobileNetV2 (Transfer Learning)**
   - Pretrained on ImageNet
   - Backbone frozen
   - Lightweight classification head trained on top

Both models were evaluated using accuracy and AUC.  
MobileNetV2 consistently outperformed the baseline CNN while training faster and more stably on CPU, and was selected as the final model.

The trained model is saved to disk and reused by the inference service.

---

## Project Structure

```bash
.
├── Dockerfile                    # Container image for inference service
├── README.md                     # This file
├── k8s                           
│   ├── deployment.yaml           # FastAPI inference service
│   └── service.yaml              # Inference Deployment within the cluster
├── models
│   ├── cats_dogs.keras           # Trained model artifact
│   └── metadata.json             # Model metadata
├── notebooks
│   └── EDA.ipynb                 # EDA + model comparison
├── predict.py                    # FastAPI inference service
├── pyproject.toml                # Dependencies managed with uv
├── train.py                      # Training script (TFDS + MobileNetV2)
├── uv.lock                       # Locked dependency versions
```

## Environment & Dependencies
- Python: 3.12
- Execution environment: Linux (GitHub Codespaces)
- Dependency management: uv
- ML framework: TensorFlow (CPU)
- API framework: FastAPI + Uvicorn

This project intentionally runs inference on CPU only to keep the setup lightweight and reproducible in constrained environments.
**Moreover, it's intended for use within a github Codespace**

To install dependencies:
```bash
uv sync
```

## Training the Model

To train the final selected model (MobileNetV2) from scratch:
```bash
uv run python train.py --epochs 1 --img-size 160 --batch-size 8
```
The training script mirrors the final configuration selected in the notebook
(augmentation, batch size, dropout, early stopping, and data splits).

This will:
1. Download and prepare the Cats vs Dogs dataset (if not already cached)
2. Train the MobileNetV2-based classifier
3. Save the trained model to models/cats_dogs.keras
4. Write training metadata to models/metadata.json

## Running the Inference Service

### Start the API locally:
```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 8000
```
### Health Check
```bash
curl -s http://localhost:8000/health | jq
```
Response:
```json
{"status":"ok"}
```

## Making Predictions
### Predict from an image URL
```bash
curl -s -X POST "http://localhost:8000/predict?url=https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg" | jq
```

Example response:
```json
{
  "label": "cat",
  "probability": 0.00012662613880820572
}
```
### Predict from a local image file
```bash
curl -s -X POST http://localhost:8000/predict -F "file=@cat.jpg" | jq
```

## Docker
### Build the image
```bash
docker build -t cats-dogs-api:latest .
```

### Run the container
```bash
docker run --rm -p 8000:8000 cats-dogs-api:latest
```
The API will be available at http://localhost:8000.

## Kubernetes (Local Cluster with KIND)
This project was deployed to a local Kubernetes cluster using kind (Kubernetes-in-Docker).

### Create the cluster
```bash
kind create cluster --name mlzoomcamp
```

### Load the Docker image into the cluster
```bash
kind load docker-image cats-dogs-api:latest --name mlzoomcamp
```

### Apply manifests
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Port-forward the service
```bash
kubectl port-forward svc/cats-dogs 8000:80
```

### Test the deployed service
#### Health check
```bash
curl http://localhost:8000/health
```

#### Predict
curl -s -X POST \
  "http://localhost:8000/predict?url=https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg" \
  | jq

## Kubernetes Image Loading Note (Important)

Loading large Docker images (e.g., TensorFlow-based images) into a local kind cluster can be slow or unreliable in disk-constrained environments such as GitHub Codespaces.

## Limitations
- This is a binary, closed-set classifier
- Inputs outside the training distribution (e.g., humans, objects) are still classified as either cat or dog
- The reported probability reflects the model’s confidence within this closed set, not real-world certainty
This behavior is expected given the problem framing and training data.