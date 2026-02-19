# ML Lifecycle & MLOps Assignment — Sentiment Analysis  
(DVC + MLflow + Docker + GitHub Actions)

This repository demonstrates a complete end-to-end ML lifecycle for sentiment analysis using modern MLOps practices.

## Included Features

- DVC pipeline (`prepare → train → evaluate`)
- Transfer learning with **DistilBERT**
- MLflow experiment tracking
- Lightweight production API (TF-IDF + Logistic Regression)
- Dockerized API (Gunicorn)
- GitHub Actions CI workflows
- GHCR container publishing (cloud-deployable without external credentials)

---

## ML Pipeline (DVC)

### Pipeline Stages

- **prepare** → creates `data/processed/train.pkl` and `test.pkl`
- **train** → fine-tunes DistilBERT and logs metrics to MLflow
- **evaluate** → generates evaluation metrics + confusion matrix

### Run the pipeline

```bash
dvc dag
dvc repro
```

If nothing changed:

```
Data and pipelines are up to date.
```

---

## Experiment Tracking (MLflow)

Training logs:

- model_name
- learning_rate
- batch_size
- epochs
- accuracy
- f1
- precision
- recall

Launch MLflow UI:

```bash
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## Run API Locally (Docker)

Build from project root:

```bash
docker build -f app/Dockerfile -t sentiment-api .
docker run -p 5000:5000 sentiment-api
```

Test endpoints:

```bash
curl http://localhost:5000/health

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"This movie was amazing\"}"
```

---

## GitHub Actions Workflows

### api-test.yml
- Builds Docker image
- Runs container
- Waits for `/health`
- Tests `/predict`

### ml-smoke-test.yml
- Loads baseline joblib artifacts
- Runs test inference

### test.yml
Runs:

```bash
pytest -q
```

### train.yml
- Reproduces DVC pipeline

### deploy.yml
- Builds and publishes Docker image to GHCR
- Uses built-in `GITHUB_TOKEN`

---

## Project Structure

```
.
├── app/
│   ├── api.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── models/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── trained/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── evaluate.py
│   └── inference.py
├── dvc.yaml
├── params.yaml
├── metrics.json
├── eval_metrics.json
└── .github/workflows/
    ├── api-test.yml
    ├── ml-smoke-test.yml
    ├── test.yml
    ├── train.yml
    └── deploy.yml
```

---

## Assignment Requirements Covered

✔ DVC data versioning  
✔ Transformer fine-tuning (DistilBERT)  
✔ MLflow experiment tracking  
✔ Evaluation metrics + confusion matrix  
✔ Flask API with `/health` and `/predict`  
✔ Docker containerization  
✔ GitHub Actions CI  
✔ Cloud-deployable container (GHCR)

---

## Final Status

The pipeline is reproducible, CI passes, Docker builds successfully, and experiments are tracked in MLflow.

This project demonstrates a production-ready ML lifecycle following modern MLOps practices.
