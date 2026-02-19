[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/EKl6m7lv)

# ML Lifecycle & MLOps Assignment — Sentiment Analysis (DVC + MLflow + Docker + CI)

This repository demonstrates an end-to-end ML lifecycle for sentiment analysis:

- **DVC pipeline**: `prepare → train → evaluate`
- **Transfer learning (DistilBERT)** training + **MLflow experiment tracking**
- **Lightweight production API** using **TF-IDF + Logistic Regression** (fast for Docker/CI)
- **Dockerized API** + **GitHub Actions CI**
- Optional: **GHCR deploy workflow** (cloud-deployable without external cloud credentials)

---

## ✅ What’s Implemented

### ML Pipeline (DVC)
Stages:
- `prepare`: creates `data/processed/train.pkl` and `data/processed/test.pkl`
- `train`: fine-tunes DistilBERT, logs metrics to MLflow, saves artifacts to `models/trained/`
- `evaluate`: writes evaluation metrics + confusion matrix artifacts

Commands:
```bash
dvc dag
dvc repro

---
Project Structure
├── app/
│   ├── api.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── models/                      # baseline artifacts for serving (joblib)
├── data/
│   ├── raw/                         # dataset (DVC tracked)
│   └── processed/                   # train.pkl / test.pkl
├── models/
│   └── trained/                     # transformer artifacts (model.safetensors, tokenizer, config)
├── src/
│   ├── data_loader.py
│   ├── model.py                     # transformer training + MLflow logging
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
---