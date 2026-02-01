![CI](https://github.com/JaiEnfer/mlops-diabetes-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![DVC](https://img.shields.io/badge/DVC-Pipeline-orange)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Model%20Serving-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Tests](https://img.shields.io/badge/Tests-pytest-yellow)

# 🧠 End-to-End MLOps Pipeline with API Deployment

A production-style Machine Learning + MLOps project demonstrating how to build, track, test, and deploy an ML system from data → model → API → CI.
This repository shows how real ML systems are engineered — not just notebooks, but reproducible pipelines, experiment tracking, model serving, and automated testing.

---
🚀 What This Project Does

1. We build a complete ML workflow:
2. Generate dataset programmatically
3. Preprocess data (raw → clean)
4. Train ML model with tracked parameters
5. Log experiments and artifacts
6. Export trained model for serving
7. Serve predictions via a REST API
8. Run automated tests
9. Execute everything in CI on every push

---
🏗️ Architecture Overview

```bash
          ┌──────────────┐
          │ make_dataset │
          └──────┬───────┘
                 ↓
          ┌──────────────┐
          │   prepare    │
          │ raw → clean  │
          └──────┬───────┘
                 ↓
          ┌──────────────┐
          │    train     │
          │ MLflow logs  │
          └──────┬───────┘
                 ↓
        models/model.joblib
                 ↓
          ┌──────────────┐
          │   FastAPI    │
          │  /predict    │
          └──────────────┘
```
---
🧩 Tech Stack

| Area                   | Tools Used         |
| ---------------------- | ------------------ |
| Pipeline Orchestration | **DVC**            |
| Experiment Tracking    | **MLflow**         |
| ML Framework           | scikit-learn       |
| API Framework          | **FastAPI**        |
| Containerization       | **Docker**         |
| Testing                | pytest             |
| CI/CD                  | **GitHub Actions** |


---
📦 Reproducible ML Pipeline

All ML steps are defined in a DVC pipeline:

```sh
dvc repro
```

This command will:
- Generate dataset
- Preprocess data
- Train model
- Log experiments
- Save models/model.joblib
No manual steps. Fully reproducible.

---
🧪 Run Locally

#### 1️⃣ Setup environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### 2️⃣ Reproduce pipeline
```sh
dvc repro
```

#### 3️⃣ Start API
```sh
uvicorn api.app:app --reload
```

Open Swagger UI:

[👉] {http://127.0.0.1:8000/docs}

---

🔮 Example Prediction Request

```sh
POST /predict

{
  "features": {
    "age": 0.05,
    "sex": -0.04,
    "bmi": 0.06,
    "bp": 0.02,
    "s1": -0.04,
    "s2": -0.03,
    "s3": -0.02,
    "s4": -0.01,
    "s5": 0.04,
    "s6": 0.03
  }
}
```
---

🐳 Run with Docker

```sh
docker build -t mlops-api .
docker run -p 8000:8000 mlops-api
```
---

🧪 Run Tests
```sh
pytest -q
```
---

🔁 Continuous Integration

Every push triggers:

✔️ Dependency install

✔️ Full pipeline reproduction

✔️ Model training

✔️ API tests


via GitHub Actions.

---

🎯 Skills Demonstrated

This project showcases practical ML engineering skills:

- Building reproducible ML pipelines
- Data + model versioning
- Experiment tracking
- Model packaging for production
- API-based ML deployment
- Writing tests for ML services
- CI/CD for ML systems
---

___Thank You___

