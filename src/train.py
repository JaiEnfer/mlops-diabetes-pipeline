from __future__ import annotations

from pathlib import Path
import joblib

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_params(path: str = "params.yaml") -> dict:
    """Load training parameters from params.yaml (DVC standard)."""
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if not isinstance(params, dict):
        raise ValueError("params.yaml must contain a YAML mapping/dict.")
    return params


def main() -> None:
    Path("models").mkdir(parents=True, exist_ok=True)

    # Load processed data produced by src/prepare.py
    data_path = "data/processed/diabetes_processed.csv"
    df = pd.read_csv(data_path)

    if "target" not in df.columns:
        raise ValueError("Expected a 'target' column in processed dataset.")

    X = df.drop(columns=["target"])
    y = df["target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Params
    params = load_params()
    alpha = float(params.get("RIDGE_ALPHA", 1.0))

    # MLflow (local folder ./mlruns)
    mlflow.set_experiment("diabetes-ridge")

    with mlflow.start_run():
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        # Log params + metrics
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("data_path", data_path)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model to MLflow artifacts
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        print(f"Done. RMSE={rmse:.3f} R2={r2:.3f}")

        #saving model for serving
        model_path = Path("models/model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

if __name__ == "__main__":
    main()
