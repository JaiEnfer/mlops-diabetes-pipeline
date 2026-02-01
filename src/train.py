from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def main() -> None:
    Path("models").mkdir(parents=True, exist_ok=True)

    # load data
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #hyperparameter
    alpha =float(os.environ.get("RIDGE_ALPHA", "1.0"))

    #MLflow setup (local folder ./mlruns)
    mlflow.set_experiment("diabetes-ridge")

    with mlflow.start_run():
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        #logs params + metrics
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)


        #Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Done. Rmse = {rmse:.3f} R2= {r2:.3f}")

if __name__ == "__main__":
    main()