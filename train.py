import argparse
import os

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "house_price_predictor"
MODEL_NAME = "HousePricePredictor"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    # Load data
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Under `mlflow run`, experiment chosen by CLI (--experiment-name); run already created.
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENT_NAME)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("train_samples", len(X_train))

        # Train model
        model = RandomForestRegressor(
            n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        metrics = {
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
        }
        mlflow.log_metrics(metrics)

        # Log + register model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=MODEL_NAME,
        )

        print(f"run_id={mlflow.active_run().info.run_id}")
        print(metrics)


if __name__ == "__main__":
    main()
