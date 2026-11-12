# House Price Predictor — MLflow hands-on

```bash
poetry install
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# 1. UI (separate terminal)
poetry run mlflow ui --port 5001   # 5000 taken by macOS AirPlay

# 2. Experiments (each run also registers model HousePricePredictor)
poetry run mlflow run . --env-manager local --experiment-name house_price_predictor -P n_estimators=50
poetry run mlflow run . --env-manager local --experiment-name house_price_predictor -P n_estimators=100
poetry run mlflow run . --env-manager local --experiment-name house_price_predictor -P n_estimators=200
poetry run mlflow run . --env-manager local --experiment-name house_price_predictor -P test_size=0.1

# 3. Registry: best run -> aliases staging/production
poetry run python registry.py

# 4. Serve + test
poetry run mlflow models serve -m "models:/HousePricePredictor@production" -p 1234 --env-manager local
poetry run python deploy.py
```

Notes vs slides:
- Stages (`Staging`/`Production`) deprecated in MLflow 2.9+; aliases used.
- No conda. `--env-manager local` runs `mlflow run` in the Poetry venv; deps live in `pyproject.toml`.
- Pins: `sqlalchemy<2.1`, `starlette<1.0` needed for mlflow 2.22 (else import/serve crash).
