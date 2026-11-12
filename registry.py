"""Pick best run by test_rmse, then promote its registered version via alias.

MLflow >=2.9 deprecates stages (Staging/Production); aliases replace them.
"""
from mlflow.tracking import MlflowClient

from train import EXPERIMENT_NAME, MODEL_NAME

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(experiment.experiment_id)
best_run = min(runs, key=lambda r: r.data.metrics["test_rmse"])
print(f"best run {best_run.info.run_id} test_rmse={best_run.data.metrics['test_rmse']:.4f}")

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
best_version = next(v for v in versions if v.run_id == best_run.info.run_id)

client.set_registered_model_alias(MODEL_NAME, "staging", best_version.version)
client.set_registered_model_alias(MODEL_NAME, "production", best_version.version)
client.set_model_version_tag(MODEL_NAME, best_version.version, "validated", "true")
print(f"{MODEL_NAME} v{best_version.version} -> aliases staging, production")
