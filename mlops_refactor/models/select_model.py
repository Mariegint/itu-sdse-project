import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient
import json
import joblib
import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
          name=model_name,
          version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)



def select_best_model(mlflow_experiment: str, save_folder: Path) -> Path:
    """
    Selects the best model from an MLflow experiment based on F1-score.
    Downloads the model artifact and saves it to: artifacts/model/model

    Parameters: 
    mlflow_experiment : str
        Name of the MLflow experiment created during training.
    save_folder : Path
        Directory to save the chosen best model.

    Returns:
    Path
        Path to the final saved model
    """

    experiment = mlflow.get_experiment_by_name(mlflow_experiment)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named '{mlflow_experiment}' found.")

    experiment_id = experiment.experiment_id

    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    if runs.empty:
        raise ValueError("No MLflow runs found. Cannot select best model.")

    # Pick only runs that recorded an F1-score
    runs = runs[runs["metrics.f1_score"].notnull()]
    if runs.empty:
        raise ValueError("No F1-score logged in MLflow runs.")

    # Get best run based on F1-score
    best_run = runs.sort_values("metrics.f1_score", ascending=False).iloc[0]
    best_run_id = best_run.run_id

    best_score = best_run["metrics.f1_score"]
    print(f"Best run ID: {best_run_id}")
    print(f"Best F1-score: {best_score}")

    run_info = mlflow.get_run(best_run_id)
    model_uri = f"{run_info.info.artifact_uri}/model"  # In MLflow, the model is stored under: artifacts/model

    # Ensure output folder exists
    save_folder.mkdir(parents=True, exist_ok=True)
    final_model_path = save_folder / "model"

    print(f"Downloading best model to: {final_model_path}")

    # Download artifact folder
    mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(final_model_path)
    )

    print(f"Final selected model saved at: {final_model_path}")

    return best_run, final_model_path

def get_production_model(model_name: str):
    """Return details of the current production model in MLflow."""
    client = MlflowClient()
    prod_models = [
        m for m in client.search_model_versions(f"name='{model_name}'")
        if dict(m)["current_stage"] == "Production"
    ]

    if not prod_models:
        print("⚠️ No model in production.")
        return None

    prod_model_version = dict(prod_models[0])['version']
    prod_model_run_id = dict(prod_models[0])['run_id']
    print(f"Production model name: {model_name}")
    print(f"Version: {prod_model_version}")
    print(f"Run ID: {prod_model_run_id}")

    return {"name": model_name, "version": prod_model_version, "run_id": prod_model_run_id}


def compare_prod_and_best_model(experiment_best: dict, model_name: str):
    """Compare best trained model vs current production model. Returns run_id if new model should be registered."""
    prod_model = get_production_model(model_name)
    best_score = experiment_best["metrics"]["f1_score"]
    run_id_to_register = None

    if prod_model:
        prod_run = mlflow.get_run(prod_model["run_id"])
        prod_score = prod_run.data.metrics.get("f1_score", 0)

        print(f"Trained F1: {best_score:.4f} | Production F1: {prod_score:.4f}")
        if best_score > prod_score:
            print("Registering new model")
            run_id_to_register = experiment_best["run_id"]
        else:
            print(" Production model performs better → no registration.")
    else:
        print(" No model in production")
        run_id_to_register = experiment_best["run_id"]

    return run_id_to_register


def register_best_model(run_id: str, artifact_path: str, model_name: str):
    """Register a model version in MLflow if run_id is provided."""

    print(f'Best model found: {run_id}')

    model_uri = "runs:/{run_id}/{artifact_path}".format(
        run_id=run_id,
        artifact_path=artifact_path
    )
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    wait_until_ready(model_details.name, model_details.version)
    model_details = dict(model_details)
    print(model_details)
    return dict(model_details)

