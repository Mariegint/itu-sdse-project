import mlflow
from pathlib import Path


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

    # Sort highest F1 first
    best_run = runs.sort_values("metrics.f1_score", ascending=False).iloc[0]
    best_run_id = best_run.run_id

    best_score = best_run["metrics.f1_score"]
    print(f"Best run ID: {best_run_id}")
    print(f"Best F1-score: {best_score}")

    # In MLflow, the model is stored under: artifacts/model
    run_info = mlflow.get_run(best_run_id)
    model_uri = f"{run_info.info.artifact_uri}/model"

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

    return final_model_path
