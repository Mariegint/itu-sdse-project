import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from dvc.api import DVCFileSystem

from mlops_refactor.data.preprocessing import preprocess_training_data
from mlops_refactor.models.train_model import (
    train_XGBRFClassifier,
    train_LogisticRegression,
)

# next step
from mlops_refactor.models.select_model import select_best_model  

# Optional: use DVC API if running in CI
try:
    from dvc.api import DVCFileSystem
except ImportError:
    raise ImportError(
        "DVC is required. Please install it with `pip install 'dvc[fs]'` "
        "or the appropriate remote plugin (e.g., 'dvc[s3]', 'dvc[gdrive]')."
    )

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_data_from_dvc(path: str) -> pd.DataFrame:
    """Load dataset directly from DVC remote."""
    print(f"Loading dataset from DVC: {path}")
    fs = DVCFileSystem(".")  # uses the current repo’s DVC config
    with fs.open(path, "r") as f:
        return pd.read_csv(f)


def run_pipeline():
    """
    Full training pipeline:
      1. Preprocess data
      2. Train/test split
      3. Train XGBoost + Logistic Regression
      4. Select best model using MLflow
      5. Save final model artifact
    """
    print("Preprocessing data...")
    dvc_data_path = "mlops_refactor/data/raw/raw_data.csv"
    data = load_data_from_dvc(dvc_data_path)


    data = preprocess_training_data(
        data=data,
        min_date="2024-01-01",
        max_date="2024-01-31",
        output_dir="artifacts"
    )

    print("Splitting train/test...")
    y = data["lead_indicator"]
    X = data.drop(columns=["lead_indicator"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training XGBoost model...")
    xgb_path = ARTIFACT_DIR / "xgb_model.json"
    train_XGBRFClassifier(
        X_train, y_train,
        save_path=str(xgb_path)
    )

    print("Training Logistic Regression...")
    train_LogisticRegression(
        X_train, y_train,
        X_test, y_test,
        experiment_name="LeadPrediction"
    )

    print("Selecting best model...")
    final_model_path = select_best_model(
        mlflow_experiment="LeadPrediction",
        save_folder=ARTIFACT_DIR
    )

    print(f"Final model saved at: {final_model_path}")
    print(str(final_model_path))
    return final_model_path


if __name__ == "__main__":
    run_pipeline()
