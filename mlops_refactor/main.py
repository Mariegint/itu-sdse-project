from datetime import datetime
import os
from data.preprocessing import preprocess_training_data
from models.train_model import prepare_features, split_features_target, train_LogisticRegression, train_XGBRFClassifier
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd
import datetime
import sys
import json


def main():
    """Main pipeline: load, preprocess, split, train, and save models."""

    #sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
    

    base_dir = os.path.abspath(os.getcwd())  # stay inside mlops_refactor
    input_path = os.path.join(base_dir, "data/raw/raw_data.csv")

    os.makedirs("artifacts",exist_ok=True)

    max_date = "2024-01-31"
    min_date = "2024-01-01"

    data = preprocess_training_data(
        input_path=input_path,
        output_dir="./artifacts",
        min_date="2024-01-01",
        max_date="2024-01-31"
    )
    data.to_csv('./artifacts/train_data_gold.csv', index=False)
    # --- Metadata setup
    current_date = datetime.datetime.now().strftime("%Y_%B_%d")
    experiment_name = current_date
    data_path = "./artifacts/train_data_gold.csv"
    os.makedirs("./artifacts", exist_ok=True)

    current_date = datetime.datetime.now().strftime("%Y_%B_%d")
    data_gold_path = "./artifacts/train_data_gold.csv"
    data_version = "00000"
    experiment_name = current_date

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("mlruns/.trash", exist_ok=True)

    mlflow.set_experiment(experiment_name)

    data= prepare_features(data, data_gold_path)

    X, y= split_features_target(data, target_col="lead_indicator")

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)
    xgb_report = train_XGBRFClassifier(X_train, y_train)
    lr_report = train_LogisticRegression(X_train, y_train, X_test, y_test, experiment_name)

    # Collect results
    model_results = {}

    xgboost_model_path = "./artifacts/lead_model_xgboost.json"
    lr_model_path = "./artifacts/lead_model_lr.pkl"

    model_results[xgboost_model_path] = xgb_report
    model_results[lr_model_path] = lr_report

    # Save combined results to one file
    model_results_path = "./artifacts/model_results.json"
    with open(model_results_path, 'w') as f:
        json.dump(model_results, f, indent=4)

    print(f"✅ Saved combined model results to: {model_results_path}")

if __name__ == "__main__":
    main()