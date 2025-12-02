import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report

from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint

import os
import json
import joblib
import mlflow
import mlflow.pyfunc


def create_dummy_cols(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Creates one-hot encoded columns for a categorical variable.
    Drops the original column.
    """
    dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
    data = pd.concat([data, dummies], axis=1)
    return data.drop(columns=[col])


def prepare_features(data: pd.DataFrame, data_path: str) -> pd.DataFrame:
    """
    Cleans and transforms the dataset for model training:
    - Drops unused columns
    - One-hot encodes categorical variables
    - Converts all columns to float64
    """
    data = pd.read_csv(data_path)
    # Drop unnecessary columns
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1, errors="ignore")

    # Define categorical columns
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]

    # Separate categorical and other variables
    cat_vars = data[cat_cols].copy()
    other_vars = data.drop(cat_cols, axis=1)

    # One-hot encode categorical variables
    for col in cat_vars.columns:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    # Combine and ensure float type
    final_data = pd.concat([other_vars, cat_vars], axis=1)
    final_data = final_data.astype("float64")
    return final_data


def split_features_target(data: pd.DataFrame, target_col: str = "lead_indicator"):
    """
    Splits the DataFrame into features (X) and target (y).
    """
    y = data[target_col]
    X = data.drop(columns=[target_col])
    return X, y



def train_XGBRFClassifier(X_train, y_train, save_path="./artifacts/lead_model_xgboost.json"):
    """
    Train an XGBRFClassifier with randomized hyperparameter search and save the best model.
    """
    model = XGBRFClassifier(random_state=42)
    
    # Define parameter search space
    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }
    model_grid = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=10,
        cv=10,
        n_jobs=-1,
        verbose=3
    )
    model_grid.fit(X_train, y_train)
    xgboost_model = model_grid.best_estimator_
    xgboost_model.save_model(save_path)
    y_pred_train = xgboost_model.predict(X_train)
    model_results = {
        save_path: classification_report(y_train, y_pred_train, output_dict=True)
    }
    
    return xgboost_model, model_results

class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

def train_LogisticRegression(X_train, y_train, X_test, y_test, experiment_name):
    os.makedirs("./artifacts", exist_ok=True)

    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id):
        model = LogisticRegression()
        params = {
            'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            'penalty': ["none", "l1", "l2", "elasticnet"],
            'C': [100, 10, 1.0, 0.1, 0.01]
        }

        grid = RandomizedSearchCV(model, params, n_iter=10, cv=3, verbose=1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred_test = best_model.predict(X_test)
        report = classification_report(y_test, y_pred_test, output_dict=True)

        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
        mlflow.log_artifacts("artifacts", artifact_path="model")

        model_path = "./artifacts/lead_model_lr.pkl"
        joblib.dump(best_model, model_path)
        mlflow.pyfunc.log_model('model', python_model=lr_wrapper(best_model))

        with open('./artifacts/columns_list.json', 'w') as f:
            json.dump({'column_names': list(X_train.columns)}, f)

        with open('./artifacts/model_results.json', 'w') as f:
            json.dump({model_path: report}, f, indent=4)

    return best_model, report


def train_and_select_best(X_train, y_train, X_test, y_test):
    """
    Trains multiple candidate models and selects the best based on F1 score.
    Returns the best model and its metrics.
    """
    candidates = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "gb": GradientBoostingClassifier()
    }

    best_model = None
    best_score = -1
    best_name = None
    results = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"f1": f1, "accuracy": acc}

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name

    return best_model, best_name, results
