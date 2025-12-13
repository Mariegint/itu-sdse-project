import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import datetime
import json
from pathlib import Path

UNUSED_FEATURES = [
    "is_active",
    "marketing_consent",
    "first_booking",
    "existing_customer",
    "last_seen",
]

#TODO : check if these columns are added back after EDA
TEMP_REMOVED_BEFORE_EDA = [
    "domain",
    "country",
    "visited_learn_more_before_booking",
    "visited_faq",
]

def describe_numeric_col(x):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x

def filter_data_by_date(data: pd.DataFrame, min_date, max_date=None, date_col="date_part", output_path="./artifacts/date_limits.json"):
    """
    Filters a DataFrame to include only rows where the date column is between min_date and max_date.
    Saves the resulting actual date limits to a JSON file.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not max_date:
        max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date).date()
    min_date = pd.to_datetime(min_date).date()

    data[date_col] = pd.to_datetime(data[date_col]).dt.date
    filtered = data[(data[date_col] >= min_date) & (data[date_col] <= max_date)]

    min_date_actual = filtered[date_col].min()
    max_date_actual = filtered[date_col].max()
    date_limits = {"min_date": str(min_date_actual), "max_date": str(max_date_actual)}

    with open(output_path, "w") as f:
        json.dump(date_limits, f, indent=4)

    return filtered


def drop_unused_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drops irrelevant or temporary columns that should 
    not be used for modelling
    """

    cols_to_drop = [col for col in UNUSED_FEATURES + TEMP_REMOVED_BEFORE_EDA
                    if col in data.columns]

    return data.drop(columns=cols_to_drop)

def remove_empty_identifiers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces empty strings with NaN in key identifier columns
    """
    for col in ["lead_indicator", "lead_id", "customer_code"]:
        if col in data:
            data[col].replace("", np.nan, inplace=True)
    return data

def drop_invalid_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows with missing target or invalid source
    """
    data = data.dropna(subset=["lead_indicator"])
    data = data.dropna(subset=["lead_id"])
    data = data[data["source"] == "signup"]
    return data

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Runs full cleaning pipeline
    """
    data = remove_empty_identifiers(data)
    data = drop_invalid_rows(data)
    return data

def cast_categorical(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Casts specified columns to 'object' datatype.
    """
    for col in columns:
        if col in data.columns:
            data[col] = data[col].astype("object")

    return data

def split_cat_cont(data: pd.DataFrame):
    """
    Splits data into categorical and continuous subsets
    based on dtype.
    """
    cont_cols = data.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = data.select_dtypes(include=["object", "bool"]).columns

    data_cat = data[cat_cols].copy()
    data_cont = data[cont_cols].copy()

    return data_cat, data_cont

def clip_outliers(data):
    """
    Clips outliers based on how many standard deviations
    a data point is from the mean
    """
    return data.apply(
        lambda x: x.clip(x.mean() - 2 * x.std(), x.mean() + 2 * x.std())
    )

def impute_all_missing(df: pd.DataFrame, method="mean"):
    """
    Impute missing values in a DataFrame for both numeric and categorical columns.
    """
    cat_vars, cont_vars = split_cat_cont(df)
    if "customer_code" in cat_vars.columns:
        cat_vars["customer_code"] = cat_vars["customer_code"].fillna("None")

    cat_vars = cat_vars.apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
    cont_vars = cont_vars.apply(lambda x: impute_missing_values(x, method))

    return cat_vars, cont_vars

def scale_continuous_values(cont_vars: pd.DataFrame, save_path=None):
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    if save_path:
        joblib.dump(scaler, save_path)

    data_scaled = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    return data_scaled

def bin_source_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces the cardinality of the source column by grouping categories.
    Unseen categories are mapped to 'Others'.
    """

    values_list = ['li', 'organic','signup','fb']
    data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
    mapping = {'li' : 'socials', 
            'fb' : 'socials', 
            'organic': 'group1', 
            'signup': 'group1'
            }

    data['bin_source'] = data['source'].map(mapping)
    return data

def preprocess_training_data(
    data: pd.DataFrame,
    min_date: str,
    max_date: str,
    output_dir: str = "./artifacts",
):
    """
    Runs the full preprocessing pipeline for training data.

    Parameters
    ----------
    input_path : str
        Path to the raw training data CSV.
    output_dir : str
        Directory to save any preprocessing artifacts (e.g., scalers).
    min_date : str
        Minimum date to include in the dataset.
    max_date : str
        Maximum date to include in the dataset.

    Returns
    -------
    data : pd.DataFrame
        Fully preprocessed training dataset ready for modeling.
    """
    data = filter_data_by_date(data, min_date=min_date, max_date=max_date)
    data = drop_unused_features(data)
    data = clean_data(data)

    cat_vars, cont_vars = split_cat_cont(data)
    cont_vars = clip_outliers(cont_vars)

    cont_vars = cont_vars.apply(impute_missing_values)
    cat_vars.loc[cat_vars['customer_code'].isna(), 'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)

    cont_vars = scale_continuous_values(cont_vars, save_path=os.path.join(output_dir, "scaler.pkl"))
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    data = bin_source_column(data)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "train_data_gold.csv"), index=False)

    # Encode all object columns for XGBoost
    #object_cols = data.select_dtypes(include="object").columns
    #if len(object_cols) > 0:
    #    print("Encoding object columns:", list(object_cols))
    #    for col in object_cols:
    #        data[col] = data[col].astype("category").cat.codes

    return data
