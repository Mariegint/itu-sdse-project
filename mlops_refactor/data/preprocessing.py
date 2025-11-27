import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

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
    cat_cols = data.select_dtypes(include=["object"]).columns

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

def describe_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates descriptive statistics for all numeric columns in the data
    """
    numeric_data = data.select_dtypes(include=["float64", "int64"])
    
    summary = numeric_data.apply(
        lambda x: pd.Series(
            [
                x.count(), x.isnull().sum(), x.mean(), x.min(), x.max(),
            ],
            index=["Count", "Missing", "Mean", "Min", "Max"],
        )
    ).T

    return summary


def impute_data(data: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """
    Imputes missing values for ALL columns in the data.
    
    - Numeric columns gets missing values filled with mean or median
    - Categorical/object gets its missing values filled with its mode
    
    Args:
        data: input data
        method: "mean" or "median" for continuous variables
    """

    def impute_column(x: pd.Series):
        # continuous imputations
        if x.dtype in ["float64", "int64"]:
            if method == "mean":
                return x.fillna(x.mean())
            return x.fillna(x.median())
        
        # categorical imputations
        return x.fillna(x.mode()[0])

    return data.apply(impute_column)

def scale_continuous_values(data: pd.DataFrame, save_path=None):
    scaler = MinMaxScaler()
    scaler.fit(data)

    if save_path:
        joblib.dump(scaler, save_path)

    data_scaled = pd.DataFrame(scaler.transform(data), columns=data.columns)
    return data_scaled


def create_dummy_cols(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Creates one-hot encoded columns for a categorical variable.
    Drops the original column.
    """
    dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
    data = pd.concat([data, dummies], axis=1)
    return data.drop(columns=[col])

def bin_source_column(data: pd.DataFrame, col: str = "source") -> pd.DataFrame:
    """
    Reduces the cardinality of the source column by grouping categories.
    Unseen categories are mapped to 'Others'.
    """
    mapping = {
        "li": "socials",
        "fb": "socials",
        "organic": "group1",
        "signup": "group1",
    }

    data["bin_source"] = data[col].map(mapping).fillna("Others")
    return data
