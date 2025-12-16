import pandas as pd
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def filter_by_data(data: pd.DataFrame, min_date, max_date) -> pd.DataFrame:
    """
    Time limit data between min_date and max_date
    """
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]