"""
Data Preprocessing Module
Handles safe loading, date/time parsing, validation, and train/test splitting.
Guarantees zero target leakage and produces clean, reproducible data splits.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# List of columns strictly forbidden in before-departure modeling to prevent target & post-trip leakage
FORBIDDEN_LEAKAGE_COLUMNS = [
    "AvgSpeed",
    "AvgMovingSpeed",
    "MaxSpeed",
    "MovingTime",
    "FuelEconomy",
    "Comments",
    "x",
]

REQUIRED_COLUMNS = [
    "Date",
    "StartTime",
    "DayOfWeek",
    "GoingTo",
    "Distance",
    "TotalTime",
    "Take407All",
]


def find_dataset_path(custom_path: Optional[str] = None) -> str:
    """Locates travel_times.csv across common project locations."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    candidates = [
        os.path.join("data", "travel_times.csv"),
        "travel_times.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "travel_times.csv"),
        os.path.join(os.path.dirname(__file__), "..", "travel_times.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("Could not locate 'travel_times.csv' in workspace root or 'data/' folder.")


def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Safely loads the travel times CSV file.
    Validates presence of required columns and reports initial data shape.
    """
    path = find_dataset_path(filepath)
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset file is empty.")

    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {missing_required}")

    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df


def parse_date_value(date_str: Any, expected_dow: Optional[str] = None) -> pd.Timestamp:
    """
    Parses a date string supporting both '%m-%d-%y' and '%m/%d/%Y' formats.
    Validates against the expected DayOfWeek when provided.
    """
    if pd.isna(date_str):
        return pd.NaT

    s = str(date_str).strip()
    formats_to_try = ["%m-%d-%y", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%Y-%m-%d"]

    for fmt in formats_to_try:
        try:
            dt = pd.to_datetime(s, format=fmt)
            if expected_dow:
                if dt.strftime("%A").lower() == expected_dow.strip().lower():
                    return dt
            else:
                return dt
        except (ValueError, TypeError):
            continue

    # Fallback to flexible parsing
    try:
        return pd.to_datetime(s)
    except Exception as e:
        logger.warning(f"Failed to parse date string '{date_str}': {e}")
        return pd.NaT


def parse_time_value(time_str: Any) -> Tuple[int, int, float]:
    """
    Parses StartTime string (e.g., '16:37' or '8:20') into:
    (hour, minute, decimal_hour).
    """
    if pd.isna(time_str):
        return 0, 0, 0.0

    s = str(time_str).strip()
    parts = s.split(":")
    if len(parts) >= 2:
        try:
            h = int(parts[0])
            m = int(parts[1])
            decimal_hour = h + m / 60.0
            return h, m, decimal_hour
        except ValueError:
            pass

    return 0, 0, 0.0


def categorize_time_of_day(hour: int) -> str:
    """Categorizes departure hour into human-understandable periods."""
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def clean_and_prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Parses dates and departure times
    - Validates plausible distance and travel time values (> 0)
    - Standardizes categorical values
    - Preserves genuine extreme travel events (documented as weather/accidents)
    """
    cleaned = df.copy()

    # Parse dates with day-of-week validation
    cleaned["ParsedDate"] = [
        parse_date_value(row["Date"], row.get("DayOfWeek"))
        for _, row in cleaned.iterrows()
    ]

    # Parse start times
    time_tuples = [parse_time_value(t) for t in cleaned["StartTime"]]
    cleaned["StartHour"] = [t[0] for t in time_tuples]
    cleaned["StartMinute"] = [t[1] for t in time_tuples]
    cleaned["DepartureDecimal"] = [t[2] for t in time_tuples]

    # Standardize string categories
    cleaned["GoingTo"] = cleaned["GoingTo"].astype(str).str.strip()
    cleaned["DayOfWeek"] = cleaned["DayOfWeek"].astype(str).str.strip()
    cleaned["Take407All"] = cleaned["Take407All"].astype(str).str.strip().str.capitalize()
    cleaned["Take407All"] = cleaned["Take407All"].replace({"Y": "Yes", "N": "No"})

    # Temporal feature derivations
    cleaned["Month"] = cleaned["ParsedDate"].dt.month
    cleaned["DayOfMonth"] = cleaned["ParsedDate"].dt.day
    cleaned["IsWeekend"] = cleaned["ParsedDate"].dt.dayofweek.isin([5, 6]).astype(int)
    cleaned["TimeOfDay"] = cleaned["StartHour"].apply(categorize_time_of_day)

    # Physical validity filtering
    initial_len = len(cleaned)
    valid_mask = (cleaned["Distance"] > 0) & (cleaned["TotalTime"] > 0)
    cleaned = cleaned[valid_mask].reset_index(drop=True)

    if len(cleaned) < initial_len:
        logger.info(f"Filtered out {initial_len - len(cleaned)} rows with non-positive distance or duration.")

    return cleaned


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    strategy: str = "stratified",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Splits dataset into train and test sets using either:
    1. 'stratified' (recommended for balanced evaluation of toll/destination)
    2. 'chronological' (evaluates performance on later trips, though suffers from covariate shift)
    """
    from sklearn.model_selection import train_test_split

    df_prepared = clean_and_prepare_dataset(df)

    # Predictor feature set - strict no-leakage pre-departure features
    feature_cols = [
        "Distance",
        "DepartureDecimal",
        "StartHour",
        "GoingTo",
        "DayOfWeek",
        "Take407All",
        "TimeOfDay",
    ]
    target_col = "TotalTime"

    X = df_prepared[feature_cols].copy()
    y = df_prepared[target_col].copy()

    if strategy == "chronological":
        # Sort by parsed date and time
        sorted_indices = df_prepared.sort_values(by=["ParsedDate", "StartTime"]).index
        split_point = int(len(sorted_indices) * (1.0 - test_size))
        train_idx = sorted_indices[:split_point]
        test_idx = sorted_indices[split_point:]

        X_train, X_test = X.loc[train_idx].reset_index(drop=True), X.loc[test_idx].reset_index(drop=True)
        y_train, y_test = y.loc[train_idx].reset_index(drop=True), y.loc[test_idx].reset_index(drop=True)
        date_train_span = (df_prepared.loc[train_idx, "ParsedDate"].min(), df_prepared.loc[train_idx, "ParsedDate"].max())
        date_test_span = (df_prepared.loc[test_idx, "ParsedDate"].min(), df_prepared.loc[test_idx, "ParsedDate"].max())

        metadata = {
            "strategy": "chronological",
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_date_span": date_train_span,
            "test_date_span": date_test_span,
            "note": "Chronological split exhibits covariate shift: 0 toll trips exist in the test period.",
        }
    else:
        # Stratified by Take407All to preserve toll-road representation in test set
        stratify_key = df_prepared["Take407All"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_key
        )
        metadata = {
            "strategy": "stratified",
            "train_size": len(X_train),
            "test_size": len(X_test),
            "random_state": random_state,
            "note": "Stratified on Take407All to maintain toll vs non-toll representation across train & test splits.",
        }

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "feature_cols": feature_cols,
        "target_col": target_col,
        "metadata": metadata,
        "full_cleaned_df": df_prepared,
    }


if __name__ == "__main__":
    raw_df = load_dataset()
    split_info = split_dataset(raw_df, strategy="stratified")
    print(f"Dataset successfully prepared. Train: {len(split_info['X_train'])}, Test: {len(split_info['X_test'])}")
    print("Features:", split_info["feature_cols"])
