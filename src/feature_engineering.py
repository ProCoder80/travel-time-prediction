"""
Feature Engineering Module
Defines scikit-learn preprocessing transformers, handles feature transformations,
and prepares runtime single-trip inputs with strict validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = ["Distance", "DepartureDecimal", "StartHour"]
CATEGORICAL_FEATURES = ["GoingTo", "DayOfWeek", "Take407All", "TimeOfDay"]
ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

VALID_DESTINATIONS = ["GSK", "Home"]
VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
VALID_TOLL_CHOICES = ["Yes", "No"]


def get_feature_names() -> List[str]:
    """Returns the ordered list of feature names used by models."""
    return list(ALL_MODEL_FEATURES)


def build_preprocessor() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer.
    - StandardScaler applied to numeric features.
    - OneHotEncoder applied to categorical features with handle_unknown='ignore'.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def time_str_to_components(time_str: str) -> Dict[str, Any]:
    """
    Converts HH:MM string (e.g., '08:15' or '8:15') into numeric components.
    """
    s = str(time_str).strip()
    if ":" not in s:
        raise ValueError(f"Invalid time format '{time_str}'. Expected 'HH:MM'.")

    parts = s.split(":")
    try:
        h = int(parts[0])
        m = int(parts[1])
    except ValueError:
        raise ValueError(f"Non-numeric values in time string '{time_str}'.")

    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Time '{time_str}' is out of 24-hour range (00:00 to 23:59).")

    decimal_hour = round(h + m / 60.0, 4)

    if 5 <= h < 12:
        tod = "Morning"
    elif 12 <= h < 17:
        tod = "Afternoon"
    elif 17 <= h < 21:
        tod = "Evening"
    else:
        tod = "Night"

    return {
        "StartHour": h,
        "StartMinute": m,
        "DepartureDecimal": decimal_hour,
        "TimeOfDay": tod,
    }


def prepare_single_trip_input(
    destination: str,
    distance: float,
    departure_time: str,
    day_of_week: str,
    take_407: str = "No",
) -> pd.DataFrame:
    """
    Validates and transforms a single trip input from UI or API into
    a 1-row DataFrame formatted exactly as expected by the model pipeline.
    """
    # Destination validation
    dest_clean = str(destination).strip()
    if dest_clean not in VALID_DESTINATIONS:
        raise ValueError(f"Unsupported destination '{destination}'. Supported: {VALID_DESTINATIONS}")

    # Distance validation
    try:
        dist_val = float(distance)
    except (ValueError, TypeError):
        raise ValueError(f"Distance must be a numeric value, got '{distance}'.")

    if dist_val <= 0:
        raise ValueError(f"Distance must be strictly positive, got {dist_val} km.")
    if dist_val > 500:
        raise ValueError(f"Distance {dist_val} km exceeds sensible commuter threshold (500 km).")

    # Day of week validation
    dow_clean = str(day_of_week).strip().capitalize()
    if dow_clean not in VALID_DAYS:
        raise ValueError(f"Invalid day of week '{day_of_week}'. Supported: {VALID_DAYS}")

    # Toll choice validation
    toll_clean = str(take_407).strip().capitalize()
    if toll_clean not in VALID_TOLL_CHOICES:
        toll_clean = "Yes" if toll_clean.lower().startswith("y") else "No"

    # Time components
    time_info = time_str_to_components(departure_time)

    row_data = {
        "Distance": dist_val,
        "DepartureDecimal": time_info["DepartureDecimal"],
        "StartHour": time_info["StartHour"],
        "GoingTo": dest_clean,
        "DayOfWeek": dow_clean,
        "Take407All": toll_clean,
        "TimeOfDay": time_info["TimeOfDay"],
    }

    return pd.DataFrame([row_data])


def check_coverage_support(
    destination: str,
    departure_hour: int,
    distance: float,
    day_of_week: str,
) -> Dict[str, Any]:
    """
    Methodological safeguard: checks whether the given trip profile is
    well-supported by the historical training dataset.
    
    Historical distribution in travel_times.csv:
    - GSK trips occur in morning hours (6:00 to 9:00).
    - Home trips occur in afternoon/evening hours (12:00 to 20:00).
    - Distances range between 48 km and 60 km.
    - Commutes only occur on weekdays (Monday - Friday).
    """
    warnings = []
    is_supported = True

    # Direction and time coverage check
    if destination == "GSK" and not (5 <= departure_hour <= 10):
        warnings.append(
            f"Caution: All historical trips to GSK departed between 6:00 and 9:00 AM. "
            f"Departure at {departure_hour:02d}:00 is outside historical morning commute patterns."
        )
        is_supported = False
    elif destination == "Home" and not (11 <= departure_hour <= 21):
        warnings.append(
            f"Caution: All historical trips to Home departed between 12:00 and 20:00. "
            f"Departure at {departure_hour:02d}:00 is outside historical afternoon/evening commute patterns."
        )
        is_supported = False

    # Weekend check
    if day_of_week in ["Saturday", "Sunday"]:
        warnings.append(
            f"Historical dataset contains 0 weekend trips. Weekend predictions assume weekday traffic behavior."
        )
        is_supported = False

    # Distance check
    if distance < 45.0 or distance > 65.0:
        warnings.append(
            f"Distance ({distance:.1f} km) differs from the historical commute corridor (48 to 60 km). "
            f"Predictions may extrapolate linearly."
        )
        is_supported = False

    return {
        "is_well_supported": is_supported,
        "warnings": warnings,
        "coverage_level": "High" if is_supported else ("Medium" if len(warnings) == 1 else "Low"),
    }
