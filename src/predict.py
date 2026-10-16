"""
Prediction Module
Provides a clean, validated interface for single-trip travel time predictions
using trained model pipelines.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.feature_engineering import (
    prepare_single_trip_input,
    check_coverage_support,
    time_str_to_components,
)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
_LOADED_MODELS: Dict[str, Any] = {}
_METADATA: Optional[Dict[str, Any]] = None


def get_available_models() -> Dict[str, str]:
    """Returns mapping of model display names to their joblib file names."""
    return {
        "AdaBoost": "adaboost.joblib",
        "Random Forest": "random_forest.joblib",
        "Linear Regression": "linear_regression.joblib",
        "KNN Regressor": "knn_regressor.joblib",
    }


def get_default_model_name() -> str:
    """Returns the name of the best-performing model identified during training."""
    global _METADATA
    if _METADATA is None:
        meta_path = os.path.join(MODELS_DIR, "training_metadata.joblib")
        if os.path.exists(meta_path):
            _METADATA = joblib.load(meta_path)
            return _METADATA.get("best_model_name", "AdaBoost")
    elif _METADATA:
        return _METADATA.get("best_model_name", "AdaBoost")
    return "AdaBoost"


def load_model_pipeline(model_name: Optional[str] = None):
    """Loads and caches the requested model pipeline from disk."""
    available = get_available_models()

    if model_name is not None and model_name not in available:
        raise ValueError(
            f"Unknown model name '{model_name}'. Available: {list(available.keys())}"
        )

    name = model_name if model_name else get_default_model_name()

    if name not in _LOADED_MODELS:
        filename = available.get(name, "adaboost.joblib")
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model file '{filename}' not found at {filepath}. "
                "Please run `python -m src.train_models` first."
            )
        _LOADED_MODELS[name] = joblib.load(filepath)

    return _LOADED_MODELS[name], name


def predict_travel_time(
    destination: str,
    distance: float,
    departure_time: str,
    day_of_week: str,
    take_407: str = "No",
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predicts journey duration in minutes for a specific trip setup.
    Performs input validation, coverage check, and returns structured prediction results.
    """
    pipeline, resolved_model_name = load_model_pipeline(model_name)

    # Validate and transform input into single-row dataframe
    input_df = prepare_single_trip_input(
        destination=destination,
        distance=distance,
        departure_time=departure_time,
        day_of_week=day_of_week,
        take_407=take_407,
    )

    # Run inference
    raw_pred = pipeline.predict(input_df)[0]
    predicted_duration = float(np.clip(raw_pred, a_min=1.0, a_max=None))

    # Coverage audit
    time_info = time_str_to_components(departure_time)
    coverage = check_coverage_support(
        destination=destination,
        departure_hour=time_info["StartHour"],
        distance=float(distance),
        day_of_week=day_of_week,
    )

    return {
        "predicted_duration_minutes": round(predicted_duration, 1),
        "model_used": resolved_model_name,
        "trip_details": {
            "destination": destination,
            "distance_km": round(float(distance), 2),
            "departure_time": departure_time,
            "day_of_week": day_of_week,
            "take_407": take_407,
            "time_of_day": time_info["TimeOfDay"],
        },
        "coverage": coverage,
    }


if __name__ == "__main__":
    # Test sample prediction
    res = predict_travel_time("GSK", 51.5, "07:30", "Tuesday", "No")
    print(f"Predicted Duration: {res['predicted_duration_minutes']} min using {res['model_used']}")
    print(f"Coverage Support: {res['coverage']['coverage_level']}")
