"""
Unit and Integration Tests for Model Pipelines, Inference, and Validation
"""

import pytest
import os
import joblib
import numpy as np

from src.predict import (
    predict_travel_time,
    load_model_pipeline,
    get_available_models,
    MODELS_DIR,
)
from src.feature_engineering import check_coverage_support, prepare_single_trip_input


def test_all_four_models_trained_and_exist():
    available = get_available_models()
    assert len(available) == 4
    expected_models = ["Linear Regression", "KNN Regressor", "Random Forest", "AdaBoost"]
    for m in expected_models:
        assert m in available
        filepath = os.path.join(MODELS_DIR, available[m])
        assert os.path.exists(filepath), f"Trained model artifact {filepath} does not exist."


def test_prediction_output_validity_all_models():
    for model_name in get_available_models():
        res = predict_travel_time(
            destination="GSK",
            distance=51.5,
            departure_time="07:45",
            day_of_week="Tuesday",
            take_407="No",
            model_name=model_name,
        )
        assert isinstance(res, dict)
        pred_val = res["predicted_duration_minutes"]
        # Travel time must be physically positive and reasonable (e.g. 10 to 120 mins)
        assert pred_val > 0.0, f"Predicted duration must be > 0, got {pred_val}"
        assert 15.0 <= pred_val <= 120.0, f"Unreasonable commute prediction: {pred_val}"
        assert res["model_used"] == model_name


def test_unsupported_destination_raises_error():
    with pytest.raises(ValueError, match="Unsupported destination"):
        predict_travel_time(
            destination="New York City",
            distance=50.0,
            departure_time="08:00",
            day_of_week="Monday",
        )


def test_invalid_negative_distance_raises_error():
    with pytest.raises(ValueError, match="strictly positive"):
        predict_travel_time(
            destination="GSK",
            distance=-10.0,
            departure_time="08:00",
            day_of_week="Monday",
        )


def test_extreme_distance_raises_error():
    with pytest.raises(ValueError, match="exceeds sensible commuter threshold"):
        predict_travel_time(
            destination="GSK",
            distance=1500.0,
            departure_time="08:00",
            day_of_week="Monday",
        )


def test_invalid_time_format_raises_error():
    with pytest.raises(ValueError, match="Invalid time format"):
        predict_travel_time(
            destination="GSK",
            distance=51.0,
            departure_time="not-a-time",
            day_of_week="Monday",
        )

    with pytest.raises(ValueError, match="out of 24-hour range"):
        predict_travel_time(
            destination="GSK",
            distance=51.0,
            departure_time="25:10",
            day_of_week="Monday",
        )


def test_invalid_day_of_week_raises_error():
    with pytest.raises(ValueError, match="Invalid day of week"):
        predict_travel_time(
            destination="GSK",
            distance=51.0,
            departure_time="08:00",
            day_of_week="Funday",
        )


def test_missing_model_file_handling():
    # Unknown model name raises ValueError
    with pytest.raises(ValueError, match="Unknown model name"):
        load_model_pipeline("NonExistentModel")

    # Missing physical file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        predict_travel_time(
            destination="GSK",
            distance=51.0,
            departure_time="08:00",
            day_of_week="Monday",
            model_name="Linear Regression",
        )
        # Temporarily mock path or test missing file condition
        fake_available = {"CorruptedModel": "corrupted_file.joblib"}
        from unittest.mock import patch
        with patch("src.predict.get_available_models", return_value=fake_available):
            load_model_pipeline("CorruptedModel")


def test_coverage_safeguard_detection():
    # GSK trip at 18:00 (evening) is out-of-distribution
    cov_evening_gsk = check_coverage_support("GSK", 18, 51.0, "Tuesday")
    assert not cov_evening_gsk["is_well_supported"]
    assert any("GSK" in w for w in cov_evening_gsk["warnings"])

    # Weekend trip is out-of-distribution
    cov_weekend = check_coverage_support("GSK", 8, 51.0, "Saturday")
    assert not cov_weekend["is_well_supported"]
    assert any("weekend" in w.lower() for w in cov_weekend["warnings"])

    # Standard weekday morning trip to GSK should be well-supported
    cov_valid = check_coverage_support("GSK", 8, 51.2, "Tuesday")
    assert cov_valid["is_well_supported"]
    assert len(cov_valid["warnings"]) == 0
