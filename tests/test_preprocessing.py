"""
Unit and Integration Tests for Data Preprocessing & Leakage Prevention
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.data_preprocessing import (
    load_dataset,
    clean_and_prepare_dataset,
    parse_date_value,
    parse_time_value,
    split_dataset,
    FORBIDDEN_LEAKAGE_COLUMNS,
    REQUIRED_COLUMNS,
)


def test_load_dataset_success():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_empty_dataset_raises_error():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("")
        temp_path = f.name
    try:
        with pytest.raises(Exception):
            load_dataset(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_missing_required_columns_raises_error():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Date,StartTime,GoingTo\n01-06-12,16:37,Home\n")
        temp_path = f.name
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_parse_date_formats():
    # Test MM-DD-YY format
    dt1 = parse_date_value("01-06-12", "Friday")
    assert dt1.year == 2012
    assert dt1.month == 1
    assert dt1.day == 6
    assert dt1.strftime("%A") == "Friday"

    # Test MM/DD/YYYY format
    dt2 = parse_date_value("12/23/2011", "Friday")
    assert dt2.year == 2011
    assert dt2.month == 12
    assert dt2.day == 23


def test_parse_time_values():
    h1, m1, dec1 = parse_time_value("16:37")
    assert h1 == 16
    assert m1 == 37
    assert abs(dec1 - (16 + 37 / 60.0)) < 1e-4

    h2, m2, dec2 = parse_time_value("8:20")
    assert h2 == 8
    assert m2 == 20
    assert abs(dec2 - (8 + 20 / 60.0)) < 1e-4


def test_clean_and_prepare_dataset_filtering():
    raw_df = load_dataset()
    cleaned = clean_and_prepare_dataset(raw_df)
    assert len(cleaned) > 0
    # Confirm no non-positive distances or travel times
    assert (cleaned["Distance"] > 0).all()
    assert (cleaned["TotalTime"] > 0).all()
    # Confirm derived temporal features exist
    assert "StartHour" in cleaned.columns
    assert "DepartureDecimal" in cleaned.columns
    assert "TimeOfDay" in cleaned.columns


def test_target_leakage_prevention():
    raw_df = load_dataset()
    split_info = split_dataset(raw_df, strategy="stratified")
    X_train = split_info["X_train"]
    feature_cols = split_info["feature_cols"]

    # Verify that forbidden post-trip telemetry columns are completely absent
    for forbidden in FORBIDDEN_LEAKAGE_COLUMNS:
        assert forbidden not in feature_cols, f"Forbidden column '{forbidden}' found in feature list!"
        assert forbidden not in X_train.columns, f"Forbidden column '{forbidden}' found in X_train!"

    # Verify target TotalTime is strictly in y, not in X
    assert "TotalTime" not in feature_cols
    assert "TotalTime" not in X_train.columns


def test_stratified_split_preserves_toll_representation():
    raw_df = load_dataset()
    split_info = split_dataset(raw_df, test_size=0.2, strategy="stratified", random_state=42)
    X_train = split_info["X_train"]
    X_test = split_info["X_test"]

    # Both train and test sets must have 'Yes' and 'No' for Take407All
    assert "Yes" in X_train["Take407All"].values
    assert "No" in X_train["Take407All"].values
    assert "Yes" in X_test["Take407All"].values
    assert "No" in X_test["Take407All"].values


def test_chronological_split_orders_dates():
    raw_df = load_dataset()
    split_info = split_dataset(raw_df, test_size=0.2, strategy="chronological")
    assert split_info["metadata"]["strategy"] == "chronological"
    assert "train_date_span" in split_info["metadata"]
    assert "test_date_span" in split_info["metadata"]
    # Verify test period begins after or at train period end
    train_end = split_info["metadata"]["train_date_span"][1]
    test_start = split_info["metadata"]["test_date_span"][0]
    assert test_start >= train_end
