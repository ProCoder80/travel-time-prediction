"""
Unit and Integration Tests for Departure Recommendation Engine
"""

import pytest
import pandas as pd
from src.recommendation import (
    generate_candidate_times,
    DepartureRecommendationEngine,
)


def test_candidate_times_generation():
    candidates = generate_candidate_times(center_time_str="08:00", window_minutes=30, interval_minutes=15)
    expected = ["07:30", "07:45", "08:00", "08:15", "08:30"]
    assert candidates == expected


def test_recommendation_evaluation_structure():
    engine = DepartureRecommendationEngine()
    rec = engine.evaluate_candidates(
        destination="GSK",
        distance=51.5,
        planned_time="08:00",
        day_of_week="Wednesday",
        take_407="No",
        window_minutes=30,
        interval_minutes=15,
        compare_toll_option=True,
    )

    assert "planned_time" in rec
    assert "recommended_time" in rec
    assert "recommended_duration_min" in rec
    assert "all_candidates" in rec
    assert "ranked_candidates" in rec
    assert "disclaimer" in rec

    df_cand = rec["all_candidates"]
    assert len(df_cand) == 5
    assert "delta_vs_planned_min" in df_cand.columns
    assert "alt_duration_min" in df_cand.columns
    assert "toll_savings_min" in df_cand.columns


def test_ranked_candidates_order():
    engine = DepartureRecommendationEngine()
    rec = engine.evaluate_candidates(
        destination="Home",
        distance=51.2,
        planned_time="17:00",
        day_of_week="Thursday",
        take_407="No",
        window_minutes=60,
        interval_minutes=20,
    )

    df_ranked = rec["ranked_candidates"]
    # Verify ranked dataframe is monotonically non-decreasing in predicted duration
    durations = df_ranked["predicted_duration_min"].tolist()
    assert durations == sorted(durations)


def test_out_of_distribution_trip_triggers_safeguard():
    engine = DepartureRecommendationEngine()
    # Saturday evening trip to GSK has 0 historical precedent
    rec = engine.evaluate_candidates(
        destination="GSK",
        distance=51.0,
        planned_time="20:00",
        day_of_week="Saturday",
        take_407="No",
        window_minutes=30,
        interval_minutes=15,
    )

    # Must contain warning about lack of historical coverage
    assert len(rec["warnings"]) > 0
    # Meaningfulness flag must be False due to lack of coverage
    assert not rec["is_meaningful"]
    assert any("historical dataset" in note.lower() for note in rec["recommendation_notes"])
