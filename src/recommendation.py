"""
Departure-Time Recommendation Engine
Compares candidate departure times within a selected window,
predicts travel durations, ranks alternatives, evaluates historical coverage,
and highlights optimal departure options with clear methodological safeguards.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.predict import predict_travel_time, load_model_pipeline, get_default_model_name
from src.feature_engineering import check_coverage_support, prepare_single_trip_input, time_str_to_components


def generate_candidate_times(
    center_time_str: str,
    window_minutes: int = 60,
    interval_minutes: int = 15,
) -> List[str]:
    """
    Generates a list of HH:MM departure times centered around center_time_str
    within [-window_minutes, +window_minutes] at step interval_minutes.
    """
    parts = center_time_str.strip().split(":")
    base_h, base_m = int(parts[0]), int(parts[1])
    base_dt = datetime(2026, 1, 1, base_h, base_m)

    start_dt = base_dt - timedelta(minutes=window_minutes)
    end_dt = base_dt + timedelta(minutes=window_minutes)

    candidates = []
    curr = start_dt
    while curr <= end_dt:
        # Wrap into standard HH:MM within 24 hours
        candidates.append(curr.strftime("%H:%M"))
        curr += timedelta(minutes=interval_minutes)

    return candidates


class DepartureRecommendationEngine:
    """
    Evaluates and compares candidate departure times for a given commuter trip.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or get_default_model_name()
        self.pipeline, self.resolved_name = load_model_pipeline(self.model_name)

    def evaluate_candidates(
        self,
        destination: str,
        distance: float,
        planned_time: str,
        day_of_week: str,
        take_407: str = "No",
        window_minutes: int = 60,
        interval_minutes: int = 15,
        compare_toll_option: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluates candidate departure times around planned_time.
        Computes predicted duration for each, calculates time differences,
        ranks candidates, and generates recommendation warnings.
        """
        candidates = generate_candidate_times(
            center_time_str=planned_time,
            window_minutes=window_minutes,
            interval_minutes=interval_minutes,
        )

        results = []
        coverage_warnings_set = set()
        lowest_time = float("inf")
        best_candidate = None

        # Predict for each candidate time
        for c_time in candidates:
            # Predict with selected toll setting
            pred_info = predict_travel_time(
                destination=destination,
                distance=distance,
                departure_time=c_time,
                day_of_week=day_of_week,
                take_407=take_407,
                model_name=self.resolved_name,
            )
            duration = pred_info["predicted_duration_minutes"]
            is_supported = pred_info["coverage"]["is_well_supported"]
            warnings = pred_info["coverage"]["warnings"]
            for w in warnings:
                coverage_warnings_set.add(w)

            row = {
                "departure_time": c_time,
                "predicted_duration_min": duration,
                "take_407": take_407,
                "is_planned_time": (c_time == planned_time),
                "is_well_supported": is_supported,
            }

            # If toll comparison is requested, also compute alternative toll duration
            if compare_toll_option:
                alt_toll = "Yes" if take_407 == "No" else "No"
                alt_pred = predict_travel_time(
                    destination=destination,
                    distance=distance,
                    departure_time=c_time,
                    day_of_week=day_of_week,
                    take_407=alt_toll,
                    model_name=self.resolved_name,
                )
                row["alt_toll_choice"] = alt_toll
                row["alt_duration_min"] = alt_pred["predicted_duration_minutes"]
                row["toll_savings_min"] = round(duration - alt_pred["predicted_duration_minutes"], 1)

            results.append(row)

        df_candidates = pd.DataFrame(results)

        # Baseline planned time duration
        planned_matches = df_candidates[df_candidates["is_planned_time"]]
        if not planned_matches.empty:
            baseline_duration = planned_matches.iloc[0]["predicted_duration_min"]
        else:
            baseline_duration = df_candidates.iloc[len(df_candidates)//2]["predicted_duration_min"]

        # Calculate difference relative to planned time (negative means time saved)
        df_candidates["delta_vs_planned_min"] = (
            df_candidates["predicted_duration_min"] - baseline_duration
        ).round(1)

        # Sort candidates to find best departure time
        df_sorted = df_candidates.sort_values(by="predicted_duration_min").reset_index(drop=True)
        recommended_row = df_sorted.iloc[0]

        time_saved_min = round(baseline_duration - recommended_row["predicted_duration_min"], 1)

        # Determine recommendation confidence & meaningfulness
        is_meaningful = True
        recommendation_notes = []

        if time_saved_min <= 0.5:
            recommendation_notes.append(
                f"Minimal time variance across window ({time_saved_min} min difference). "
                "Departing at your planned time is practically as efficient."
            )
            is_meaningful = False
        else:
            recommendation_notes.append(
                f"Historical patterns estimate departing at {recommended_row['departure_time']} "
                f"could save approximately {time_saved_min} minutes compared to {planned_time}."
            )

        if len(coverage_warnings_set) > 0:
            is_meaningful = is_meaningful and False
            recommendation_notes.append(
                "Notice: Selected inputs have limited coverage in the historical dataset. "
                "Treat this ranking as exploratory analysis rather than a guaranteed recommendation."
            )

        return {
            "planned_time": planned_time,
            "baseline_duration_min": baseline_duration,
            "recommended_time": recommended_row["departure_time"],
            "recommended_duration_min": recommended_row["predicted_duration_min"],
            "estimated_savings_min": max(0.0, time_saved_min),
            "is_meaningful": is_meaningful,
            "all_candidates": df_candidates,
            "ranked_candidates": df_sorted,
            "warnings": list(coverage_warnings_set),
            "recommendation_notes": recommendation_notes,
            "model_used": self.resolved_name,
            "disclaimer": (
                "IMPORTANT: Recommendations reflect patterns learned from historical commuter trips (2011–2012). "
                "They are statistical estimates, not live traffic feeds, and cannot guarantee real-time arrival times."
            ),
        }
