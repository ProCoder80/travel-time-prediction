"""
Model Evaluation and Diagnostics Module
Computes regression metrics (MAE, RMSE, R²), produces diagnostic plots
(Actual vs Predicted, Residuals, Model Comparison), and provides beginner-friendly explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "figures"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates regression performance metrics in minutes."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
        "MAPE": round(mape, 2),
    }


def load_test_evaluation_bundle() -> Optional[Dict[str, Any]]:
    """Loads saved test evaluation artifacts from models directory."""
    path = os.path.join(MODELS_DIR, "test_eval_bundle.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def generate_evaluation_plots(save_to_disk: bool = True) -> Dict[str, plt.Figure]:
    """
    Generates high-resolution diagnostic charts:
    1. Model Comparison Bar Chart (MAE & RMSE)
    2. Actual vs Predicted Scatter Plots for all 4 models
    3. Residual Distribution & Residual vs Predicted Plots
    """
    bundle = load_test_evaluation_bundle()
    if bundle is None:
        raise FileNotFoundError("Test evaluation bundle not found. Please train models first.")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    figures = {}

    y_test = bundle["y_test"]
    predictions = bundle["test_predictions"]
    results_df = bundle["results_df"]

    # 1. Model Comparison Chart
    fig1, ax1 = plt.subplots(figsize=(8, 4.5), dpi=150)
    models = results_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, results_df["Test_MAE"], width, label="Test MAE (min)", color="#2563EB", alpha=0.9)
    ax1.bar(x + width/2, results_df["Test_RMSE"], width, label="Test RMSE (min)", color="#7C3AED", alpha=0.9)

    ax1.set_ylabel("Error (minutes)", fontsize=11, fontweight="bold")
    ax1.set_title("Model Performance Comparison (MAE vs RMSE)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.legend(frameon=True, facecolor="#F8FAFC")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_to_disk:
        fig1.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"))
    figures["comparison"] = fig1

    # 2. Actual vs Predicted Plot (2x2 grid)
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    axes_flat = axes2.flatten()

    for idx, (m_name, preds) in enumerate(predictions.items()):
        ax = axes_flat[idx]
        ax.scatter(y_test, preds, color="#0284C7", alpha=0.75, edgecolors="white", s=60, label="Test Trips")
        min_val = min(y_test.min(), preds.min()) - 2
        max_val = max(y_test.max(), preds.max()) + 2
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal (y=x)")
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        ax.set_title(f"{m_name}\nMAE: {mae:.2f} min | R²: {r2:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual TotalTime (min)", fontsize=9)
        ax.set_ylabel("Predicted TotalTime (min)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    if save_to_disk:
        fig2.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"))
    figures["actual_vs_predicted"] = fig2

    # 3. Residual Analysis for Best Model (AdaBoost / Random Forest)
    best_model_name = results_df.sort_values(by="CV_MAE_Mean").iloc[0]["Model"]
    best_preds = predictions[best_model_name]
    residuals = y_test - best_preds

    fig3, (ax_res1, ax_res2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Residuals vs Predicted
    ax_res1.scatter(best_preds, residuals, color="#D97706", alpha=0.75, edgecolors="white", s=60)
    ax_res1.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax_res1.set_title(f"Residuals vs Predicted ({best_model_name})", fontsize=11, fontweight="bold")
    ax_res1.set_xlabel("Predicted Duration (min)", fontsize=9)
    ax_res1.set_ylabel("Residual (Actual - Predicted, min)", fontsize=9)
    ax_res1.grid(True, linestyle=":", alpha=0.6)

    # Residual distribution histogram
    ax_res2.hist(residuals, bins=12, color="#059669", edgecolor="white", alpha=0.85)
    ax_res2.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_res2.set_title(f"Residual Distribution ({best_model_name})\nMean: {np.mean(residuals):.2f} min", fontsize=11, fontweight="bold")
    ax_res2.set_xlabel("Residual (min)", fontsize=9)
    ax_res2.set_ylabel("Frequency", fontsize=9)
    ax_res2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_to_disk:
        fig3.savefig(os.path.join(FIGURES_DIR, "residual_analysis.png"))
    figures["residual_analysis"] = fig3

    return figures


def get_metric_explanations() -> Dict[str, str]:
    """Returns accessible, beginner-friendly explanations of each evaluation metric."""
    return {
        "MAE (Mean Absolute Error)": (
            "Measures the average magnitude of prediction errors in minutes. "
            "For example, an MAE of 3.90 minutes means the model's prediction is, on average, "
            "about 3.9 minutes off from the actual commute duration."
        ),
        "RMSE (Root Mean Squared Error)": (
            "Measures the square root of average squared errors in minutes. "
            "Because errors are squared before averaging, RMSE penalizes large prediction mistakes heavily. "
            "An RMSE of ~5.7 minutes indicates that occasional major traffic delays pull up the squared error."
        ),
        "R² (Coefficient of Determination)": (
            "Represents the proportion of variance in travel time explained by the pre-departure predictors. "
            "An R² of ~0.15 (15%) reflects that scheduled features (hour, day, destination, toll) capture "
            "systematic commuting patterns, while unobserved factors (sudden accidents, weather, construction) "
            "account for remaining variance in historical trips."
        ),
        "Negative R² Explanation": (
            "When R² is negative (as observed on the small chronological test split or certain folds), "
            "it indicates that predicting the naive historical mean of the test set would have smaller squared errors "
            "than the model's predictions. This typically happens when the test set has strong covariate shift "
            "or low variance relative to training data."
        ),
    }


if __name__ == "__main__":
    figs = generate_evaluation_plots(save_to_disk=True)
    print(f"Generated {len(figs)} diagnostic figures in {FIGURES_DIR}")
