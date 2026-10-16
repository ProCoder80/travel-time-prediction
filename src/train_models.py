"""
Model Training Module
Trains and compares the four core regression models:
1. Linear Regression (Baseline)
2. K-Nearest Neighbors Regressor (KNN)
3. Random Forest Regressor
4. AdaBoost Regressor

Evaluates models using cross-validation and held-out test splits,
saves trained scikit-learn pipelines, and writes comparative reports.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from src.data_preprocessing import load_dataset, split_dataset
from src.feature_engineering import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))


def get_model_definitions() -> Dict[str, Any]:
    """Returns the four core regression models with tuned, stable configurations."""
    return {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=50,
            learning_rate=0.05,
            loss="linear",
            random_state=42,
        ),
    }


def train_and_evaluate_all(
    strategy: str = "stratified",
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, Dict[str, Any]]:
    """
    Executes end-to-end training and evaluation:
    1. Loads dataset and applies preprocessing/split.
    2. Constructs scikit-learn Pipeline for each model.
    3. Runs 5-fold cross-validation on the training set.
    4. Evaluates on the held-out test set (MAE, RMSE, R²).
    5. Persists pipelines to disk.
    6. Identifies the best-performing model based on validation MAE.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    raw_df = load_dataset()
    split_info = split_dataset(
        raw_df, test_size=test_size, strategy=strategy, random_state=random_state
    )

    X_train = split_info["X_train"]
    y_train = split_info["y_train"]
    X_test = split_info["X_test"]
    y_test = split_info["y_test"]

    model_defs = get_model_definitions()
    trained_pipelines: Dict[str, Pipeline] = {}
    comparison_rows = []
    test_predictions: Dict[str, np.ndarray] = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    logger.info(f"Training 4 models using '{strategy}' split (Train: {len(X_train)}, Test: {len(X_test)})...")

    for name, estimator in model_defs.items():
        # Build composite pipeline: Preprocessing + Model
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", estimator),
        ])

        # Cross-validation on training data only
        cv_results = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=kf,
            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
            return_train_score=False,
        )

        cv_mae = -cv_results["test_neg_mean_absolute_error"].mean()
        cv_mae_std = cv_results["test_neg_mean_absolute_error"].std()
        cv_rmse = -cv_results["test_neg_root_mean_squared_error"].mean()
        cv_r2 = cv_results["test_r2"].mean()

        # Fit on entire training set
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Evaluate on held-out test set
        preds = pipe.predict(X_test)
        # Ensure positive travel times physically
        preds = np.clip(preds, a_min=1.0, a_max=None)
        test_predictions[name] = preds

        test_mae = mean_absolute_error(y_test, preds)
        test_rmse = root_mean_squared_error(y_test, preds)
        test_r2 = r2_score(y_test, preds)

        comparison_rows.append({
            "Model": name,
            "CV_MAE_Mean": round(cv_mae, 2),
            "CV_MAE_Std": round(cv_mae_std, 2),
            "CV_RMSE_Mean": round(cv_rmse, 2),
            "CV_R2_Mean": round(cv_r2, 3),
            "Test_MAE": round(test_mae, 2),
            "Test_RMSE": round(test_rmse, 2),
            "Test_R2": round(test_r2, 3),
        })

        logger.info(
            f"[{name}] CV MAE: {cv_mae:.2f} min | Test MAE: {test_mae:.2f} min | "
            f"Test RMSE: {test_rmse:.2f} min | Test R2: {test_r2:.3f}"
        )

    results_df = pd.DataFrame(comparison_rows)
    # Select best model based on lowest Validation (CV) MAE
    best_row = results_df.sort_values(by="CV_MAE_Mean").iloc[0]
    best_model_name = best_row["Model"]

    logger.info(f"Selected Best Model: {best_model_name} (CV MAE: {best_row['CV_MAE_Mean']} min)")

    if save_artifacts:
        # Save individual model pipelines
        for name, pipe in trained_pipelines.items():
            filename = name.lower().replace(" ", "_") + ".joblib"
            filepath = os.path.join(MODELS_DIR, filename)
            joblib.dump(pipe, filepath)

        # Save metadata and best model reference
        metadata = {
            "best_model_name": best_model_name,
            "strategy": strategy,
            "test_size": test_size,
            "random_state": random_state,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_cols": split_info["feature_cols"],
            "target_col": split_info["target_col"],
            "models_available": list(trained_pipelines.keys()),
        }
        joblib.dump(metadata, os.path.join(MODELS_DIR, "training_metadata.joblib"))

        # Save comparison results CSV
        csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved model comparison table to {csv_path}")

        # Save test evaluations for plotting & inspection
        eval_bundle = {
            "y_test": y_test.values,
            "test_predictions": test_predictions,
            "X_test": X_test,
            "results_df": results_df,
        }
        joblib.dump(eval_bundle, os.path.join(MODELS_DIR, "test_eval_bundle.joblib"))

    return trained_pipelines, results_df, split_info


if __name__ == "__main__":
    train_and_evaluate_all(strategy="stratified", save_artifacts=True)
