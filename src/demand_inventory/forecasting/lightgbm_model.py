"""LightGBM global panel regressor using Part 3 lag/calendar features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from demand_inventory.forecasting.data import TARGET_COLUMN
from demand_inventory.forecasting.common import persist_model_evaluation
from demand_inventory.forecasting.metrics import calculate_regression_metrics


def _feature_matrices(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    categorical = [column for column in feature_columns if column.endswith("_code")]
    matrices: list[pd.DataFrame] = []
    for frame in (train, validation, test):
        matrix = frame[feature_columns].copy()
        for column in feature_columns:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
        for column in categorical:
            categories = sorted(
                set(train[column].dropna().tolist())
                | set(validation[column].dropna().tolist())
                | set(test[column].dropna().tolist())
            )
            matrix[column] = pd.Categorical(matrix[column], categories=categories)
        if matrix.isna().any().any():
            raise ValueError("Feature matrix contains missing or non-numeric values")
        matrices.append(matrix)
    return matrices[0], matrices[1], matrices[2], categorical


def run_lightgbm(
    splits: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
    output_root: str | Path,
    *,
    random_state: int = 42,
    n_estimators: int = 600,
    n_jobs: int = 4,
) -> dict[str, Any]:
    """Train, validate and test LightGBM; persist model, forecasts and metrics."""
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError(
            "LightGBM is not installed. Install with: pip install -r requirements-prophet-lightgbm.txt"
        ) from exc

    feature_columns = list(metadata["feature_columns"])
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    X_train, X_validation, X_test, categorical_features = _feature_matrices(
        train, validation, test, feature_columns
    )
    y_train = train[TARGET_COLUMN].astype(float)
    y_validation = validation[TARGET_COLUMN].astype(float)
    y_test = test[TARGET_COLUMN].astype(float)

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    callbacks = [lgb.early_stopping(stopping_rounds=40, verbose=False), lgb.log_evaluation(period=0)]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        eval_metric="l1",
        categorical_feature=categorical_features or "auto",
        callbacks=callbacks,
    )

    validation_prediction = np.maximum(model.predict(X_validation), 0.0)
    test_prediction = np.maximum(model.predict(X_test), 0.0)
    baseline_validation = validation["units_sold_lag_7d"].to_numpy(dtype=float)
    baseline_test = test["units_sold_lag_7d"].to_numpy(dtype=float)

    output_root = Path(output_root)
    model_dir = output_root / "models/trained"
    forecast_dir = output_root / "outputs/forecasts"
    reports_dir = output_root / "outputs/reports"
    for directory in (model_dir, forecast_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "lightgbm_demand_regressor.joblib"
    joblib.dump(model, model_path)
    metrics = persist_model_evaluation(
        "LightGBM",
        {"validation": validation_prediction, "test": test_prediction},
        splits,
        output_root,
        model_path=str(model_path),
        extra={
            "target": TARGET_COLUMN,
            "feature_count": len(feature_columns),
            "features": feature_columns,
            "categorical_code_features": categorical_features,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "best_iteration": int(model.best_iteration_ or n_estimators),
            "baseline_7_day_lag_validation": calculate_regression_metrics(y_validation, baseline_validation),
            "baseline_7_day_lag_test": calculate_regression_metrics(y_test, baseline_test),
            "training_note": (
                "Trained only on the chronological training split. Validation is used for early stopping. "
                "Test metrics are reported once on the untouched chronological test split. "
                "Predictions are rolling one-step-ahead: Part 3 lag features contain prior observed demand."
            ),
            "data_source_note": metadata.get("data_source_note", "Not specified"),
        },
    )
    metrics_path = reports_dir / "part4_lightgbm_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    booster = model.booster_
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    importance_path = reports_dir / "lightgbm_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    metrics["metrics_path"] = metrics_path.relative_to(output_root.resolve()).as_posix()
    metrics["feature_importance_path"] = importance_path.relative_to(output_root.resolve()).as_posix()
    return metrics
