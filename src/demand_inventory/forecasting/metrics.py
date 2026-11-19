"""Consistent regression metrics for demand forecasts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(actual, predicted) -> dict[str, float | int | None]:
    """Calculate MAE, RMSE, WAPE, sMAPE, bias and R2 with safe zero handling.

    WAPE is returned as a percentage. It is undefined when total actual demand is zero.
    sMAPE uses a zero-safe denominator and ignores rows where both values are zero.
    """
    y_true = np.asarray(actual, dtype=float).reshape(-1)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)
    if y_true.size == 0 or y_true.size != y_pred.size:
        raise ValueError("actual and predicted must be non-empty arrays of equal length")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("actual and predicted must contain only finite values")

    error = y_pred - y_true
    abs_error = np.abs(error)
    total_actual = float(np.abs(y_true).sum())
    wape = float(abs_error.sum() / total_actual * 100.0) if total_actual > 0 else None

    denominator = np.abs(y_true) + np.abs(y_pred)
    valid = denominator > 0
    smape = (
        float(np.mean(2.0 * abs_error[valid] / denominator[valid]) * 100.0)
        if valid.any()
        else 0.0
    )
    bias_pct = float(error.sum() / total_actual * 100.0) if total_actual > 0 else None
    r2 = float(r2_score(y_true, y_pred)) if y_true.size > 1 and np.var(y_true) > 0 else None

    return {
        "n_observations": int(y_true.size),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "wape_pct": wape,
        "smape_pct": smape,
        "bias_pct": bias_pct,
        "r2": r2,
    }
