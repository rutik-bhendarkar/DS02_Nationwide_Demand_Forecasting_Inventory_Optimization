from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_inventory.evaluation.backtesting import split_dates_into_windows
from demand_inventory.forecasting.metrics import calculate_regression_metrics


def test_split_dates_into_non_overlapping_windows() -> None:
    dates = pd.Series(pd.date_range("2025-01-01", periods=16, freq="D").append(
        pd.DatetimeIndex([pd.Timestamp("2025-01-01")])
    ))
    windows = split_dates_into_windows(dates, window_days=7)
    assert [len(window) for window in windows] == [7, 7, 2]
    flattened = [date for window in windows for date in window]
    assert len(flattened) == len(set(flattened)) == 16
    assert flattened == sorted(flattened)


def test_split_dates_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        split_dates_into_windows(pd.Series(pd.date_range("2025-01-01", periods=3)), 0)


def test_metrics_are_stable_for_constant_bias() -> None:
    result = calculate_regression_metrics([10, 20, 30], [12, 22, 32])
    assert result["mae"] == pytest.approx(2.0)
    assert result["bias_pct"] == pytest.approx(6 / 60 * 100)
    assert result["wape_pct"] == pytest.approx(6 / 60 * 100)


def test_baseline_7_day_lag_is_non_negative_in_test_split(project_root) -> None:
    test = pd.read_csv(project_root / "outputs/features/splits/test.csv")
    assert "units_sold_lag_7d" in test.columns
    assert np.isfinite(test["units_sold_lag_7d"]).all()
    assert (test["units_sold_lag_7d"] >= 0).all()


def test_prediction_file_preserves_model_name_and_unique_keys(project_root) -> None:
    from pathlib import Path

    from demand_inventory.evaluation.comparison import _load_prediction_file

    prediction = _load_prediction_file(
        Path(project_root) / "outputs/forecasts/test_lightgbm_predictions.csv", "test"
    )
    assert prediction.attrs["model_label"] == "LightGBM"
    assert not prediction.duplicated(["date", "store_id", "sku_id"]).any()
    assert (prediction["forecast_units_sold"] >= 0).all()


def test_segment_metrics_and_demand_tiers_cover_all_test_rows(project_root) -> None:
    from demand_inventory.evaluation.comparison import (
        _baseline_frame,
        _make_demand_tier_metrics,
        _make_segment_metrics,
    )
    from demand_inventory.forecasting.data import load_feature_splits

    splits, _ = load_feature_splits(project_root)
    predictions = _baseline_frame(splits["test"])
    prediction_map = {"Seasonal naive (7-day lag)": predictions}
    segments = _make_segment_metrics(splits["test"], prediction_map, "test")
    tiers = _make_demand_tier_metrics(splits["test"], prediction_map)
    assert {"region", "store_id", "sku_id"}.issubset(set(segments["dimension"]))
    assert int(tiers["n_observations"].sum()) == len(splits["test"])


def test_backtest_feature_matrix_keeps_categorical_codes_categorical() -> None:
    from demand_inventory.evaluation.backtesting import _model_matrices

    train = pd.DataFrame({"store_id_code": [0, 1], "lag": [4.0, 5.0]})
    future = pd.DataFrame({"store_id_code": [1, 0], "lag": [6.0, 7.0]})
    train_x, future_x = _model_matrices(
        train, future, ["store_id_code", "lag"], ["store_id_code"]
    )
    assert isinstance(train_x["store_id_code"].dtype, pd.CategoricalDtype)
    assert isinstance(future_x["store_id_code"].dtype, pd.CategoricalDtype)
    assert list(train_x.columns) == list(future_x.columns)
