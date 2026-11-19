from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_inventory.forecasting.data import load_feature_splits
from demand_inventory.forecasting.metrics import calculate_regression_metrics
from demand_inventory.forecasting.registry import get_model_availability


def test_metrics_perfect_predictions_are_zero_error() -> None:
    metrics = calculate_regression_metrics([1, 2, 3], [1, 2, 3])
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["wape_pct"] == 0
    assert metrics["smape_pct"] == 0
    assert metrics["bias_pct"] == 0


def test_metrics_handle_zero_actuals() -> None:
    metrics = calculate_regression_metrics([0, 0], [0, 2])
    assert metrics["mae"] == 1
    assert metrics["wape_pct"] is None
    assert metrics["bias_pct"] is None
    assert metrics["smape_pct"] == 200


def test_metrics_reject_mismatched_or_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        calculate_regression_metrics([1, 2], [1])
    with pytest.raises(ValueError):
        calculate_regression_metrics([1, np.nan], [1, 2])


def test_part3_splits_load_and_are_chronological(project_root) -> None:
    splits, metadata = load_feature_splits(project_root)
    assert metadata["feature_count"] == len(metadata["feature_columns"])
    assert set(splits) == {"train", "validation", "test"}
    assert splits["train"]["date"].max() < splits["validation"]["date"].min()
    assert splits["validation"]["date"].max() < splits["test"]["date"].min()
    assert all(not frame[metadata["feature_columns"]].isna().any().any() for frame in splits.values())


def test_model_availability_registry_has_all_required_frameworks() -> None:
    statuses = get_model_availability()
    assert set(statuses) == {"lightgbm", "prophet", "darts", "pytorch_forecasting"}
    assert all("install_instruction" in status for status in statuses.values())


def test_prediction_records_reject_wrong_length() -> None:
    from demand_inventory.forecasting.common import build_prediction_records

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "store_id": ["ST001"],
            "sku_id": ["SKU001"],
            "target_units_sold": [10.0],
        }
    )
    with pytest.raises(ValueError):
        build_prediction_records(frame, np.array([1.0, 2.0]), "test", "test")
