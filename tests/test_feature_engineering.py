from __future__ import annotations

import pandas as pd
import pytest

from demand_inventory.features.engineering import (
    build_feature_frame,
    chronological_split,
    prepare_modeling_frame,
    run_feature_engineering,
)


def _frame(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "store_id", "sku_id", "units_sold"]).assign(
        unit_price=100.0,
        discount_pct=0.1,
        promo_flag=0,
        holiday_flag=0,
        region="North",
        category="Staples",
        opening_stock_units=50,
        replenishment_units=5,
        closing_stock_units=40,
        net_sales_value=1000.0,
    )


def test_calendar_and_known_future_features_are_created() -> None:
    frame = _frame([("2025-01-01", "S1", "P1", 10)])
    result, metadata = build_feature_frame(frame, lag_days=(1,), rolling_windows=(7,))
    row = result.iloc[0]
    assert row["month"] == 1
    assert row["day_of_week"] == 2  # Wednesday
    assert row["unit_price_known"] == pytest.approx(100.0)
    assert row["price_after_discount_known"] == pytest.approx(90.0)
    assert "month_sin" in metadata["feature_columns"]


def test_lags_join_exact_calendar_dates_not_previous_available_row() -> None:
    frame = _frame([
        ("2025-01-01", "S1", "P1", 10),
        ("2025-01-02", "S1", "P1", 20),
        ("2025-01-04", "S1", "P1", 40),
    ])
    result, _ = build_feature_frame(frame, lag_days=(1, 2), rolling_windows=(2,))
    jan4 = result.loc[result["date"].eq(pd.Timestamp("2025-01-04"))].iloc[0]
    assert pd.isna(jan4["units_sold_lag_1d"])
    assert jan4["units_sold_lag_2d"] == pytest.approx(20.0)


def test_rolling_features_exclude_target_date() -> None:
    frame = _frame([
        ("2025-01-01", "S1", "P1", 10),
        ("2025-01-02", "S1", "P1", 100),
        ("2025-01-04", "S1", "P1", 30),
    ])
    result, _ = build_feature_frame(frame, lag_days=(1,), rolling_windows=(7,))
    jan4 = result.loc[result["date"].eq(pd.Timestamp("2025-01-04"))].iloc[0]
    assert jan4["units_sold_rolling_7d_mean"] == pytest.approx(55.0)
    assert jan4["units_sold_rolling_7d_count"] == pytest.approx(2.0)


def test_target_inventory_and_same_day_outcome_not_in_feature_columns() -> None:
    frame = _frame([("2025-01-01", "S1", "P1", 10)])
    result, metadata = build_feature_frame(frame, lag_days=(1,), rolling_windows=(7,))
    assert "target_units_sold" in result.columns
    assert "target_units_sold" not in metadata["feature_columns"]
    assert "units_sold" not in result.columns
    assert not set(("opening_stock_units", "replenishment_units", "closing_stock_units", "net_sales_value")) & set(result.columns)


def test_build_does_not_mutate_input() -> None:
    frame = _frame([("2025-01-01", "S1", "P1", 10)])
    original = frame.copy(deep=True)
    build_feature_frame(frame, lag_days=(1,), rolling_windows=(7,))
    pd.testing.assert_frame_equal(frame, original)


def test_feature_frame_rejects_duplicate_store_sku_date() -> None:
    frame = _frame([
        ("2025-01-01", "S1", "P1", 10),
        ("2025-01-01", "S1", "P1", 20),
    ])
    with pytest.raises(ValueError, match="unique"):
        build_feature_frame(frame, lag_days=(1,), rolling_windows=(7,))


def test_feature_frame_rejects_invalid_arguments() -> None:
    frame = _frame([("2025-01-01", "S1", "P1", 10)])
    with pytest.raises(ValueError, match="positive integers"):
        build_feature_frame(frame, lag_days=(0,), rolling_windows=(7,))


def test_modeling_frame_requires_complete_direct_lag_history() -> None:
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    frame = _frame([(d.strftime("%Y-%m-%d"), "S1", "P1", float(i + 1)) for i, d in enumerate(dates)])
    features, metadata = build_feature_frame(frame, lag_days=(1, 7, 14, 28), rolling_windows=(7, 14, 28))
    ready = prepare_modeling_frame(features, metadata, min_history_days=28)
    assert len(ready) == 7
    assert ready["date"].min() == pd.Timestamp("2025-01-29")
    assert ready[[f"units_sold_lag_{d}d" for d in metadata["lag_days"]]].notna().all().all()


def test_pipeline_writes_feature_outputs(project_root, tmp_path) -> None:
    source = project_root / "data/sample/nationwide_retail_daily_sales_synthetic.csv"
    raw = pd.read_csv(source)
    result = run_feature_engineering(
        raw,
        tmp_path / "features",
        data_source_note="Synthetic test fixture; not real retailer data.",
    )
    assert result["status"] == "success"
    assert (tmp_path / "features/demand_features.csv").is_file()
    assert (tmp_path / "features/modeling_dataset.csv").is_file()
    assert (tmp_path / "features/feature_metadata.json").is_file()
    assert (tmp_path / "features/feature_engineering_summary.json").is_file()
    assert (tmp_path / "features/splits/train.csv").is_file()
    assert (tmp_path / "features/splits/validation.csv").is_file()
    assert (tmp_path / "features/splits/test.csv").is_file()
    assert result["feature_rows"] == len(raw)
    assert result["modeling_rows"] > 0
    assert result["modeling_feature_missing_cells"] == 0
    assert result["modeling_rows_with_missing_features"] == 0


def test_modeling_frame_has_no_missing_declared_features() -> None:
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    frame = _frame([(d.strftime("%Y-%m-%d"), "S1", "P1", float(i + 1)) for i, d in enumerate(dates)])
    features, metadata = build_feature_frame(frame)
    ready = prepare_modeling_frame(features, metadata, min_history_days=28)
    assert ready[metadata["feature_columns"]].notna().all().all()


def test_modeling_frame_requires_matching_rolling_window() -> None:
    frame = _frame([("2025-01-01", "S1", "P1", 10)])
    features, metadata = build_feature_frame(frame, lag_days=(1,), rolling_windows=(7,))
    with pytest.raises(ValueError, match="rolling_windows must include"):
        prepare_modeling_frame(features, metadata, min_history_days=14)


def test_chronological_split_keeps_all_rows_for_a_date_together() -> None:
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    rows = []
    for date in dates:
        rows.extend([
            (date.strftime("%Y-%m-%d"), "S1", "P1", 10.0),
            (date.strftime("%Y-%m-%d"), "S2", "P1", 20.0),
        ])
    frame = _frame(rows)
    splits, summary = chronological_split(frame)
    assert set(splits) == {"train", "validation", "test"}
    train_end = splits["train"]["date"].max()
    validation_start = splits["validation"]["date"].min()
    validation_end = splits["validation"]["date"].max()
    test_start = splits["test"]["date"].min()
    assert train_end < validation_start <= validation_end < test_start
    for part in splits.values():
        assert part.groupby("date").size().eq(2).all()
    assert summary["method"].startswith("chronological split")


def test_chronological_split_rejects_too_few_dates() -> None:
    frame = _frame([("2025-01-01", "S1", "P1", 10), ("2025-01-02", "S1", "P1", 12)])
    with pytest.raises(ValueError, match="At least three distinct"):
        chronological_split(frame)
