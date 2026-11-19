from __future__ import annotations

import json

import pandas as pd
import pytest

from demand_inventory.analysis.eda import run_eda


@pytest.fixture
def small_sales_frame() -> pd.DataFrame:
    """Small, deterministic daily sales sample for EDA unit tests."""
    rows = [
        # day 1: region-wide demand = 12; SKU-A demand = 8; SKU-B demand = 4
        ("2025-01-01", "ST001", "SKU-A", 5, 10.0, "North", "Staples", 50.0, 0, 0, 20),
        ("2025-01-01", "ST002", "SKU-A", 3, 10.0, "North", "Staples", 30.0, 0, 0, 18),
        ("2025-01-01", "ST001", "SKU-B", 4, 20.0, "North", "Snacks", 80.0, 1, 0, 16),
        # day 2: region-wide demand = 12; SKU-A demand = 10; SKU-B demand = 2
        ("2025-01-02", "ST001", "SKU-A", 6, 10.0, "North", "Staples", 60.0, 0, 0, 17),
        ("2025-01-02", "ST002", "SKU-A", 4, 10.0, "North", "Staples", 40.0, 0, 1, 15),
        ("2025-01-02", "ST001", "SKU-B", 2, 20.0, "North", "Snacks", 40.0, 0, 1, 14),
    ]
    columns = [
        "date", "store_id", "sku_id", "units_sold", "unit_price", "region",
        "category", "net_sales_value", "promo_flag", "holiday_flag", "closing_stock_units",
    ]
    return pd.DataFrame(rows, columns=columns)


def test_eda_outputs_tables_summary_and_dashboard(small_sales_frame, tmp_path) -> None:
    output_dir = tmp_path / "outputs" / "eda"
    summary = run_eda(
        small_sales_frame,
        output_dir,
        data_source_note="Synthetic unit-test fixture; not real retailer data.",
    )

    assert summary["row_count"] == 6
    assert summary["date_range"]["calendar_days"] == 2
    assert summary["demand"]["total_units_sold"] == 24
    assert summary["quality"]["duplicate_business_keys"] == 0
    assert (output_dir / "daily_demand.csv").is_file()
    assert (output_dir / "region_performance.csv").is_file()
    assert (output_dir / "sku_performance.csv").is_file()
    assert (tmp_path / "reports" / "part2_eda_summary.json").is_file()
    assert (tmp_path / "reports" / "part2_eda_findings.md").is_file()
    persisted = json.loads((tmp_path / "reports" / "part2_eda_summary.json").read_text())
    assert persisted["data_source_note"].startswith("Synthetic unit-test")


def test_average_daily_units_is_daily_dimension_total(small_sales_frame, tmp_path) -> None:
    output_dir = tmp_path / "outputs" / "eda"
    run_eda(small_sales_frame, output_dir, data_source_note="Synthetic test fixture.")

    region = pd.read_csv(output_dir / "region_performance.csv")
    sku = pd.read_csv(output_dir / "sku_performance.csv")

    # Two stores contribute to each region-day: 5+3+4=12 and 6+4+2=12.
    assert region.loc[0, "average_daily_units"] == pytest.approx(12.0)
    # SKU-A daily totals are 8 and 10, so mean daily SKU demand is 9.
    sku_a = sku.loc[sku["sku_id"] == "SKU-A"].iloc[0]
    assert sku_a["average_daily_units"] == pytest.approx(9.0)
    # The per-record mean remains separately available and is not mislabeled as daily total.
    assert sku_a["average_units_per_record"] == pytest.approx(4.5)


def test_eda_does_not_mutate_input(small_sales_frame, tmp_path) -> None:
    original = small_sales_frame.copy(deep=True)
    run_eda(small_sales_frame, tmp_path / "outputs" / "eda")
    pd.testing.assert_frame_equal(small_sales_frame, original)


def test_eda_rejects_missing_required_column(small_sales_frame, tmp_path) -> None:
    invalid = small_sales_frame.drop(columns=["sku_id"])
    with pytest.raises(ValueError, match="sku_id"):
        run_eda(invalid, tmp_path / "outputs" / "eda")


def test_eda_rejects_invalid_dates(small_sales_frame, tmp_path) -> None:
    invalid = small_sales_frame.copy()
    invalid.loc[0, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="invalid dates"):
        run_eda(invalid, tmp_path / "outputs" / "eda")


def test_eda_rejects_non_positive_top_n(small_sales_frame, tmp_path) -> None:
    with pytest.raises(ValueError, match="top_n"):
        run_eda(small_sales_frame, tmp_path / "outputs" / "eda", top_n=0)
