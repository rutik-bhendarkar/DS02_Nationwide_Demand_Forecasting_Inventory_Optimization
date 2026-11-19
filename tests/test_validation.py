import pandas as pd

from demand_inventory.data.validation import validate_sales_frame


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "store_id": ["ST001", "ST001"],
            "sku_id": ["SKU001", "SKU001"],
            "units_sold": [10, 12],
            "unit_price": [100.0, 100.0],
            "discount_pct": [0.0, 0.1],
        }
    )


def test_valid_frame_passes() -> None:
    report = validate_sales_frame(valid_frame())
    assert report.is_valid
    assert report.row_count == 2
    assert report.duplicate_business_key_count == 0


def test_missing_required_column_fails() -> None:
    frame = valid_frame().drop(columns=["sku_id"])
    report = validate_sales_frame(frame)
    assert not report.is_valid
    assert any("sku_id" in error for error in report.errors)


def test_negative_units_sold_fails() -> None:
    frame = valid_frame()
    frame.loc[0, "units_sold"] = -2
    report = validate_sales_frame(frame)
    assert not report.is_valid
    assert any("negative" in error for error in report.errors)


def test_discount_outside_decimal_range_fails() -> None:
    frame = valid_frame()
    frame.loc[0, "discount_pct"] = 15
    report = validate_sales_frame(frame)
    assert not report.is_valid
    assert any("discount_pct" in error for error in report.errors)


def test_duplicate_store_sku_date_fails() -> None:
    frame = valid_frame()
    frame.loc[1, "date"] = frame.loc[0, "date"]
    report = validate_sales_frame(frame)
    assert not report.is_valid
    assert report.duplicate_business_key_count == 1


def test_invalid_date_fails() -> None:
    frame = valid_frame()
    frame.loc[0, "date"] = "not-a-date"
    report = validate_sales_frame(frame)
    assert not report.is_valid
    assert any("invalid date" in error for error in report.errors)
