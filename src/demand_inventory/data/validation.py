"""Validation rules for the daily store-SKU sales dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from demand_inventory.data.schema import (
    BUSINESS_KEY,
    NON_NEGATIVE_COLUMNS,
    NUMERIC_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
)


@dataclass
class ValidationReport:
    """Machine-readable dataset quality report."""

    is_valid: bool
    row_count: int
    column_count: int
    missing_value_count: int
    duplicate_business_key_count: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetValidationError(ValueError):
    """Raised when a dataset fails the required input contract."""

    def __init__(self, report: ValidationReport):
        self.report = report
        details = "; ".join(report.errors) or "Dataset validation failed"
        super().__init__(details)


def validate_sales_frame(frame: pd.DataFrame) -> ValidationReport:
    """Validate required fields, types, ranges, and row uniqueness.

    The function does not mutate ``frame``. It checks the input contract used by
    later forecasting and inventory-planning phases.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(frame, pd.DataFrame):
        return ValidationReport(
            is_valid=False,
            row_count=0,
            column_count=0,
            missing_value_count=0,
            duplicate_business_key_count=0,
            errors=["Input must be a pandas DataFrame."],
        )

    row_count, column_count = frame.shape
    missing_value_count = int(frame.isna().sum().sum())

    if frame.empty:
        errors.append("Dataset contains no rows.")

    missing_required = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing_required:
        errors.append(f"Missing required columns: {', '.join(missing_required)}.")
        return ValidationReport(
            is_valid=False,
            row_count=row_count,
            column_count=column_count,
            missing_value_count=missing_value_count,
            duplicate_business_key_count=0,
            errors=errors,
            warnings=warnings,
        )

    for column in REQUIRED_COLUMNS:
        if frame[column].isna().any():
            count = int(frame[column].isna().sum())
            errors.append(f"Required column '{column}' contains {count} missing value(s).")

    for column in ("store_id", "sku_id"):
        blank_mask = frame[column].astype("string").str.strip().eq("")
        if blank_mask.fillna(False).any():
            errors.append(f"Required identifier '{column}' contains blank value(s).")

    parsed_dates = pd.to_datetime(frame["date"], format="mixed", errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_dates:
        errors.append(f"Column 'date' contains {invalid_dates} invalid date value(s).")

    numeric_values: dict[str, pd.Series] = {}
    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        numeric_values[column] = converted
        invalid_numeric = int(converted.isna().sum() - frame[column].isna().sum())
        if invalid_numeric > 0:
            errors.append(
                f"Column '{column}' contains {invalid_numeric} non-numeric value(s)."
            )

    for column in NON_NEGATIVE_COLUMNS:
        if column in numeric_values:
            negative_count = int((numeric_values[column] < 0).sum())
            if negative_count:
                errors.append(f"Column '{column}' contains {negative_count} negative value(s).")

    if "discount_pct" in numeric_values:
        discount = numeric_values["discount_pct"]
        invalid_discount = int(((discount < 0) | (discount > 1)).sum())
        if invalid_discount:
            errors.append(
                "Column 'discount_pct' must use decimal proportions from 0 to 1 "
                f"(found {invalid_discount} out-of-range value(s))."
            )

    duplicate_count = 0
    if all(column in frame.columns for column in BUSINESS_KEY) and not invalid_dates:
        key_frame = frame.loc[:, list(BUSINESS_KEY)].copy()
        key_frame["date"] = parsed_dates
        duplicate_count = int(key_frame.duplicated(keep="first").sum())
        if duplicate_count:
            errors.append(
                f"Found {duplicate_count} duplicate store-SKU-date business key row(s)."
            )

    absent_optional = [column for column in OPTIONAL_COLUMNS if column not in frame.columns]
    if absent_optional:
        warnings.append(
            "Optional enrichment columns not supplied: " + ", ".join(absent_optional) + "."
        )

    if missing_value_count:
        warnings.append(
            f"Dataset contains {missing_value_count} missing cell(s) across all columns."
        )

    return ValidationReport(
        is_valid=not errors,
        row_count=row_count,
        column_count=column_count,
        missing_value_count=missing_value_count,
        duplicate_business_key_count=duplicate_count,
        errors=errors,
        warnings=warnings,
    )
