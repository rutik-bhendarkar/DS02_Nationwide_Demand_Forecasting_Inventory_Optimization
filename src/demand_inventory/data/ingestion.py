"""CSV ingestion entry points for retail sales and inventory data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from demand_inventory.data.validation import (
    DatasetValidationError,
    ValidationReport,
    validate_sales_frame,
)


def inspect_sales_csv(path: str | Path) -> tuple[pd.DataFrame, ValidationReport]:
    """Read a CSV and return its DataFrame plus a validation report."""
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")
    if not csv_path.is_file():
        raise IsADirectoryError(f"Input path is not a file: {csv_path}")

    frame = pd.read_csv(csv_path)
    report = validate_sales_frame(frame)
    return frame, report


def load_sales_csv(path: str | Path, *, strict: bool = True) -> pd.DataFrame:
    """Load a sales CSV; raise ``DatasetValidationError`` when strict and invalid."""
    frame, report = inspect_sales_csv(path)
    if strict and not report.is_valid:
        raise DatasetValidationError(report)
    return frame
