"""Load the leakage-aware Part 3 chronological model splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

TARGET_COLUMN = "target_units_sold"
IDENTIFIER_COLUMNS = [
    "date", "store_id", "store_name", "city", "state", "region",
    "sku_id", "product_name", "category",
]


def load_feature_splits(root: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    root = Path(root)
    metadata_path = root / "outputs/features/feature_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run scripts/run_feature_engineering.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = metadata.get("feature_columns", [])
    if not feature_columns:
        raise ValueError("Feature metadata does not declare feature_columns")

    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "validation", "test"):
        path = root / f"outputs/features/splits/{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Re-run Part 3 feature engineering.")
        frame = pd.read_csv(path, parse_dates=["date"])
        required = [*feature_columns, TARGET_COLUMN, "date", "store_id", "sku_id"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} split is missing required columns: {missing}")
        if frame[feature_columns].isna().any().any():
            raise ValueError(f"{name} split contains missing model features")
        if frame[TARGET_COLUMN].isna().any():
            raise ValueError(f"{name} split contains missing targets")
        if (frame[TARGET_COLUMN] < 0).any():
            raise ValueError(f"{name} split contains negative demand targets")
        splits[name] = frame

    # Guard against date overlap between partitions.
    date_sets = {name: set(frame["date"].dt.normalize().unique()) for name, frame in splits.items()}
    names = list(date_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = date_sets[left].intersection(date_sets[right])
            if overlap:
                raise ValueError(f"Chronological split date leakage between {left} and {right}")
    if max(date_sets["train"]) >= min(date_sets["validation"]):
        raise ValueError("Training dates must end before validation dates begin")
    if max(date_sets["validation"]) >= min(date_sets["test"]):
        raise ValueError("Validation dates must end before test dates begin")
    return splits, metadata
