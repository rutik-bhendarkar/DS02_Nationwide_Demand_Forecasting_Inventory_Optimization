"""Leakage-aware feature engineering for daily store-SKU demand.

The output is a *one-step-ahead target-date panel*: each row represents a
store/SKU/date to predict. Historical sales features use dates strictly before
the row date. Same-date price, discount, promotion and holiday fields are only
included as known-future covariates; they must be available from a planning
calendar/catalog when predictions are made.

Inventory fields measured during/after the target date are deliberately not
used as model features. The target itself is retained as ``target_units_sold``
for supervised training and excluded from ``feature_columns`` metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("date", "store_id", "sku_id", "units_sold")
DEFAULT_LAG_DAYS = (1, 7, 14, 28)
DEFAULT_ROLLING_WINDOWS = (7, 14, 28)
DEFAULT_KNOWN_FUTURE_COLUMNS = ("unit_price", "discount_pct", "promo_flag", "holiday_flag")
STATIC_CATEGORICAL_COLUMNS = ("region", "state", "city", "category", "supplier_id")
BLOCKED_OUTCOME_OR_INVENTORY_COLUMNS = (
    "opening_stock_units",
    "replenishment_units",
    "closing_stock_units",
    "net_sales_value",
)


def _validate_arguments(lag_days: Iterable[int], rolling_windows: Iterable[int]) -> tuple[list[int], list[int]]:
    lags = sorted(set(int(value) for value in lag_days))
    windows = sorted(set(int(value) for value in rolling_windows))
    if not lags or any(value <= 0 for value in lags):
        raise ValueError("lag_days must contain positive integers")
    if not windows or any(value <= 0 for value in windows):
        raise ValueError("rolling_windows must contain positive integers")
    return lags, windows


def _prepare_input(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature engineering requires columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Feature engineering input is empty")

    data = frame.copy(deep=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce", format="mixed").dt.normalize()
    if data["date"].isna().any():
        raise ValueError("Feature engineering found invalid dates")
    for column in ("store_id", "sku_id"):
        if data[column].isna().any() or data[column].astype("string").str.strip().eq("").any():
            raise ValueError(f"Feature engineering found missing/blank {column} values")
        data[column] = data[column].astype(str)
    data["units_sold"] = pd.to_numeric(data["units_sold"], errors="coerce")
    if data["units_sold"].isna().any() or (data["units_sold"] < 0).any():
        raise ValueError("units_sold must contain non-missing, non-negative numeric values")

    duplicate_mask = data.duplicated(["date", "store_id", "sku_id"], keep=False)
    if duplicate_mask.any():
        raise ValueError(
            "Feature engineering requires unique (date, store_id, sku_id) rows; "
            f"found {int(duplicate_mask.sum())} duplicate row(s)"
        )
    return data.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)


def _add_calendar_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    date = data["date"]
    iso = date.dt.isocalendar()
    start = data.groupby(["store_id", "sku_id"], observed=True)["date"].transform("min")
    data["year"] = date.dt.year.astype("int16")
    data["quarter"] = date.dt.quarter.astype("int8")
    data["month"] = date.dt.month.astype("int8")
    data["week_of_year"] = iso.week.astype("int16")
    data["day_of_month"] = date.dt.day.astype("int8")
    data["day_of_year"] = date.dt.dayofyear.astype("int16")
    data["day_of_week"] = date.dt.dayofweek.astype("int8")
    data["is_weekend"] = date.dt.dayofweek.isin([5, 6]).astype("int8")
    data["is_month_start"] = date.dt.is_month_start.astype("int8")
    data["is_month_end"] = date.dt.is_month_end.astype("int8")
    data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12)
    data["weekday_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    data["series_age_days"] = (date - start).dt.days.astype("int32")
    return data, [
        "year", "quarter", "month", "week_of_year", "day_of_month", "day_of_year",
        "day_of_week", "is_weekend", "is_month_start", "is_month_end", "month_sin",
        "month_cos", "weekday_sin", "weekday_cos", "series_age_days",
    ]


def _add_identifier_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, int]]]:
    columns = ["store_id", "sku_id", *[c for c in STATIC_CATEGORICAL_COLUMNS if c in data.columns]]
    feature_columns: list[str] = []
    mappings: dict[str, dict[str, int]] = {}
    for column in columns:
        values = data[column].astype("string").fillna("__MISSING__")
        levels = sorted(values.unique().tolist())
        mapping = {value: index for index, value in enumerate(levels)}
        code_column = f"{column}_code"
        data[code_column] = values.map(mapping).astype("int32")
        feature_columns.append(code_column)
        mappings[column] = mapping
    return data, feature_columns, mappings


def _add_known_future_features(
    data: pd.DataFrame, known_future_columns: Iterable[str] | None
) -> tuple[pd.DataFrame, list[str]]:
    requested = (
        list(DEFAULT_KNOWN_FUTURE_COLUMNS)
        if known_future_columns is None
        else list(known_future_columns)
    )
    allowed = set(DEFAULT_KNOWN_FUTURE_COLUMNS)
    unsupported = sorted(set(requested).difference(allowed))
    if unsupported:
        raise ValueError(
            "Only explicitly approved known-future covariates are supported: "
            + ", ".join(sorted(allowed))
        )

    features: list[str] = []
    for source in requested:
        if source not in data.columns:
            continue
        converted = pd.to_numeric(data[source], errors="coerce")
        name = f"{source}_known"
        data[name] = converted
        features.append(name)

    if "unit_price" in data.columns:
        price = pd.to_numeric(data["unit_price"], errors="coerce")
        discount = (
            pd.to_numeric(data["discount_pct"], errors="coerce").fillna(0)
            if "discount_pct" in data.columns
            else pd.Series(0.0, index=data.index)
        )
        data["price_after_discount_known"] = price * (1 - discount)
        features.append("price_after_discount_known")

    if {"promo_flag", "discount_pct"}.issubset(data.columns):
        promo = pd.to_numeric(data["promo_flag"], errors="coerce").fillna(0)
        discount = pd.to_numeric(data["discount_pct"], errors="coerce").fillna(0)
        data["promo_discount_interaction"] = promo * discount
        features.append("promo_discount_interaction")
    return data, features


def _add_lag_features(data: pd.DataFrame, lag_days: list[int]) -> tuple[pd.DataFrame, list[str]]:
    keys = ["date", "store_id", "sku_id"]
    source = data[keys + ["units_sold"]].copy()
    output_columns: list[str] = []
    for lag in lag_days:
        name = f"units_sold_lag_{lag}d"
        shifted = source.copy()
        # Move historical observation forward so it joins to its target date.
        shifted["date"] = shifted["date"] + pd.Timedelta(days=lag)
        shifted = shifted.rename(columns={"units_sold": name})
        data = data.merge(shifted, on=keys, how="left", validate="one_to_one", sort=False)
        output_columns.append(name)
    return data, output_columns


def _add_rolling_features(data: pd.DataFrame, rolling_windows: list[int]) -> tuple[pd.DataFrame, list[str]]:
    columns: list[str] = []
    for window in rolling_windows:
        names = [
            f"units_sold_rolling_{window}d_mean",
            f"units_sold_rolling_{window}d_std",
            f"units_sold_rolling_{window}d_sum",
            f"units_sold_rolling_{window}d_count",
        ]
        for name in names:
            data[name] = np.nan
        for _, group in data.groupby(["store_id", "sku_id"], sort=False, observed=True):
            ordered = group.sort_values("date")
            series = pd.Series(
                ordered["units_sold"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(ordered["date"]),
            )
            rolling = series.rolling(f"{window}D", min_periods=1, closed="left")
            rolling_std = series.rolling(f"{window}D", min_periods=2, closed="left")
            data.loc[ordered.index, names[0]] = rolling.mean().to_numpy()
            data.loc[ordered.index, names[1]] = rolling_std.std(ddof=1).to_numpy()
            data.loc[ordered.index, names[2]] = rolling.sum().to_numpy()
            data.loc[ordered.index, names[3]] = rolling.count().to_numpy()
        columns.extend(names)
    return data, columns


def build_feature_frame(
    frame: pd.DataFrame,
    *,
    lag_days: Iterable[int] = DEFAULT_LAG_DAYS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    known_future_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build target-date features using only past sales and known-future covariates."""
    lags, windows = _validate_arguments(lag_days, rolling_windows)
    data = _prepare_input(frame)
    data["target_units_sold"] = data["units_sold"].astype(float)

    data, calendar_features = _add_calendar_features(data)
    data, identifier_features, category_mappings = _add_identifier_features(data)
    data, known_features = _add_known_future_features(data, known_future_columns)
    data, lag_features = _add_lag_features(data, lags)
    data, rolling_features = _add_rolling_features(data, windows)

    # Do not allow same-day outcome or stock-state fields into model features.
    data = data.drop(columns=["units_sold", *[c for c in BLOCKED_OUTCOME_OR_INVENTORY_COLUMNS if c in data.columns]])
    feature_columns = [
        *identifier_features,
        *calendar_features,
        *known_features,
        *lag_features,
        *rolling_features,
    ]
    metadata: dict[str, Any] = {
        "part": "Part 3 — Feature Engineering",
        "target_column": "target_units_sold",
        "forecast_setup": "one-step-ahead target-date panel; one row per store-SKU-date",
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "feature_groups": {
            "identifier_codes": identifier_features,
            "calendar": calendar_features,
            "known_future_covariates": known_features,
            "historical_lags": lag_features,
            "historical_rolling_statistics": rolling_features,
        },
        "lag_days": lags,
        "rolling_windows_days": windows,
        "categorical_mappings": category_mappings,
        "known_future_assumption": (
            "Same-target-date price, discount, promotion and holiday inputs are only valid "
            "when they are known from the pricing/promotion/calendar plan at prediction time."
        ),
        "leakage_controls": [
            "Historical lag features join exact prior calendar dates, not prior available rows.",
            "Rolling statistics use a time window closed on the left, excluding the target date.",
            "Raw same-day units_sold is retained only as target_units_sold, never as a feature.",
            "Opening, replenishment and closing inventory plus same-day net sales are excluded.",
            "No interpolation or zero-filling is applied to missing historical observations.",
        ],
        "excluded_columns": [
            "units_sold", "opening_stock_units", "replenishment_units",
            "closing_stock_units", "net_sales_value",
        ],
    }
    ordered = data.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    return ordered, metadata


def prepare_modeling_frame(
    feature_frame: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    min_history_days: int = 28,
) -> pd.DataFrame:
    """Return rows with every configured direct lag available for model training."""
    if min_history_days < 1:
        raise ValueError("min_history_days must be at least 1")
    required_lags = [f"units_sold_lag_{lag}d" for lag in metadata["lag_days"]]
    missing = [column for column in required_lags if column not in feature_frame.columns]
    if missing:
        raise ValueError(f"Feature frame is missing required lag columns: {missing}")
    history_count_col = f"units_sold_rolling_{int(min_history_days)}d_count"
    if history_count_col not in feature_frame.columns:
        raise ValueError(
            f"rolling_windows must include min_history_days={min_history_days} "
            "to verify complete calendar-day history"
        )
    mask = feature_frame[required_lags].notna().all(axis=1)
    mask &= feature_frame[history_count_col].ge(int(min_history_days))
    return feature_frame.loc[mask].copy().reset_index(drop=True)



def chronological_split(
    frame: pd.DataFrame, *, train_ratio: float = 0.70, validation_ratio: float = 0.15
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Split complete target dates into train/validation/test without random shuffling.

    All store-SKU rows for a target date stay in the same partition. The split is
    performed over sorted unique dates, not row indices, to avoid cross-sectional
    date leakage between stores and products.
    """
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("train_ratio and validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")
    if "date" not in frame.columns or frame.empty:
        raise ValueError("Chronological splitting requires a non-empty frame with a date column")

    dates = pd.to_datetime(frame["date"], errors="coerce", format="mixed").dt.normalize()
    if dates.isna().any():
        raise ValueError("Chronological splitting found invalid dates")
    unique_dates = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
    if len(unique_dates) < 3:
        raise ValueError("At least three distinct target dates are required for train/validation/test splits")

    train_end = max(1, int(np.floor(len(unique_dates) * train_ratio)))
    validation_end = max(train_end + 1, int(np.floor(len(unique_dates) * (train_ratio + validation_ratio))))
    validation_end = min(validation_end, len(unique_dates) - 1)
    boundaries = {
        "train": unique_dates[:train_end],
        "validation": unique_dates[train_end:validation_end],
        "test": unique_dates[validation_end:],
    }
    splits: dict[str, pd.DataFrame] = {}
    summary: dict[str, Any] = {
        "method": "chronological split by unique target date; no row shuffling",
        "train_ratio_requested": train_ratio,
        "validation_ratio_requested": validation_ratio,
        "test_ratio_requested": 1 - train_ratio - validation_ratio,
        "total_unique_dates": int(len(unique_dates)),
        "partitions": {},
    }
    normalized = frame.copy()
    normalized["date"] = dates
    for name, partition_dates in boundaries.items():
        part = normalized.loc[normalized["date"].isin(partition_dates)].copy().reset_index(drop=True)
        if part.empty:
            raise ValueError(f"Chronological split produced an empty {name} partition")
        splits[name] = part
        summary["partitions"][name] = {
            "rows": int(len(part)),
            "unique_dates": int(part["date"].nunique()),
            "start_date": part["date"].min().strftime("%Y-%m-%d"),
            "end_date": part["date"].max().strftime("%Y-%m-%d"),
        }
    return splits, summary

def run_feature_engineering(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    lag_days: Iterable[int] = DEFAULT_LAG_DAYS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    min_history_days: int = 28,
    known_future_columns: Iterable[str] | None = None,
    data_source_note: str = "Source provenance not supplied.",
) -> dict[str, Any]:
    """Build and persist the complete Part 3 feature datasets and metadata."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features, metadata = build_feature_frame(
        frame,
        lag_days=lag_days,
        rolling_windows=rolling_windows,
        known_future_columns=known_future_columns,
    )
    modeling = prepare_modeling_frame(features, metadata, min_history_days=min_history_days)
    splits, split_summary = chronological_split(modeling)
    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_frame in splits.items():
        split_frame.to_csv(split_dir / f"{split_name}.csv", index=False)
    metadata["chronological_split"] = split_summary
    metadata["data_source_note"] = data_source_note
    metadata["row_count_full_feature_frame"] = int(len(features))
    metadata["row_count_modeling_frame"] = int(len(modeling))
    metadata["warmup_rows_excluded"] = int(len(features) - len(modeling))
    metadata["minimum_history_days"] = int(min_history_days)
    metadata["target_mean_units"] = float(modeling["target_units_sold"].mean()) if not modeling.empty else None

    features.to_csv(output / "demand_features.csv", index=False)
    modeling.to_csv(output / "modeling_dataset.csv", index=False)
    (output / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    summary = {
        "status": "success",
        "part": metadata["part"],
        "data_source_note": data_source_note,
        "input_rows": int(len(frame)),
        "feature_rows": int(len(features)),
        "modeling_rows": int(len(modeling)),
        "warmup_rows_excluded": int(len(features) - len(modeling)),
        "store_count": int(frame["store_id"].nunique()),
        "sku_count": int(frame["sku_id"].nunique()),
        "date_start": pd.to_datetime(frame["date"]).min().strftime("%Y-%m-%d"),
        "date_end": pd.to_datetime(frame["date"]).max().strftime("%Y-%m-%d"),
        "feature_count": int(metadata["feature_count"]),
        "target_column": metadata["target_column"],
        "missing_feature_cells_full_panel": int(features[metadata["feature_columns"]].isna().sum().sum()),
        "modeling_feature_missing_cells": int(modeling[metadata["feature_columns"]].isna().sum().sum()) if not modeling.empty else 0,
        "modeling_rows_with_missing_features": int(modeling[metadata["feature_columns"]].isna().any(axis=1).sum()) if not modeling.empty else 0,
        "lag_days": metadata["lag_days"],
        "rolling_windows_days": metadata["rolling_windows_days"],
        "output_files": [
            "demand_features.csv", "modeling_dataset.csv", "feature_metadata.json",
            "feature_engineering_summary.json", "splits/train.csv",
            "splits/validation.csv", "splits/test.csv",
        ],
        "chronological_split": split_summary,
        "leakage_controls": metadata["leakage_controls"],
    }
    (output / "feature_engineering_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
