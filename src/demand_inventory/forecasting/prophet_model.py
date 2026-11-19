"""Per-store/SKU Prophet adapter for the DS-02 daily demand panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand_inventory.forecasting.common import persist_model_evaluation

GROUP_COLUMNS = ["store_id", "sku_id"]
REGRESSOR_CANDIDATES = [
    "unit_price_known", "discount_pct_known", "promo_flag_known", "holiday_flag_known",
    "price_after_discount_known", "promo_discount_interaction",
]


def run_prophet(
    splits: dict[str, pd.DataFrame], metadata: dict[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Fit one Prophet model per series on training data and predict val/test dates.

    Prophet receives the `ds`/`y` schema required by its documented Python API. Known-future
    regressors are used only where they are present, finite and non-constant in training.
    """
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError("Prophet is not installed. Install requirements-prophet-lightgbm.txt.") from exc

    train = splits["train"].copy()
    evaluation = pd.concat([splits["validation"], splits["test"]], ignore_index=True)
    regressors = [
        column for column in REGRESSOR_CANDIDATES
        if column in train.columns and column in evaluation.columns
        and train[column].nunique(dropna=True) > 1
        and not train[column].isna().any() and not evaluation[column].isna().any()
    ]
    predictions: dict[str, list[pd.DataFrame]] = {"validation": [], "test": []}
    fitted_series = 0
    skipped_series: list[str] = []

    for key, future_group in evaluation.groupby(GROUP_COLUMNS, sort=False, observed=True):
        history = train.loc[
            (train["store_id"] == key[0]) & (train["sku_id"] == key[1])
        ].sort_values("date")
        future_group = future_group.sort_values("date")
        if len(history) < 28 or future_group.empty:
            skipped_series.append(f"{key[0]}::{key[1]}")
            continue
        use_regressors = [column for column in regressors if history[column].nunique(dropna=True) > 1]
        prophet_model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            uncertainty_samples=0,
        )
        for column in use_regressors:
            prophet_model.add_regressor(column, standardize="auto")
        history_frame = history[["date", "target_units_sold", *use_regressors]].rename(
            columns={"date": "ds", "target_units_sold": "y"}
        )
        future_frame = future_group[["date", *use_regressors]].rename(columns={"date": "ds"})
        prophet_model.fit(history_frame)
        forecast = prophet_model.predict(future_frame)
        records = future_group[["date", *GROUP_COLUMNS]].copy()
        records["prediction"] = np.maximum(forecast["yhat"].to_numpy(dtype=float), 0.0)
        for split_name in ("validation", "test"):
            dates = set(splits[split_name]["date"].unique())
            predictions[split_name].append(records.loc[records["date"].isin(dates)].copy())
        fitted_series += 1

    if fitted_series == 0:
        raise RuntimeError("Prophet did not fit any store-SKU series; check training history.")
    prediction_arrays: dict[str, np.ndarray] = {}
    for split_name, parts in predictions.items():
        output = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        merged = splits[split_name][["date", *GROUP_COLUMNS]].merge(
            output, on=["date", *GROUP_COLUMNS], how="left", validate="one_to_one"
        )
        if merged["prediction"].isna().any():
            raise RuntimeError(f"Prophet predictions missing for {int(merged['prediction'].isna().sum())} rows")
        prediction_arrays[split_name] = merged["prediction"].to_numpy(dtype=float)

    return persist_model_evaluation(
        "Prophet",
        prediction_arrays,
        splits,
        output_root,
        extra={
            "series_fitted": fitted_series,
            "series_skipped": skipped_series,
            "known_future_regressors": regressors,
            "seasonality": {"weekly": True, "yearly": False, "daily": False},
            "data_source_note": metadata.get("data_source_note", "Not specified"),
        },
    )
