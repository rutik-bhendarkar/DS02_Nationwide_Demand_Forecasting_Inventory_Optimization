"""Darts ExponentialSmoothing adapter for multiple store-SKU demand series."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand_inventory.forecasting.common import persist_model_evaluation

GROUP_COLUMNS = ["store_id", "sku_id"]


def run_darts(
    splits: dict[str, pd.DataFrame], metadata: dict[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Fit Darts ExponentialSmoothing per store-SKU series and predict val/test dates."""
    try:
        from darts import TimeSeries
        from darts.models import ExponentialSmoothing
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError("Darts is not installed. Install requirements-darts.txt.") from exc

    train = splits["train"].copy()
    evaluation = pd.concat([splits["validation"], splits["test"]], ignore_index=True)
    predictions: dict[str, list[pd.DataFrame]] = {"validation": [], "test": []}
    fitted_series = 0
    for key, future_group in evaluation.groupby(GROUP_COLUMNS, sort=False, observed=True):
        history = train.loc[
            (train["store_id"] == key[0]) & (train["sku_id"] == key[1])
        ].sort_values("date")
        future_group = future_group.sort_values("date")
        if len(history) < 28 or future_group.empty:
            continue
        expected_history_dates = pd.date_range(history["date"].min(), history["date"].max(), freq="D")
        expected_future_dates = pd.date_range(
            history["date"].max() + pd.Timedelta(days=1), future_group["date"].max(), freq="D"
        )
        if not history["date"].reset_index(drop=True).equals(pd.Series(expected_history_dates)):
            raise ValueError(f"Darts requires a contiguous daily training index for {key}")
        if not future_group["date"].reset_index(drop=True).equals(pd.Series(expected_future_dates)):
            raise ValueError(f"Darts requires a contiguous daily forecast index for {key}")

        series_frame = history[["date", "target_units_sold"]].rename(
            columns={"date": "ds", "target_units_sold": "y"}
        )
        series = TimeSeries.from_dataframe(series_frame, time_col="ds", value_cols="y", freq="D")
        model = ExponentialSmoothing()
        model.fit(series)
        forecast = model.predict(len(future_group))
        values = forecast.values().reshape(-1)
        if len(values) != len(future_group):
            raise RuntimeError(f"Darts forecast length mismatch for {key}")
        records = future_group[["date", *GROUP_COLUMNS]].copy()
        records["prediction"] = np.maximum(values, 0.0)
        for split_name in ("validation", "test"):
            dates = set(splits[split_name]["date"].unique())
            predictions[split_name].append(records.loc[records["date"].isin(dates)].copy())
        fitted_series += 1

    if fitted_series == 0:
        raise RuntimeError("Darts did not fit any eligible store-SKU series.")
    prediction_arrays: dict[str, np.ndarray] = {}
    for split_name, parts in predictions.items():
        output = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        merged = splits[split_name][["date", *GROUP_COLUMNS]].merge(
            output, on=["date", *GROUP_COLUMNS], how="left", validate="one_to_one"
        )
        if merged["prediction"].isna().any():
            raise RuntimeError(f"Darts predictions missing for {int(merged['prediction'].isna().sum())} rows")
        prediction_arrays[split_name] = merged["prediction"].to_numpy(dtype=float)

    return persist_model_evaluation(
        "Darts ExponentialSmoothing",
        prediction_arrays,
        splits,
        output_root,
        extra={
            "series_fitted": fitted_series,
            "model_type": "Darts ExponentialSmoothing per store-SKU series",
            "uses_future_covariates": False,
            "data_source_note": metadata.get("data_source_note", "Not specified"),
        },
    )
