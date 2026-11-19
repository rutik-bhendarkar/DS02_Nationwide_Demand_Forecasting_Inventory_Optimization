"""PyTorch Forecasting Temporal Fusion Transformer adapter.

This optional adapter is designed for rolling one-step-ahead panel forecasting. It is kept
separate from core imports because PyTorch/Lightning are large platform-dependent packages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand_inventory.forecasting.common import persist_model_evaluation

GROUP_COLUMNS = ["store_id", "sku_id"]
KNOWN_REAL_CANDIDATES = [
    "time_idx", "month_sin", "month_cos", "weekday_sin", "weekday_cos", "is_weekend",
    "unit_price_known", "discount_pct_known", "promo_flag_known", "holiday_flag_known",
    "units_sold_lag_1d", "units_sold_lag_7d", "units_sold_rolling_7d_mean",
    "units_sold_rolling_28d_mean",
]


def run_pytorch_forecasting(
    splits: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
    output_root: str | Path,
    *,
    max_epochs: int = 5,
    batch_size: int = 128,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train a Temporal Fusion Transformer and predict validation/test one-day horizons."""
    try:
        import torch
        from lightning.pytorch import Trainer
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import MAE
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError(
            "PyTorch Forecasting stack is unavailable. Install a compatible PyTorch build, "
            "then requirements-pytorch-forecasting.txt."
        ) from exc

    torch.manual_seed(random_seed)
    train = splits["train"].copy()
    validation = splits["validation"].copy()
    test = splits["test"].copy()
    all_rows = pd.concat([train, validation, test], ignore_index=True)
    origin = all_rows["date"].min()
    for frame in (train, validation, test, all_rows):
        frame["date"] = pd.to_datetime(frame["date"])
        frame["time_idx"] = (frame["date"] - origin).dt.days.astype("int64")
        frame["series_id"] = frame["store_id"].astype(str) + "__" + frame["sku_id"].astype(str)
        frame["store_id"] = frame["store_id"].astype(str)
        frame["sku_id"] = frame["sku_id"].astype(str)

    known_reals = [
        column for column in KNOWN_REAL_CANDIDATES
        if column in train.columns and column in validation.columns and column in test.columns
    ]
    static_categoricals = [column for column in ("store_id", "sku_id", "region", "category") if column in train]
    for frame in (train, validation, test, all_rows):
        for column in static_categoricals:
            frame[column] = frame[column].astype(str).fillna("__MISSING__")

    training_dataset = TimeSeriesDataSet(
        train.sort_values(["series_id", "time_idx"]),
        time_idx="time_idx",
        target="target_units_sold",
        group_ids=["series_id"],
        max_encoder_length=28,
        min_encoder_length=28,
        max_prediction_length=1,
        min_prediction_length=1,
        static_categoricals=static_categoricals,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=["target_units_sold"],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )
    train_loader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )

    validation_context = pd.concat([train, validation], ignore_index=True).sort_values(
        ["series_id", "time_idx"]
    )
    test_context = pd.concat([train, validation, test], ignore_index=True).sort_values(
        ["series_id", "time_idx"]
    )
    validation_start = int(validation["time_idx"].min())
    test_start = int(test["time_idx"].min())
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        validation_context,
        min_prediction_idx=validation_start,
        stop_randomization=True,
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        test_context,
        min_prediction_idx=test_start,
        stop_randomization=True,
    )
    validation_loader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )
    test_loader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.003,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=MAE(),
        log_interval=-1,
        reduce_on_plateau_patience=3,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    def get_predictions(loader) -> pd.DataFrame:
        result = model.predict(loader, mode="prediction", return_index=True)
        prediction = result.output.detach().cpu().numpy()
        index = result.index.copy()
        prediction = prediction.reshape(prediction.shape[0], -1)[:, 0]
        if "time_idx" not in index or "series_id" not in index:
            raise RuntimeError("PyTorch Forecasting prediction index lacks time_idx/series_id")
        index["date"] = origin + pd.to_timedelta(index["time_idx"].astype(int), unit="D")
        index["store_id"] = index["series_id"].astype(str).str.split("__", n=1).str[0]
        index["sku_id"] = index["series_id"].astype(str).str.split("__", n=1).str[1]
        index["prediction"] = np.maximum(prediction, 0.0)
        return index[["date", *GROUP_COLUMNS, "prediction"]]

    predictions_by_split: dict[str, np.ndarray] = {}
    for split_name, loader in (("validation", validation_loader), ("test", test_loader)):
        prediction_frame = get_predictions(loader)
        source = splits[split_name][["date", *GROUP_COLUMNS]].copy()
        source["date"] = pd.to_datetime(source["date"])
        merged = source.merge(
            prediction_frame, on=["date", *GROUP_COLUMNS], how="left", validate="one_to_one"
        )
        if merged["prediction"].isna().any():
            raise RuntimeError(
                f"TFT predictions missing for {int(merged['prediction'].isna().sum())} {split_name} rows"
            )
        predictions_by_split[split_name] = merged["prediction"].to_numpy(dtype=float)

    model_dir = Path(output_root) / "models/trained"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pytorch_forecasting_tft.ckpt"
    model.save(str(model_path))
    return persist_model_evaluation(
        "PyTorch Forecasting TFT",
        predictions_by_split,
        splits,
        output_root,
        model_path=str(model_path),
        extra={
            "architecture": "TemporalFusionTransformer",
            "max_epochs": max_epochs,
            "max_encoder_length": 28,
            "max_prediction_length": 1,
            "known_reals": known_reals,
            "static_categoricals": static_categoricals,
            "training_note": "Rolling one-step-ahead panel forecasts using observed lag covariates.",
            "data_source_note": metadata.get("data_source_note", "Not specified"),
        },
    )
