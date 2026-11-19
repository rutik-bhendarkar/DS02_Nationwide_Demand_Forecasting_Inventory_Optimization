"""Run DS-02 Part 4 demand forecasting models on Part 3 chronological splits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demand_inventory.forecasting.data import load_feature_splits  # noqa: E402
from demand_inventory.forecasting.darts_model import run_darts  # noqa: E402
from demand_inventory.forecasting.lightgbm_model import run_lightgbm  # noqa: E402
from demand_inventory.forecasting.prophet_model import run_prophet  # noqa: E402
from demand_inventory.forecasting.pytorch_forecasting_model import run_pytorch_forecasting  # noqa: E402
from demand_inventory.forecasting.registry import get_model_availability  # noqa: E402
from demand_inventory.forecasting.reporting import build_completion_summary, build_part4_reports  # noqa: E402

MODEL_RUNNERS = {
    "lightgbm": run_lightgbm,
    "prophet": run_prophet,
    "darts": run_darts,
    "pytorch_forecasting": run_pytorch_forecasting,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*MODEL_RUNNERS, "all"],
        default=["lightgbm"],
        help="Models to train. Default runs LightGBM; 'all' attempts every optional framework.",
    )
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--tft-epochs", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        splits, metadata = load_feature_splits(ROOT)
    except (OSError, ValueError, KeyError) as exc:
        print(f"FORECASTING INPUT ERROR: {exc}")
        print("Run scripts/run_feature_engineering.py first.")
        return 2

    requested = list(MODEL_RUNNERS) if "all" in args.models else args.models
    availability = get_model_availability()
    results: dict[str, object] = {
        "part": "Part 4 — Forecasting Models",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source_note": metadata.get("data_source_note", "Not specified"),
        "forecast_protocol": (
            "Train split only for fitting; validation split for model selection/early stopping; "
            "test split held out for final evaluation. Forecasts are rolling one-step-ahead."
        ),
        "availability": availability,
        "models": {},
    }
    failed = False
    for model_name in requested:
        print(f"\n=== {model_name.upper()} ===")
        if not availability[model_name]["available"]:
            detail = {
                "status": "not_installed",
                "message": str(availability[model_name]["install_instruction"]),
            }
            results["models"][model_name] = detail  # type: ignore[index]
            print(f"SKIPPED: {detail['message']}")
            continue
        try:
            if model_name == "lightgbm":
                model_result = run_lightgbm(
                    splits,
                    metadata,
                    ROOT,
                    random_state=args.random_state,
                    n_estimators=args.n_estimators,
                    n_jobs=args.n_jobs,
                )
            elif model_name == "pytorch_forecasting":
                model_result = run_pytorch_forecasting(
                    splits, metadata, ROOT, max_epochs=args.tft_epochs, random_seed=args.random_state
                )
            else:
                model_result = MODEL_RUNNERS[model_name](splits, metadata, ROOT)
            results["models"][model_name] = model_result  # type: ignore[index]
            print(json.dumps(model_result, indent=2, default=str))
        except Exception as exc:  # keep other models' statuses visible if one optional stack fails
            failed = True
            results["models"][model_name] = {  # type: ignore[index]
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            print(f"FAILED: {type(exc).__name__}: {exc}")

    report_dir = ROOT / "outputs/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_paths = build_part4_reports(ROOT, results["models"])  # type: ignore[arg-type]
    results["generated_reports"] = report_paths
    status_path = report_dir / "part4_model_status.json"
    status_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    summary_path = build_completion_summary(ROOT, results)
    print(f"\nModel status report: {status_path}")
    print(f"Completion summary: {summary_path}")
    for label, path in report_paths.items():
        print(f"{label}: {path}")

    trained = [
        name for name, result in results["models"].items()  # type: ignore[union-attr]
        if isinstance(result, dict) and result.get("status") == "trained"
    ]
    print(f"Trained models: {trained or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
