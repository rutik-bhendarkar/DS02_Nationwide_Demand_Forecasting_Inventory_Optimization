"""Run Part 5 model comparison, diagnostics and rolling-origin backtesting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demand_inventory.evaluation.backtesting import run_lightgbm_rolling_backtest  # noqa: E402
from demand_inventory.evaluation.comparison import run_model_comparison  # noqa: E402
from demand_inventory.forecasting.data import load_feature_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int, default=7, help="Test dates per expanding-origin fold")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--skip-backtest", action="store_true", help="Generate comparison without refitting rolling-origin folds")
    args = parser.parse_args()
    if args.window_days < 1:
        parser.error("--window-days must be at least 1")

    try:
        splits, metadata = load_feature_splits(ROOT)
        summary = run_model_comparison(ROOT, splits, metadata)
    except Exception as exc:
        print(f"PART 5 EVALUATION FAILED: {type(exc).__name__}: {exc}")
        return 1

    if args.skip_backtest:
        summary["rolling_origin_backtest"] = {"status": "skipped_by_user"}
    else:
        try:
            summary["rolling_origin_backtest"] = run_lightgbm_rolling_backtest(
                ROOT, splits, metadata, window_days=args.window_days, n_jobs=args.n_jobs
            )
        except Exception as exc:
            summary["rolling_origin_backtest"] = {
                "status": "unavailable_or_failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            print(f"ROLLING-ORIGIN BACKTEST NOT COMPLETED: {type(exc).__name__}: {exc}")

    summary_path = ROOT / "outputs/reports/part5_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nPART 5 EVALUATION COMPLETED")
    print(f"Selected model (validation WAPE): {summary['selected_model']}")
    print(f"Validation WAPE: {summary['selected_validation_metrics']['wape_pct']:.3f}%")
    print(f"Final test WAPE: {summary['selected_test_metrics']['wape_pct']:.3f}%")
    print(f"Report: {summary['files']['html_report']}")
    backtest = summary["rolling_origin_backtest"]
    print(f"Rolling-origin backtest status: {backtest.get('status')}")
    if backtest.get("status") == "success":
        print(f"Backtest folds: {backtest['fold_count']}")
        print(f"Backtest LightGBM WAPE: {backtest['overall_lightgbm_metrics']['wape_pct']:.3f}%")
        print(f"Backtest baseline WAPE: {backtest['overall_seasonal_naive_metrics']['wape_pct']:.3f}%")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
