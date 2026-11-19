"""Run the DS-02 Part 3 feature engineering pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demand_inventory.data.ingestion import load_sales_csv  # noqa: E402
from demand_inventory.features.engineering import run_feature_engineering  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(ROOT / "data/sample/nationwide_retail_daily_sales_synthetic.csv"),
        help="Input daily store-SKU sales CSV",
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs/features"), help="Feature output folder"
    )
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 7, 14, 28])
    parser.add_argument("--rolling-windows", nargs="+", type=int, default=[7, 14, 28])
    parser.add_argument("--min-history-days", type=int, default=28)
    parser.add_argument(
        "--source-note",
        default="Bundled synthetic development dataset; not real retailer data.",
    )
    args = parser.parse_args()
    try:
        frame = load_sales_csv(args.input, strict=True)
        summary = run_feature_engineering(
            frame,
            args.output_dir,
            lag_days=args.lags,
            rolling_windows=args.rolling_windows,
            min_history_days=args.min_history_days,
            data_source_note=args.source_note,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"FEATURE ENGINEERING FAILED: {exc}")
        return 2
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
