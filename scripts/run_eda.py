"""Run the DS-02 Part 2 exploratory data analysis pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demand_inventory.analysis.eda import run_eda  # noqa: E402
from demand_inventory.data.ingestion import load_sales_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(ROOT / "data/sample/nationwide_retail_daily_sales_synthetic.csv"),
        help="Input sales CSV path",
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs/eda"), help="EDA table/chart output folder"
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of top entities in report")
    parser.add_argument(
        "--source-note",
        default="Bundled synthetic development dataset; not real retailer data.",
        help="Data provenance note written into the report",
    )
    args = parser.parse_args()

    try:
        frame = load_sales_csv(args.input, strict=True)
        summary = run_eda(
            frame,
            args.output_dir,
            top_n=args.top_n,
            data_source_note=args.source_note,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"EDA FAILED: {exc}")
        return 2

    print(json.dumps({
        "status": "success",
        "rows": summary["row_count"],
        "date_range": summary["date_range"],
        "stores": summary["cardinality"]["store_id"],
        "skus": summary["cardinality"]["sku_id"],
        "regions": summary["region_count"],
        "outlier_observations": summary["outliers"]["outlier_observations"],
        "interactive_dashboard_created": summary["interactive_dashboard_created"],
        "output_dir": str(Path(args.output_dir).resolve()),
        "summary_report": str((ROOT / "reports/part2_eda_summary.json").resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
