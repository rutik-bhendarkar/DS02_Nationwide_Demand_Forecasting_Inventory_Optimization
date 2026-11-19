"""Run Part 1 validation on a selected CSV file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demand_inventory.data.ingestion import inspect_sales_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(ROOT / "data/sample/nationwide_retail_daily_sales_synthetic.csv"),
    )
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    try:
        frame, report = inspect_sales_csv(args.input)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    print(json.dumps(report.to_dict(), indent=2))
    if report.is_valid:
        print("\nVALIDATION PASSED")
    else:
        print("\nVALIDATION FAILED")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Report saved: {report_path}")

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
