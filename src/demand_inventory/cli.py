"""Command-line interface for Part 1 data setup and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from demand_inventory.data.ingestion import inspect_sales_csv


def _print_report(report) -> None:
    print(json.dumps(report.to_dict(), indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demand-inventory",
        description="DS-02 data ingestion and quality checks",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a sales CSV file")
    validate_parser.add_argument("--input", required=True, help="Path to the input CSV")
    validate_parser.add_argument(
        "--json-output", default=None, help="Optional path to save the validation report"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate":
        try:
            _, report = inspect_sales_csv(args.input)
        except (FileNotFoundError, IsADirectoryError, OSError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 2

        _print_report(report)
        if args.json_output:
            output_path = Path(args.json_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            print(f"Report saved to: {output_path}")
        return 0 if report.is_valid else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
