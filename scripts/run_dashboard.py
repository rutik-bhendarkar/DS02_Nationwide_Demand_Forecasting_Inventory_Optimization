"""Start the DS-02 Part 7 Plotly Dash dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demand_inventory.dashboard.app import create_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sales", type=Path, default=None, help="Optional validated sales CSV")
    parser.add_argument("--forecasts", type=Path, default=None, help="Optional forecast predictions CSV")
    parser.add_argument("--inventory", type=Path, default=None, help="Optional optimized inventory CSV")
    args = parser.parse_args()

    try:
        app = create_app(ROOT, sales_path=args.sales, forecasts_path=args.forecasts, inventory_path=args.inventory)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"DASHBOARD STARTUP ERROR: {exc}")
        return 2
    print(f"DS-02 dashboard: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the dashboard.")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
