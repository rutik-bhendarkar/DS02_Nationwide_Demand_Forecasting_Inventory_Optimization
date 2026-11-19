"""Perform a local Dash/Flask HTTP smoke test without starting a persistent server."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        import dash  # noqa: F401
        from demand_inventory.dashboard.app import create_app
    except ImportError as exc:
        print("DASH SMOKE TEST BLOCKED: install requirements-dashboard.txt or requirements-integration.txt first.")
        print(f"Missing dependency: {exc}")
        return 2

    try:
        app = create_app(args.root.resolve())
        client = app.server.test_client()
        response = client.get("/")
        if response.status_code != 200:
            print(f"DASH SMOKE TEST FAILED: GET / returned HTTP {response.status_code}")
            return 1
        body = response.get_data(as_text=True)
        if "Nationwide Demand & Inventory Intelligence" not in body and "Dash" not in body:
            print("DASH SMOKE TEST FAILED: response did not contain expected Dash page content")
            return 1
    except Exception as exc:  # report app wiring/data errors with their type
        print(f"DASH SMOKE TEST FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("DASH SMOKE TEST PASSED")
    print(f"HTTP GET / -> {response.status_code}")
    print(f"Dash version: {dash.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
