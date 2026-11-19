"""Show which Part 4 forecasting model dependencies are installed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demand_inventory.forecasting.registry import get_model_availability  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(get_model_availability(), indent=2))
