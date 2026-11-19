"""Verify that all expected integrated DS-02 deliverables exist and are readable."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "data/sample/nationwide_retail_daily_sales_synthetic.csv",
    "outputs/eda/eda_dashboard.html",
    "outputs/features/modeling_dataset.csv",
    "outputs/features/splits/train.csv",
    "outputs/features/splits/validation.csv",
    "outputs/features/splits/test.csv",
    "outputs/reports/part4_model_status.json",
    "outputs/forecasts/test_lightgbm_predictions.csv",
    "outputs/reports/part5_evaluation_summary.json",
    "outputs/inventory/inventory_policy.csv",
    "outputs/inventory/optimized_purchase_orders.csv",
    "outputs/dashboard/part7_dashboard_preview.html",
    "scripts/run_dashboard.py",
    "setup_windows.ps1",
    "requirements-integration.txt",
]


def verify(root: Path, require_dashboard: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for relative in REQUIRED_FILES:
        target = root / relative
        record(f"file:{relative}", target.is_file() and target.stat().st_size > 0,
               f"{target.stat().st_size} bytes" if target.is_file() else "missing or empty")

    sample = root / REQUIRED_FILES[0]
    record("synthetic_data_label", "synthetic" in sample.name.lower(),
           "Bundled sample data is explicitly labelled synthetic." if "synthetic" in sample.name.lower() else "Sample file name does not identify synthetic provenance.")

    status_path = root / "outputs/reports/part4_model_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        model_status = status.get("models", {}).get("lightgbm", {}).get("status")
        record("lightgbm_trained_artifact", model_status == "trained", f"LightGBM status: {model_status}")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        record("lightgbm_trained_artifact", False, str(exc))

    eval_path = root / "outputs/reports/part5_evaluation_summary.json"
    try:
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        selected = evaluation.get("selected_model")
        record("evaluation_model_selection", bool(selected), f"Selected on validation data: {selected}")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        record("evaluation_model_selection", False, str(exc))

    dash_available = importlib.util.find_spec("dash") is not None
    record("dash_runtime_dependency", dash_available,
           "Dash import available in current environment." if dash_available else "Dash is not installed in this environment; install requirements-dashboard.txt or requirements-integration.txt before launching the web app.")
    if require_dashboard and not dash_available:
        record("dash_runtime_required", False, "Dashboard runtime was explicitly required but Dash is unavailable.")

    passed = all(item["passed"] for item in checks if require_dashboard or item["check"] != "dash_runtime_dependency")
    report = {
        "project": "DS-02 — Nationwide Demand Forecasting and Inventory Optimization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": ".",
        "status": "PASS" if passed else "FAIL",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "dashboard_runtime_available": dash_available,
        "notes": [
            "The bundled dataset is synthetic development data and is not evidence of real retailer performance.",
            "Prophet, Darts and PyTorch Forecasting remain optional and are not required for the default LightGBM pipeline.",
            "The dashboard is delivered as a Dash application; a live server check must be performed on the target Windows machine after dependency installation.",
        ],
        "checks": checks,
    }
    output = root / "outputs/final/release_readiness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-dashboard", action="store_true", help="Fail when Dash is not installed")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = verify(args.root.resolve(), require_dashboard=args.require_dashboard)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
