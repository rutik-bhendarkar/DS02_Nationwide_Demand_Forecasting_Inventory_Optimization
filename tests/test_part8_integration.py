from __future__ import annotations

import json
from pathlib import Path

from scripts.run_pipeline import STAGES
from scripts.verify_release import verify


def test_pipeline_stages_are_ordered_parts_one_to_seven():
    names = [command[0] for _, command in STAGES]
    assert names == [
        "validate_data.py",
        "run_eda.py",
        "run_feature_engineering.py",
        "run_forecasting.py",
        "run_evaluation.py",
        "run_inventory_optimization.py",
        "build_dashboard_preview.py",
    ]


def _write_release_fixture(root: Path) -> None:
    files = [
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
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    (root / "outputs/reports/part4_model_status.json").write_text(
        json.dumps({"models": {"lightgbm": {"status": "trained"}}}), encoding="utf-8"
    )
    (root / "outputs/reports/part5_evaluation_summary.json").write_text(
        json.dumps({"selected_model": "lightgbm"}), encoding="utf-8"
    )


def test_release_verifier_passes_complete_fixture(tmp_path: Path):
    _write_release_fixture(tmp_path)
    report = verify(tmp_path)
    assert report["status"] == "PASS"
    assert report["project_root"] == "."
    assert (tmp_path / "outputs/final/release_readiness.json").is_file()


def test_release_verifier_fails_when_required_artifact_missing(tmp_path: Path):
    _write_release_fixture(tmp_path)
    (tmp_path / "outputs/inventory/optimized_purchase_orders.csv").unlink()
    report = verify(tmp_path)
    assert report["status"] == "FAIL"
    failed = [item["check"] for item in report["checks"] if not item["passed"]]
    assert any("optimized_purchase_orders.csv" in item for item in failed)
