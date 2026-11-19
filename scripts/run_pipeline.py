"""Run the integrated DS-02 pipeline in dependency order (Parts 1–7)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

STAGES: list[tuple[str, list[str]]] = [
    ("Part 1 — validate source data", ["validate_data.py"]),
    ("Part 2 — exploratory data analysis", ["run_eda.py"]),
    ("Part 3 — feature engineering", ["run_feature_engineering.py"]),
    ("Part 4 — LightGBM forecasting", ["run_forecasting.py", "--models", "lightgbm"]),
    ("Part 5 — model evaluation and rolling-origin backtesting", ["run_evaluation.py", "--window-days", "7", "--n-jobs", "4"]),
    ("Part 6 — inventory optimization", ["run_inventory_optimization.py", "--solver", "auto"]),
    ("Part 7 — dashboard preview build", ["build_dashboard_preview.py"]),
]


def run_stage(label: str, command: list[str]) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    completed = subprocess.run([sys.executable, str(SCRIPTS / command[0]), *command[1:]], cwd=ROOT)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", choices=["all", *(f"part{i}" for i in range(1, 8))], default="all")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip the rolling-origin refit step in Part 5")
    parser.add_argument("--skip-tests", action="store_true", help="Do not run pytest at the end")
    parser.add_argument("--skip-release-check", action="store_true", help="Do not generate release readiness report")
    args = parser.parse_args()

    start_index = 0 if args.from_stage == "all" else int(args.from_stage.replace("part", "")) - 1
    stages = STAGES[start_index:]
    if args.skip_backtest:
        stages = [
            (name, [*cmd, "--skip-backtest"] if cmd[0] == "run_evaluation.py" else cmd)
            for name, cmd in stages
        ]

    started = time.time()
    try:
        for label, command in stages:
            run_stage(label, command)
        if not args.skip_tests:
            print(f"\n{'=' * 78}\nIntegrated automated tests\n{'=' * 78}", flush=True)
            completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
            if completed.returncode != 0:
                return completed.returncode
        if not args.skip_release_check:
            print(f"\n{'=' * 78}\nRelease readiness verification\n{'=' * 78}", flush=True)
            completed = subprocess.run([sys.executable, str(SCRIPTS / "verify_release.py")], cwd=ROOT)
            if completed.returncode != 0:
                return completed.returncode
    except subprocess.CalledProcessError as exc:
        print(f"PIPELINE STOPPED: {exc.cmd} exited with code {exc.returncode}", file=sys.stderr)
        return int(exc.returncode or 1)

    print(f"\nDS-02 pipeline completed in {(time.time() - started) / 60:.1f} minutes.")
    print("To launch the interactive dashboard, run: .\\run_dashboard.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
