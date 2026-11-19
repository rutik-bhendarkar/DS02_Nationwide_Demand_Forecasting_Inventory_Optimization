"""Budget- and capacity-constrained integer replenishment optimization.

OR-Tools is the preferred solver. When it is not installed, SciPy MILP is used
as a clearly identified local fallback. Both formulations use integer pack
counts and binary activation variables to enforce MOQ and pack multiples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "store_id", "sku_id", "supplier_id", "suggested_order_qty_units",
    "minimum_order_qty_units", "order_multiple_units", "unit_cost_inr",
    "priority_score", "reorder_required",
}


def _prepare_candidates(policy: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(policy.columns))
    if missing:
        raise ValueError(f"Inventory policy is missing optimization columns: {missing}")
    data = policy.copy()
    numeric = [
        "suggested_order_qty_units", "minimum_order_qty_units",
        "order_multiple_units", "unit_cost_inr", "priority_score",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[numeric].isna().any().any() or not np.isfinite(data[numeric].to_numpy()).all():
        raise ValueError("Inventory policy contains invalid optimization values")
    if (data[["suggested_order_qty_units", "minimum_order_qty_units", "unit_cost_inr", "priority_score"]] < 0).any().any():
        raise ValueError("Order quantities, MOQ, cost and priority must be non-negative")
    if (data["order_multiple_units"] <= 0).any() or (data["unit_cost_inr"] <= 0).any():
        raise ValueError("Order multiples and unit costs must be positive")
    if data.duplicated(["store_id", "sku_id"]).any():
        raise ValueError("Inventory policy contains duplicate store/SKU rows")
    data = data.loc[data["reorder_required"].astype(bool) & (data["suggested_order_qty_units"] > 0)].copy()
    data["_max_packs"] = np.floor(data["suggested_order_qty_units"] / data["order_multiple_units"]).astype(int)
    data["_min_packs"] = np.ceil(data["minimum_order_qty_units"] / data["order_multiple_units"]).astype(int)
    data = data.loc[data["_max_packs"] >= data["_min_packs"]].reset_index(drop=True)
    return data


def _solve_scipy(candidates: pd.DataFrame, budget: float, capacity: int) -> tuple[np.ndarray, str]:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n = len(candidates)
    if n == 0:
        return np.zeros(0, dtype=int), "scipy_milp_optimal_no_candidates"
    # Decision variables: [integer pack counts (n), binary order activations (n)].
    objective = np.zeros(2 * n, dtype=float)
    objective[:n] = -candidates["priority_score"].to_numpy(float) * candidates["order_multiple_units"].to_numpy(float)
    integrality = np.ones(2 * n, dtype=int)
    lower = np.zeros(2 * n, dtype=float)
    upper = np.concatenate([candidates["_max_packs"].to_numpy(float), np.ones(n)])
    rows = 2 * n + 2
    matrix = lil_matrix((rows, 2 * n), dtype=float)
    lb = np.full(rows, -np.inf)
    ub = np.full(rows, np.inf)
    pack = candidates["order_multiple_units"].to_numpy(float)
    min_packs = candidates["_min_packs"].to_numpy(float)
    max_packs = candidates["_max_packs"].to_numpy(float)
    unit_cost = candidates["unit_cost_inr"].to_numpy(float)
    for i in range(n):
        # packs_i <= max_packs_i * active_i
        matrix[i, i] = 1.0
        matrix[i, n + i] = -max_packs[i]
        ub[i] = 0.0
        # packs_i >= min_packs_i * active_i
        matrix[n + i, i] = -1.0
        matrix[n + i, n + i] = min_packs[i]
        ub[n + i] = 0.0
    matrix[2 * n, :n] = unit_cost * pack
    ub[2 * n] = budget
    matrix[2 * n + 1, :n] = pack
    ub[2 * n + 1] = capacity
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix.tocsr(), lb, ub),
        options={"time_limit": 60},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"SciPy MILP failed: {result.message}")
    packs = np.rint(result.x[:n]).astype(int)
    return packs * candidates["order_multiple_units"].to_numpy(int), "scipy_milp_optimal"


def _solve_ortools(candidates: pd.DataFrame, budget: float, capacity: int) -> tuple[np.ndarray, str]:
    try:
        from ortools.linear_solver import pywraplp
    except ImportError as exc:
        raise ImportError("OR-Tools is not installed. Install with: pip install -r requirements-optimization.txt") from exc
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("OR-Tools CBC solver is unavailable in this installation")
    n = len(candidates)
    if n == 0:
        return np.zeros(0, dtype=int), "ortools_cbc_optimal_no_candidates"
    packs = []
    active = []
    objective = solver.Objective()
    budget_constraint = solver.Constraint(0, float(budget), "budget")
    capacity_constraint = solver.Constraint(0, float(capacity), "capacity")
    for i, row in candidates.iterrows():
        max_packs = int(row["_max_packs"])
        min_packs = int(row["_min_packs"])
        pack_size = int(row["order_multiple_units"])
        p = solver.IntVar(0, max_packs, f"packs_{i}")
        y = solver.BoolVar(f"active_{i}")
        solver.Add(p <= max_packs * y)
        solver.Add(p >= min_packs * y)
        budget_constraint.SetCoefficient(p, float(row["unit_cost_inr"]) * pack_size)
        capacity_constraint.SetCoefficient(p, pack_size)
        objective.SetCoefficient(p, float(row["priority_score"]) * pack_size)
        packs.append(p)
        active.append(y)
    objective.SetMaximization()
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"OR-Tools CBC did not prove optimality (status={status})")
    quantities = np.array([int(round(variable.solution_value())) for variable in packs], dtype=int)
    quantities *= candidates["order_multiple_units"].to_numpy(int)
    return quantities, "ortools_cbc_optimal"


def optimize_replenishment(
    policy: pd.DataFrame,
    budget_inr: float = 500_000.0,
    capacity_units: int = 5_000,
    backend: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select order quantities to maximize priority-weighted replenishment.

    Constraints: total procurement budget, total unit capacity, MOQ per active
    line, order multiples, and no order beyond each policy's suggested need.
    """
    if not np.isfinite(budget_inr) or budget_inr <= 0:
        raise ValueError("budget_inr must be greater than zero")
    if int(capacity_units) <= 0:
        raise ValueError("capacity_units must be greater than zero")
    if backend not in {"auto", "ortools", "scipy"}:
        raise ValueError("backend must be one of: auto, ortools, scipy")

    candidates = _prepare_candidates(policy)
    used_backend = backend
    if backend == "ortools":
        quantities, solver_status = _solve_ortools(candidates, budget_inr, int(capacity_units))
        used_backend = "ortools"
    elif backend == "scipy":
        quantities, solver_status = _solve_scipy(candidates, budget_inr, int(capacity_units))
        used_backend = "scipy"
    else:
        try:
            quantities, solver_status = _solve_ortools(candidates, budget_inr, int(capacity_units))
            used_backend = "ortools"
        except ImportError:
            quantities, solver_status = _solve_scipy(candidates, budget_inr, int(capacity_units))
            used_backend = "scipy_fallback_ortools_unavailable"

    output = policy.copy()
    output["optimized_order_qty_units"] = 0
    if not candidates.empty:
        key_to_qty = {
            (row.store_id, row.sku_id): int(qty)
            for row, qty in zip(candidates.itertuples(index=False), quantities, strict=True)
        }
        output["optimized_order_qty_units"] = [
            key_to_qty.get((row.store_id, row.sku_id), 0)
            for row in output.itertuples(index=False)
        ]
    output["optimized_order_cost_inr"] = output["optimized_order_qty_units"] * output["unit_cost_inr"]
    output["optimization_selected"] = output["optimized_order_qty_units"] > 0
    output["unfunded_suggested_qty_units"] = (
        output["suggested_order_qty_units"] - output["optimized_order_qty_units"]
    ).clip(lower=0)
    output["procurement_priority_rank"] = (
        output["priority_score"].rank(method="dense", ascending=False).astype(int)
    )
    output = output.sort_values(
        ["optimization_selected", "priority_score", "optimized_order_cost_inr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    spend = float(output["optimized_order_cost_inr"].sum())
    total_units = int(output["optimized_order_qty_units"].sum())
    summary: dict[str, Any] = {
        "solver_backend": used_backend,
        "solver_status": solver_status,
        "budget_inr": float(budget_inr),
        "capacity_units": int(capacity_units),
        "optimized_spend_inr": spend,
        "budget_remaining_inr": float(budget_inr - spend),
        "budget_utilization_pct": (100.0 * spend / budget_inr) if budget_inr else 0.0,
        "total_order_units": total_units,
        "order_lines_selected": int(output["optimization_selected"].sum()),
        "candidate_lines": int(len(candidates)),
        "unfunded_suggested_units": int(output["unfunded_suggested_qty_units"].sum()),
        "capacity_used_units": total_units,
        "capacity_remaining_units": int(capacity_units - total_units),
        "optimization_objective": "maximize priority_score-weighted order units subject to budget, capacity, MOQ and pack-multiple constraints",
        "solver_note": "OR-Tools CBC preferred; SciPy MILP used as a clearly labelled fallback when OR-Tools is unavailable.",
    }
    if spend > budget_inr + 1e-5 or total_units > capacity_units:
        raise AssertionError("Optimizer returned an infeasible budget or capacity allocation")
    return output, summary
