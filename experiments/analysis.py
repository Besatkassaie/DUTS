"""Derived summaries over a tidy results table (PLAN-integration.md §4 Phase E).

Pure functions of a ``pandas.DataFrame`` in the ``schema.ResultRow`` shape --
no side effects, no I/O -- so they're usable both from ``cli.py`` (to print a
human-readable summary after a sweep) and from ``tests/test_experiments.py``
(to check aggregation correctness on a small synthetic table).
"""
from typing import List

import pandas as pd


def infeasibility_by_cause(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Counts of each ``infeasibility_cause`` (plus ``"feasible"``) per group.

    ``group_cols`` is typically ``["k", "alpha"]`` for the alpha sweep. Every
    row contributes to exactly one cell: ``"feasible"`` if ``feasible`` else
    the recorded cause (``insufficient_candidates`` / ``stage2_infeasible``
    / ``error``).
    """
    d = df.copy()
    d["_outcome"] = d.apply(
        lambda r: "feasible" if r["feasible"] else (r["infeasibility_cause"] or "error"), axis=1
    )
    pivot = d.pivot_table(
        index=group_cols, columns="_outcome", values="q_table", aggfunc="count", fill_value=0,
    )
    pivot["total"] = pivot.sum(axis=1)
    return pivot.reset_index()


def quality_vs_cost_by_config(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Mean ``sum_U``/``F_R`` (quality, feasible rows only) vs. mean
    ``n_unionability_computations``/``wall_time_s`` (cost, all rows) per
    group -- the alpha-sweep headline table."""
    feasible = df[df["feasible"]]
    quality = feasible.groupby(group_cols).agg(
        mean_sum_U=("sum_U", "mean"),
        mean_F_R=("F_R", "mean"),
        n_feasible=("q_table", "count"),
    )
    cost = df.groupby(group_cols).agg(
        mean_n_unionability_computations=("n_unionability_computations", "mean"),
        mean_wall_time_s=("wall_time_s", "mean"),
        n_total=("q_table", "count"),
    )
    return cost.join(quality, how="left").reset_index()


def check_efficiency_invariant(df: pd.DataFrame) -> pd.DataFrame:
    """PLAN-integration.md §4 Phase E, experiment 4: for EVERY row that
    reached Stage 1 (``n_D >= k``, i.e. not ``insufficient_candidates``),
    ``n_unionability_computations`` must equal ``n_P``, and ``n_P`` must equal
    ``min(floor(alpha*k), n_D)`` -- the pool-clamping rule (C5). This is a
    structural property of ``dutsx/runner.py``'s composition (already pinned
    at the unit level by ``tests/test_dutsx_runner.py``); checking it across
    the WHOLE sweep's rows (every (query, k, alpha) condition that reached
    Stage 1, not a couple of spot checks) is what this function is for.

    Returns the (hopefully empty) subset of rows that violate the invariant;
    an empty result is the expected, passing outcome.
    """
    # "insufficient_candidates" (|D| < k) short-circuits BEFORE Stage 1
    # runs -- n_P is legitimately 0 there, not a violation. It is now the
    # only such cause (the C4 certificate, which also short-circuited, was
    # removed 2026-08-10). Every other outcome (feasible,
    # "stage2_infeasible") means Stage 1 ran and produced a real pool.
    pre_stage1_causes = {"insufficient_candidates"}
    reached_stage1 = df[~df["infeasibility_cause"].isin(pre_stage1_causes)].copy()
    reached_stage1 = reached_stage1[reached_stage1["error"].isna()]
    expected_pool = reached_stage1.apply(
        lambda r: min(int(r["alpha"] * r["k"]), int(r["n_D"])), axis=1
    )
    violates = reached_stage1[
        (reached_stage1["n_unionability_computations"] != reached_stage1["n_P"])
        | (reached_stage1["n_P"] != expected_pool)
    ]
    return violates
