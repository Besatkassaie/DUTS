"""The tidy results schema (PLAN-integration.md §4 Phase E, deliverable 1).

One ``ResultRow`` per ``(query, config)`` evaluated by any experiment driver
in this package -- every driver (``alpha_sweep.py``, ``lp_precheck.py``,
``retrieval_ablation.py``) constructs rows through ``row_from_result`` (the
common path) so a reader joining CSVs across experiments sees one column
layout, not four incompatible ones. ``experiment``/``config_id`` distinguish
which sweep and which condition within it produced a given row.
"""
import time
from typing import List, NamedTuple, Optional

import pandas as pd


class ResultRow(NamedTuple):
    # --- experiment identity ---
    experiment: str        # "alpha_sweep" | "lp_precheck" | "retrieval_ablation"
    config_id: str          # human-readable condition key, e.g. "k=10_alpha=2"
    seed: int

    # --- query identity ---
    q_table: str
    attr: str
    M_size: int              # |M| -- santos queries are all |M|=1 today (PLAN-integration.md §7 open Q3)

    # --- two-stage parameters (QueryTask) ---
    k: int
    alpha: float
    F_star: float
    delta: float
    include_query: bool
    top_n: int

    # --- adapters in effect ---
    adapter_synopsis: str
    adapter_semantic: str
    adapter_overlap: str
    adapter_unionability: str
    lp_precheck: bool         # Stage 2's own lp_precheck flag (experiments/lp_precheck.py varies this)

    # --- retrieval telemetry (§7.3) ---
    n_sem: Optional[int]
    n_ovl: Optional[int]
    n_pair: Optional[int]

    # --- outcome ---
    feasible: bool
    infeasibility_cause: Optional[str]
    error: Optional[str]      # set (and feasible=False) if run_query itself raised

    n_D: int
    n_P: int
    n_R: int
    F_P: Optional[float]
    F_R: Optional[float]
    delta_R: Optional[float]
    sum_U: Optional[float]
    n_unionability_computations: int
    dinkelbach_iterations_stage1: int
    retrieval_time_s: Optional[float]
    stage1_time_s: Optional[float]
    effective_alpha: Optional[float]

    ilp_time_s: Optional[float]
    lp_precheck_ran: bool
    lp_feasible: Optional[bool]
    lp_time_s: Optional[float]
    wall_time_s: float

    # --- groundtruth precision/recall (experiments/groundtruth_eval.py only;
    #     None for every other experiment's rows, per the module docstring's
    #     "one column layout" contract -- these columns are simply absent
    #     signal, not a broken join, when concatenated with other experiments) ---
    gt_size: Optional[int] = None            # |groundtruth[q_table]|; None if q_table has no entry
    d_hits: Optional[int] = None             # |D_tables ∩ groundtruth|
    p_hits: Optional[int] = None             # |P_tables ∩ groundtruth|
    r_hits: Optional[int] = None             # |R_tables ∩ groundtruth|
    precision_d: Optional[float] = None
    recall_d: Optional[float] = None
    precision_p: Optional[float] = None
    recall_p: Optional[float] = None
    precision_r: Optional[float] = None
    recall_r: Optional[float] = None

    # --- stage1-skip ablation (experiments/stage1_skip_ablation.py only) ---
    scoring_time_s: Optional[float] = None   # wall time in unionability scoring, isolated from ILP time


FIELDS = ResultRow._fields


def row_from_result(
    experiment: str,
    config_id: str,
    seed: int,
    task,                      # dutsx.runner.QueryTask
    result,                     # dutsx.runner.RunResult, or None on error
    wall_time_s: float,
    adapters: dict,             # {"synopsis":..., "semantic":..., "overlap":..., "unionability":...}
    lp_precheck: bool = True,
    error: Optional[str] = None,
    gt_size: Optional[int] = None,
    d_hits: Optional[int] = None,
    p_hits: Optional[int] = None,
    r_hits: Optional[int] = None,
    precision_d: Optional[float] = None,
    recall_d: Optional[float] = None,
    precision_p: Optional[float] = None,
    recall_p: Optional[float] = None,
    precision_r: Optional[float] = None,
    recall_r: Optional[float] = None,
) -> ResultRow:
    """Build one tidy row from a completed (or failed) ``run_query`` call."""
    if result is None:
        tel = None
        retrieval = None
    else:
        tel = result.telemetry
        retrieval = tel.retrieval

    return ResultRow(
        experiment=experiment, config_id=config_id, seed=seed,
        q_table=task.q_table, attr=str(task.attr), M_size=len(task.M),
        k=task.k, alpha=task.alpha, F_star=task.F_star, delta=task.delta,
        include_query=task.include_query, top_n=task.top_n,
        adapter_synopsis=adapters.get("synopsis", ""),
        adapter_semantic=adapters.get("semantic", ""),
        adapter_overlap=adapters.get("overlap", ""),
        adapter_unionability=adapters.get("unionability", ""),
        lp_precheck=lp_precheck,
        n_sem=retrieval.n_sem if retrieval else None,
        n_ovl=retrieval.n_ovl if retrieval else None,
        n_pair=retrieval.n_pair if retrieval else None,
        feasible=bool(tel.feasible) if tel else False,
        infeasibility_cause=tel.infeasibility_cause if tel else (error and "error"),
        error=error,
        n_D=tel.n_D if tel else 0,
        n_P=tel.n_P if tel else 0,
        n_R=tel.n_R if tel else 0,
        F_P=tel.F_P if tel else None,
        F_R=tel.F_R if tel else None,
        delta_R=tel.delta_R if tel else None,
        sum_U=tel.sum_U if tel else None,
        n_unionability_computations=tel.n_unionability_computations if tel else 0,
        dinkelbach_iterations_stage1=tel.dinkelbach_iterations_stage1 if tel else 0,
        retrieval_time_s=tel.retrieval_time_s if tel else None,
        stage1_time_s=tel.stage1_time_s if tel else None,
        effective_alpha=tel.effective_alpha if tel else None,
        scoring_time_s=tel.scoring_time_s if tel else None,
        ilp_time_s=tel.ilp_time_s if tel else None,
        lp_precheck_ran=bool(tel.lp_precheck_ran) if tel else False,
        lp_feasible=tel.lp_feasible if tel else None,
        lp_time_s=tel.lp_time_s if tel else None,
        wall_time_s=wall_time_s,
        gt_size=gt_size, d_hits=d_hits, p_hits=p_hits, r_hits=r_hits,
        precision_d=precision_d, recall_d=recall_d,
        precision_p=precision_p, recall_p=recall_p,
        precision_r=precision_r, recall_r=recall_r,
    )


def rows_to_dataframe(rows: List[ResultRow]) -> pd.DataFrame:
    return pd.DataFrame([r._asdict() for r in rows], columns=FIELDS)


def write_csv(rows: List[ResultRow], path: str) -> None:
    rows_to_dataframe(rows).to_csv(path, index=False)


def write_parquet(rows: List[ResultRow], path: str) -> None:
    """Best-effort: only requires a parquet engine (pyarrow) to be present;
    callers that just want the CSV should not depend on this succeeding."""
    rows_to_dataframe(rows).to_parquet(path, index=False)


def timed_run_query(run_query_fn, task, ctx):
    """Run ``run_query(task, ctx)``, returning ``(result_or_None, wall_time_s,
    error_or_None)``. Never raises -- a real-data sweep over 48 queries x many
    configs must not die on one query's edge case (e.g. a missing vector);
    the failure is recorded as a row instead, per this module's contract."""
    t0 = time.perf_counter()
    try:
        result = run_query_fn(task, ctx)
        return result, time.perf_counter() - t0, None
    except Exception as exc:  # noqa: BLE001 -- intentionally broad, see docstring
        return None, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"
