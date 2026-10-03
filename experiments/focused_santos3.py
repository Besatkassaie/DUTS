"""Focused santos3 study: only the queries for which ``tau`` is REACHABLE,
at ``alpha = 2`` and ``k in {3, 5}``.

Why restrict the query set. On the full 48 santos3 queries most failures are
``insufficient_candidates`` -- retrieval simply didn't supply ``k`` tables -- and
those rows carry no Stage 1, no Stage 2 and no timings, so they dilute every
mean without saying anything about the optimizer. This module first runs the
brute-force reachability check (``experiments.fraction_reachability``) and keeps
only the queries where **some** ``k``-subset of ``D`` attains ``F >= tau``, i.e.
where a feasible answer provably exists. Every number it reports is therefore
conditioned on "the instance was solvable", which is the population the
two-stage design is actually claiming things about.

Per retained query it reports, for each ``k``:

    reachability   n_D, attainable [F_min, F_max], best_gap to F*
    outcome        feasible, F_R, sum_U, |F* - F_R|
    stage timing   retrieval / stage1 / scoring / stage2(ILP) / end-to-end
    Stage 1        Dinkelbach iterations (and whether it ran at all)
    LP pre-check   did it run, was it DECISIVE (proved infeasible => ILP skipped)
    skip-Stage-1   same query with all of D handed to Stage 2: time and F reached

The ``skip_stage1`` arm duplicates ``dutsx.runner``'s composition deliberately
rather than calling ``_score_pool``, whose docstring guarantees exactly one call
site that is not ``D`` -- see ``experiments/stage1_skip_ablation.py`` for the
same reasoning.
"""
import csv
import os
import time
from typing import List, NamedTuple, Optional

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import F as F_ratio
from duts.stats import aggregate, drop_empty
from duts.types import CandidateStats
from dutsx.runner import retrieve_unscored_candidates

from . import context as ctx_mod
from .fraction_reachability import analyse_query


class FocusedRow(NamedTuple):
    q_table: str
    attr: int
    k: int
    alpha: float
    F_star: float
    delta: float
    tau: float
    # ---- reachability (brute force over every k-subset of D) ----
    n_D: int
    n_subsets: int
    F_min: Optional[float]
    F_max: Optional[float]
    best_gap: Optional[float]          # min |F(S) - F*| over all k-subsets
    tau_reachable: bool
    # ---- two-stage outcome ----
    feasible: bool
    infeasibility_cause: Optional[str]
    n_P: int
    n_R: int
    n_unionability_computations: int
    F_P: Optional[float]
    F_R: Optional[float]
    abs_gap_R: Optional[float]         # |F* - F_R| actually achieved
    sum_U: Optional[float]
    # ---- Stage 1 ----
    dinkelbach_ran: bool
    dinkelbach_iterations: int
    # ---- LP pre-check ----
    lp_ran: bool
    lp_feasible: Optional[bool]
    lp_decisive: bool                  # LP proved infeasible => ILP skipped entirely
    lp_time_s: Optional[float]
    # ---- timings, seconds ----
    retrieval_time_s: Optional[float]
    stage1_time_s: Optional[float]
    scoring_time_s: Optional[float]
    stage2_time_s: Optional[float]
    end_to_end_s: Optional[float]
    # ---- skip-Stage-1 arm ----
    skip_n_scored: Optional[int]
    skip_scoring_time_s: Optional[float]
    skip_stage2_time_s: Optional[float]
    skip_end_to_end_s: Optional[float]
    skip_feasible: Optional[bool]
    skip_F_R: Optional[float]
    skip_abs_gap_R: Optional[float]
    skip_sum_U: Optional[float]


def _score(cands, ctx, q_table, q_attr_idx) -> List[CandidateStats]:
    """U for each candidate against the query's pinned column. Duplicated from
    ``dutsx.runner._score_pool`` on purpose -- that function's docstring makes a
    grep-able "exactly one call site, and it is not D" guarantee which the
    skip-Stage-1 arm would falsify by calling it with D."""
    out = []
    for c in cands:
        u = ctx.unionability.score(q_table, c.table, (q_attr_idx, c.attr))
        out.append(CandidateStats(table=c.table, N=c.N, n=c.n, U=u))
    return out


def run_focused(
    benchmark: str = "santos3",
    ks=(3, 5),
    alpha: float = 2.0,
    F_star: float = 0.3,
    delta: float = 0.15,
    top_n: int = 100,
    unionability: str = "pinned_match",
    verbose: bool = True,
) -> List[FocusedRow]:
    paths = ctx_mod.paths_for(benchmark)
    ctx, _ = ctx_mod.build_santos_context(unionability=unionability, paths=paths)
    tau = F_star - delta
    rows: List[FocusedRow] = []

    for k in ks:
        tasks, _ = ctx_mod.load_base_tasks(
            k=k, alpha=alpha, F_star=F_star, delta=delta, top_n=top_n, paths=paths,
        )
        kept = 0
        for task in tasks:
            reach = analyse_query(task, ctx, benchmark)
            if reach.skipped or not reach.tau_reachable:
                continue          # keep ONLY provably-solvable instances
            kept += 1

            # ---------------- two-stage pipeline, timed per stage ----------------
            t_e2e = time.perf_counter()
            t0 = time.perf_counter()
            D, N_Q, n_Q, _ = retrieve_unscored_candidates(task, ctx)
            D = drop_empty(D)
            retrieval_time = time.perf_counter() - t0

            pool_size = int(alpha * k)
            t0 = time.perf_counter()
            if pool_size >= len(D):
                P_un, iters, dink_ran = D, 0, False
                F_P = F_ratio(*aggregate(P_un, N_Q, n_Q, task.include_query))
            else:
                s1 = stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, task.include_query)
                P_un, iters, dink_ran = s1.selected, s1.iterations, True
                F_P = s1.objective_value
            stage1_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            P = _score(P_un, ctx, task.q_table, int(task.attr))
            scoring_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            r2 = stage2_ilp.solve(P, k, tau, N_Q, n_Q, task.include_query)
            stage2_time = time.perf_counter() - t0
            end_to_end = time.perf_counter() - t_e2e
            info = r2.info or {}

            F_R = F_ratio(*aggregate(r2.selected, N_Q, n_Q, task.include_query)) if r2.feasible else None
            lp_ran = bool(info.get("lp_precheck_ran", False))
            lp_feas = info.get("lp_feasible")
            # DECISIVE means the LP alone settled it: infeasible, so no ILP ran.
            lp_decisive = bool(lp_ran and lp_feas is False)

            # ---------------- skip Stage 1: all of D straight to Stage 2 ----------------
            t_skip = time.perf_counter()
            t0 = time.perf_counter()
            D_scored = _score(D, ctx, task.q_table, int(task.attr))
            skip_scoring = time.perf_counter() - t0
            t0 = time.perf_counter()
            r2s = stage2_ilp.solve(D_scored, k, tau, N_Q, n_Q, task.include_query)
            skip_stage2 = time.perf_counter() - t0
            skip_e2e = retrieval_time + (time.perf_counter() - t_skip)
            skip_F_R = (F_ratio(*aggregate(r2s.selected, N_Q, n_Q, task.include_query))
                        if r2s.feasible else None)

            rows.append(FocusedRow(
                q_table=task.q_table, attr=int(task.attr), k=k, alpha=alpha,
                F_star=F_star, delta=delta, tau=tau,
                n_D=reach.n_D, n_subsets=reach.n_subsets, F_min=reach.F_min,
                F_max=reach.F_max, best_gap=reach.best_gap, tau_reachable=True,
                feasible=r2.feasible,
                infeasibility_cause=None if r2.feasible else "stage2_infeasible",
                n_P=len(P), n_R=len(r2.selected), n_unionability_computations=len(P_un),
                F_P=F_P, F_R=F_R,
                abs_gap_R=abs(F_star - F_R) if F_R is not None else None,
                sum_U=r2.objective_value if r2.feasible else None,
                dinkelbach_ran=dink_ran, dinkelbach_iterations=iters,
                lp_ran=lp_ran, lp_feasible=lp_feas, lp_decisive=lp_decisive,
                lp_time_s=info.get("lp_time_s"),
                retrieval_time_s=retrieval_time, stage1_time_s=stage1_time,
                scoring_time_s=scoring_time, stage2_time_s=stage2_time,
                end_to_end_s=end_to_end,
                skip_n_scored=len(D), skip_scoring_time_s=skip_scoring,
                skip_stage2_time_s=skip_stage2, skip_end_to_end_s=skip_e2e,
                skip_feasible=r2s.feasible, skip_F_R=skip_F_R,
                skip_abs_gap_R=abs(F_star - skip_F_R) if skip_F_R is not None else None,
                skip_sum_U=r2s.objective_value if r2s.feasible else None,
            ))
        if verbose:
            print("k={}: {} of {} queries have tau reachable -> retained".format(
                k, kept, len(tasks)))
    return rows


def write_rows(rows: List[FocusedRow], name: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FocusedRow._fields)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    return path
