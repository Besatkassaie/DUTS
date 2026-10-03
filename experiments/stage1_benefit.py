"""Benefit of Stage 1: post-retrieval pipeline time, two_stage vs skip_stage1.

Fixed setting (user-specified): top_n=30000, k=10, alpha=3 (pool=30).
F*/delta per benchmark: santosLarge 0.122/0.07, WDC tiers 0.20/0.10.

TIMED WINDOW: starts immediately before Stage 1, ends when the pipeline returns R.
Retrieval, the overlap intersection and the N_i/n_i synopsis lookup are all OUTSIDE
the window -- each query is retrieved ONCE and the identical D is handed to both
conditions, so they differ in nothing but whether Stage 1 runs.

    two_stage    Stage 1 (Dinkelbach) -> score U on P (=30) -> Stage 2 over P
    skip_stage1  score U on ALL of D  -> Stage 2 over D, cardinality k

Speedup reported downstream as (T_skip - T_two)/T_skip (time saved), baseline=skip.
sum_U / F_R / feasibility are recorded for BOTH conditions: skip_stage1 optimizes
over a superset of two_stage's candidates under the identical constraint, so its
sum_U can only be >= two_stage's, and that trade-off should be inspectable.
"""
import csv
import os
import sys
import time
from typing import List, NamedTuple

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import F as F_ratio
from duts.stats import aggregate, drop_empty
from dutsx.runner import QueryTask, _score_pool, retrieve_unscored_candidates

from .stage1_skip_ablation import _score_all

REPO = "/u6/bkassaie/DUTS"
K, ALPHA, TOP_N = 10, 3.0, 30000
POOL = int(K * ALPHA)
PARAMS = {"santosLarge": (0.122, 0.07), "tier_10k": (0.20, 0.10),
          "tier_100k": (0.20, 0.10), "tier_1m": (0.20, 0.10)}


class Row(NamedTuple):
    dataset: str
    q_table: str
    n_D: int
    two_n_scored: int
    two_stage1_s: float
    two_scoring_s: float
    two_lp_s: float
    two_milp_s: float
    two_stage2_s: float
    two_total_s: float
    two_feasible: bool
    two_sum_U: float
    two_F_R: float
    skip_n_scored: int
    skip_scoring_s: float
    skip_lp_s: float
    skip_milp_s: float
    skip_stage2_s: float
    skip_total_s: float
    skip_feasible: bool
    skip_sum_U: float
    skip_F_R: float


def _tasks_and_ctx(dataset):
    """Returns (ctx, [QueryTask]) for one dataset, at the fixed setting."""
    F_star, delta = PARAMS[dataset]
    if dataset == "santosLarge":
        from . import context as ctx_mod
        from .santoslarge_maxd_study import MAXD_PROTECTED_CSV, _cohort_tables
        from .santoslarge_study import build_context
        ctx, _paths, _bt = build_context()
        paths = ctx_mod.paths_for("santosLarge", protected_csv=MAXD_PROTECTED_CSV)
        base, _skipped = ctx_mod.load_base_tasks(
            k=K, alpha=ALPHA, F_star=F_star, delta=delta, top_n=TOP_N, paths=paths)
        cohort = set(_cohort_tables())
        return ctx, [t for t in base if t.q_table in cohort]

    from .wdc.context import build_wdc_context
    from .wdc.wdc_end2end_timing import build_tier_1m_ctx
    from .wdc.wdc_topn_pool_sweep import _load_cohort
    ctx = build_tier_1m_ctx() if dataset == "tier_1m" else build_wdc_context(dataset)[0]
    tasks = [QueryTask(q_table=q, attr=a, M={v}, F_star=F_star, delta=delta, k=K,
                       alpha=ALPHA, include_query=True, top_n=TOP_N)
             for (q, a, v) in _load_cohort()]
    return ctx, tasks


def _stage2(P, k, tau, N_Q, n_Q):
    t = time.perf_counter()
    r = stage2_ilp.solve(P, k, tau, N_Q, n_Q, True)
    el = time.perf_counter() - t
    info = r.info or {}
    F_R = F_ratio(*aggregate(r.selected, N_Q, n_Q, True)) if r.feasible else 0.0
    return r, el, info.get("lp_time_s") or 0.0, info.get("milp_time_s") or 0.0, F_R


def run(dataset: str) -> List[Row]:
    F_star, delta = PARAMS[dataset]
    tau = F_star - delta
    ctx, tasks = _tasks_and_ctx(dataset)
    print("[{}] ctx ready, {} cohort queries".format(dataset, len(tasks)), flush=True)

    rows, excluded = [], 0
    for i, task in enumerate(tasks):
        try:                                     # retrieval: OUTSIDE the timed window
            D, N_Q, n_Q, _ = retrieve_unscored_candidates(task, ctx)
        except Exception as e:
            print("  skip {}: {}".format(task.q_table, e), flush=True)
            continue
        D = drop_empty(D)
        if len(D) < K:                           # |D|<k -> empty in BOTH, no work done
            excluded += 1
            continue
        qa = task.attr if isinstance(task.attr, int) else int(task.attr)

        t0 = time.perf_counter()                 # --- two_stage ---
        P_un = stage1_dinkelbach.solve(D, POOL, N_Q, n_Q, True).selected
        s1 = time.perf_counter() - t0
        t = time.perf_counter()
        P = _score_pool(P_un, ctx, task.q_table, qa)
        sc2 = time.perf_counter() - t
        r2, st2, lp2, mi2, fr2 = _stage2(P, K, tau, N_Q, n_Q)
        two_total = s1 + sc2 + st2

        t = time.perf_counter()                  # --- skip_stage1 ---
        Pall = _score_all(D, ctx, task.q_table, qa)
        scS = time.perf_counter() - t
        rS, stS, lpS, miS, frS = _stage2(Pall, K, tau, N_Q, n_Q)
        skip_total = scS + stS

        rows.append(Row(dataset, task.q_table, len(D), len(P), s1, sc2, lp2, mi2, st2,
                        two_total, r2.feasible, r2.objective_value if r2.feasible else 0.0, fr2,
                        len(Pall), scS, lpS, miS, stS, skip_total, rS.feasible,
                        rS.objective_value if rS.feasible else 0.0, frS))
        if (i + 1) % 10 == 0:
            print("  [{}] {}/{}".format(dataset, i + 1, len(tasks)), flush=True)
    print("[{}] {} rows, {} excluded (|D|<k)".format(dataset, len(rows), excluded), flush=True)
    return rows


if __name__ == "__main__":
    ds = sys.argv[1]
    rows = run(ds)
    out = os.path.join(REPO, "experiments/results/stage1_benefit_{}.csv".format(ds))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(Row._fields)
        for r in rows:
            w.writerow(r)
    print("wrote", out, len(rows), flush=True)
