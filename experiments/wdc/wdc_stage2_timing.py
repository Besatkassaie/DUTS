"""Stage 2 execution time (LP pre-check vs. MILP solver) on WDC, all three tiers -- the WDC
analogue of RESULTS-santoslarge.md §5c.

Grid: top_n=5000, alpha in {10,5}, k in {5,10,20} (pool alpha*k = 25..200), the finalized
6-cell grid; F*=0.20, delta=0.10 (tau=0.10); same 67-query cohort as every other WDC section.

Retrieval does not depend on k/alpha, so D is retrieved once per (tier, query) and reused across
the 6 cells. Stage 1 (Dinkelbach) picks the pool P per cell (or clamps P=D when pool>=|D|), then
``stage2_ilp.solve`` is timed via its own ``lp_time_s`` / ``milp_time_s`` info fields plus an
outer wall-clock timer around the whole call.

U is SEEDED SYNTHETIC uniform(0,1), not real unionability: tier_1m's lean context has no full
vectors dict (memory ceiling, see wdc-report.md Sec 2), and real matching is not what's being
timed. Stage 2 feasibility depends only on (N,n); U only shapes the MILP objective.
"""
import csv
import os
import time
import zlib
from typing import List, NamedTuple

import numpy as np

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import drop_empty
from duts.types import CandidateStats
from dutsx.runner import QueryTask, retrieve_unscored_candidates

from .context import build_wdc_context
from .wdc_topn_pool_sweep import _build_context_lean, _load_cohort

REPO = "/u6/bkassaie/DUTS"
TOP_N = 5000
KS = (5, 10, 20)
ALPHAS = (10.0, 5.0)
F_STAR, DELTA = 0.20, 0.10
TAU = F_STAR - DELTA
OUT_CSV = os.path.join(REPO, "experiments/results/wdc_stage2_timing.csv")


class Row(NamedTuple):
    tier: str
    q_table: str
    k: int
    alpha: float
    pool: int
    n_D: int
    clamped: bool
    feasible: bool
    lp_ran: bool
    lp_feasible: bool
    lp_time_s: float
    milp_time_s: float
    stage2_wall_s: float


def run_tier(tier: str) -> List[Row]:
    cohort = _load_cohort()
    t0 = time.perf_counter()
    if tier == "tier_1m":
        ctx = _build_context_lean(tier, needed_q_tables={c[0] for c in cohort})
    else:
        ctx, _ = build_wdc_context(tier)
    print("[{}] context ready in {:.1f}s".format(tier, time.perf_counter() - t0), flush=True)

    rows: List[Row] = []
    for qi, (q_table, attr, value) in enumerate(cohort):
        task = QueryTask(q_table=q_table, attr=attr, M={value}, F_star=F_STAR, delta=DELTA,
                         k=KS[0], alpha=ALPHAS[0], include_query=True, top_n=TOP_N)
        try:
            D, N_Q, n_Q, _ = retrieve_unscored_candidates(task, ctx)
        except Exception:
            continue
        D = drop_empty(D)
        if not D:
            continue
        rng = np.random.default_rng(zlib.crc32((tier + q_table).encode()))
        Uvals = rng.uniform(0.0, 1.0, size=len(D))
        scored = {c.table: float(u) for c, u in zip(D, Uvals)}

        for alpha in ALPHAS:
            for k in KS:
                pool_size = int(alpha * k)
                clamped = pool_size >= len(D)
                P_un = D if clamped else stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, True).selected
                P = [CandidateStats(table=c.table, N=c.N, n=c.n, U=scored[c.table]) for c in P_un]
                t1 = time.perf_counter()
                r = stage2_ilp.solve(P, k, TAU, N_Q, n_Q, True)
                wall = time.perf_counter() - t1
                info = r.info
                rows.append(Row(
                    tier=tier, q_table=q_table, k=k, alpha=alpha, pool=pool_size, n_D=len(D),
                    clamped=clamped, feasible=r.feasible,
                    lp_ran=bool(info.get("lp_precheck_ran")),
                    lp_feasible=bool(info.get("lp_feasible")),
                    lp_time_s=info.get("lp_time_s") or 0.0,
                    milp_time_s=info.get("milp_time_s") or 0.0,
                    stage2_wall_s=wall,
                ))
        if (qi + 1) % 10 == 0:
            print("  [{}] {}/{} queries".format(tier, qi + 1, len(cohort)), flush=True)
    return rows


if __name__ == "__main__":
    import sys
    tiers = sys.argv[1:] or ["tier_10k", "tier_100k", "tier_1m"]
    out = OUT_CSV
    if len(tiers) < 3:
        out = OUT_CSV.replace(".csv", "_{}.csv".format("_".join(tiers)))
    all_rows: List[Row] = []
    for tier in tiers:
        all_rows.extend(run_tier(tier))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(Row._fields)
        for r in all_rows:
            w.writerow(r)
    print("wrote", out, "({} rows)".format(len(all_rows)), flush=True)
