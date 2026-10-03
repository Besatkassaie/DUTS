"""Reach-maximizing F*/delta/tau sweep for WDC, on the maxD-selected candidates
(experiments/wdc/wdc_maxd_selection.py) -- the WDC analogue of
experiments/santoslarge_tau_sweep.py, extended to a much larger grid per user
instruction: k in {10,20,30,40,50} x alpha spanning 2 to 60.

**"Feasible" here means Stage 2's tau-constraint, checked across EVERY cell of
the (k,alpha) grid for a query** -- same convention as santosLarge's cohort
selection (Option 3): a query only counts toward the target count if ALL of
its grid-cell instances are feasible at the chosen tau, not just some of them.

**Why U doesn't matter for this sweep.** Stage 2's feasibility (does some
k-subset of P satisfy F>=tau) depends only on each candidate's (N,n) -- U only
ever appears in the ILP's objective, never its constraints (verified directly
in duts/stage2_ilp.py). So every candidate here is scored with a dummy
U=1.0 -- no real unionability computation (the expensive step) is needed just
to determine feasibility, matching santoslarge_tau_sweep.py's same shortcut.

**Cells are built once per query** (one retrieval + up to 45 Stage-1 calls,
reusing the SAME retrieved D across every (k,alpha) cell) and cached to disk,
since retrieval is the expensive part and is independent of (k,alpha).
"""
import csv
import os
import pickle
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import F as F_ratio
from duts.stats import aggregate, drop_empty
from duts.types import CandidateStats
from dutsx.runner import QueryTask, retrieve_unscored_candidates

from .context import build_wdc_context

REPO = "/u6/bkassaie/DUTS"
TOP_N = 5000
D_THRESHOLD = 100
KS = (10, 20, 30, 40, 50)
ALPHAS = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0)
TAU_FLOOR = 0.05


class Cell(NamedTuple):
    q_table: str
    k: int
    alpha: float
    N_Q: int
    n_Q: int
    P_scored: List[CandidateStats]   # dummy U=1.0 -- feasibility-only


def _load_maxd_cohort(tier: str, d_threshold: int = D_THRESHOLD) -> List[Tuple[str, int, str, int]]:
    path = os.path.join(REPO, "experiments/results/wdc_{}_maxd_selection_report.csv".format(tier))
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if int(r["new_n_D"]) >= d_threshold:
            out.append((r["q_table"], int(r["new_attr"]), r["new_value"], int(r["new_n_D"])))
    return out


def build_cells(tier: str, ks=KS, alphas=ALPHAS, top_n: int = TOP_N,
                 d_threshold: int = D_THRESHOLD, verbose: bool = True) -> List[Cell]:
    cache_path = os.path.join(REPO, "experiments/results/_cache_wdc_tau_sweep_cells_{}.pkl".format(tier))
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            cells = pickle.load(f)
        if verbose:
            print("[{}] loaded {} cached cells from {}".format(tier, len(cells), cache_path))
        return cells

    ctx, build_times = build_wdc_context(tier)
    cohort = _load_maxd_cohort(tier, d_threshold)
    if verbose:
        print("[{}] context built ({} tables); {} candidates at |D|>={}".format(
            tier, build_times.n_tables, len(cohort), d_threshold), flush=True)

    cells: List[Cell] = []
    for i, (q_table, attr, value, _n_d) in enumerate(cohort):
        task0 = QueryTask(q_table=q_table, attr=attr, M={value}, F_star=0.3, delta=0.15,
                           k=KS[0], alpha=ALPHAS[0], include_query=True, top_n=top_n)
        try:
            D, N_Q, n_Q, _ = retrieve_unscored_candidates(task0, ctx)
        except Exception as e:
            if verbose:
                print("  [{}/{}] {}: retrieval FAILED ({})".format(i + 1, len(cohort), q_table, e))
            continue
        D = drop_empty(D)
        if len(D) == 0:
            continue

        for k in ks:
            for alpha in alphas:
                pool_size = int(alpha * k)
                if pool_size >= len(D):
                    P_un = D
                else:
                    s1 = stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, True)
                    P_un = s1.selected
                P_scored = [CandidateStats(table=c.table, N=c.N, n=c.n, U=1.0) for c in P_un]
                cells.append(Cell(q_table=q_table, k=k, alpha=alpha, N_Q=N_Q, n_Q=n_Q,
                                   P_scored=P_scored))
        if verbose:
            print("  [{}/{}] {}: |D|={}, built {} cells".format(
                i + 1, len(cohort), q_table, len(D), len(ks) * len(alphas)), flush=True)

    with open(cache_path, "wb") as f:
        pickle.dump(cells, f)
    if verbose:
        print("[{}] cached {} cells to {}".format(tier, len(cells), cache_path))
    return cells


def sweep(cells: List[Cell], F_star: float, delta: float) -> Dict:
    tau = F_star - delta
    by_query: Dict[str, List[bool]] = {}
    for c in cells:
        feasible = False
        if len(c.P_scored) >= c.k:
            r2 = stage2_ilp.solve(c.P_scored, c.k, tau, c.N_Q, c.n_Q, True)
            feasible = r2.feasible
        by_query.setdefault(c.q_table, []).append(feasible)

    fully_feasible = [q for q, flags in by_query.items() if all(flags)]
    return {
        "F_star": F_star, "delta": delta, "tau": tau,
        "n_queries": len(by_query),
        "n_fully_feasible": len(fully_feasible),
        "fully_feasible_tables": fully_feasible,
        "cell_feasible_rate": sum(sum(f) for f in by_query.values()) / max(sum(len(f) for f in by_query.values()), 1),
    }


if __name__ == "__main__":
    import sys
    tier = sys.argv[1] if len(sys.argv) > 1 else "tier_10k"
    t0 = time.perf_counter()
    cells = build_cells(tier)
    print("built/loaded {} cells in {:.1f}s".format(len(cells), time.perf_counter() - t0))
