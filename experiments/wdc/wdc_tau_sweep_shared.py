"""Reach-maximizing F*/delta/tau sweep on the FINAL 30-query shared cohort
(experiments/results/wdc_shared_final_cohort.csv) -- same (table, attr, value)
evaluated on BOTH tiers, so the resulting F*/delta/tau is usable across
tier_10k -> tier_100k -> (later) tier_1m, not two independently-tuned choices.

2026-09-17 methodology fix: the grid used to be k in {10,20,30,40,50} x
alpha in {10,...,60} (max pool 3000), applied to the pre-filter candidate
list (wdc_shared_maxd_selection_report.csv) to help DECIDE cohort
membership. That grid was never achievable unclamped on tier_10k (87.1% of
its (query,cell) pairs were clamped even for the best query) -- clamped
cells don't invalidate a feasibility check (P=D is just the most generous
candidate set available), but it meant "feasible on every cell" was mostly
measuring "feasible in whatever raw D got retrieved", not "Stage 1 can
build a properly-sized pool at every requested scale". See
experiments/results/wdc_shared_selection_notes.md for the full writeup.

Fixed: cohort membership is now decided upstream, by
wdc_maxd_selection_unclamped.py's |D| >= max(alpha*k) threshold (mirroring
santosLarge's own cohort-selection principle) -- every one of the 30 queries
in wdc_shared_final_cohort.csv is guaranteed genuinely unclamped at every
cell of THIS grid, on both tiers. This script's job is now just to find the
best F*/delta/tau for that fixed, already-validated cohort -- not to help
pick cohort membership.

Grid: k in {5,10,20} x alpha in {5,10} (6 cells, pools 25/50/100/200,
matching santosLarge's 6-cell grid size). tau floor 0.05, dummy U=1.0
(feasibility depends only on (N,n), never U).
"""
import csv
import os
import pickle
import time
from typing import Dict, List, NamedTuple, Tuple

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import drop_empty
from duts.types import CandidateStats
from dutsx.runner import QueryTask, retrieve_unscored_candidates

from .context import build_wdc_context

REPO = "/u6/bkassaie/DUTS"
TOP_N = 5000
KS = (5, 10, 20)
ALPHAS = (5.0, 10.0)
TAU_FLOOR = 0.05
SHARED_CSV = os.path.join(REPO, "experiments/results/wdc_unclamped_final_cohort.csv")


class Cell(NamedTuple):
    q_table: str
    tier: str
    k: int
    alpha: float
    N_Q: int
    n_Q: int
    P_scored: List[CandidateStats]


def _load_shared_candidates() -> List[Tuple[str, int, str]]:
    with open(SHARED_CSV) as f:
        rows = list(csv.DictReader(f))
    return [(r["q_table"], int(r["attr"]), r["value"]) for r in rows]


def build_cells_for_tier(tier: str, candidates, ks=KS, alphas=ALPHAS, top_n: int = TOP_N,
                          verbose: bool = True) -> List[Cell]:
    cache_path = os.path.join(REPO, "experiments/results/_cache_wdc_shared_cells_{}.pkl".format(tier))
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            cells = pickle.load(f)
        if verbose:
            print("[{}] loaded {} cached cells".format(tier, len(cells)))
        return cells

    ctx, build_times = build_wdc_context(tier)
    if verbose:
        print("[{}] context built ({} tables)".format(tier, build_times.n_tables), flush=True)

    cells: List[Cell] = []
    for i, (q_table, attr, value) in enumerate(candidates):
        task0 = QueryTask(q_table=q_table, attr=attr, M={value}, F_star=0.3, delta=0.15,
                           k=KS[0], alpha=ALPHAS[0], include_query=True, top_n=top_n)
        try:
            D, N_Q, n_Q, _ = retrieve_unscored_candidates(task0, ctx)
        except Exception as e:
            if verbose:
                print("  [{}/{}] {}: retrieval FAILED ({})".format(i + 1, len(candidates), q_table, e))
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
                cells.append(Cell(q_table=q_table, tier=tier, k=k, alpha=alpha, N_Q=N_Q, n_Q=n_Q,
                                   P_scored=P_scored))
        if verbose:
            print("  [{}/{}] {}: |D|={}, built {} cells".format(
                i + 1, len(candidates), q_table, len(D), len(ks) * len(alphas)), flush=True)

    with open(cache_path, "wb") as f:
        pickle.dump(cells, f)
    if verbose:
        print("[{}] cached {} cells".format(tier, len(cells)))
    return cells


def fully_feasible_set(cells: List[Cell], F_star: float, delta: float) -> Dict[str, bool]:
    tau = F_star - delta
    by_query: Dict[str, List[bool]] = {}
    for c in cells:
        feasible = False
        if len(c.P_scored) >= c.k:
            r2 = stage2_ilp.solve(c.P_scored, c.k, tau, c.N_Q, c.n_Q, True)
            feasible = r2.feasible
        by_query.setdefault(c.q_table, []).append(feasible)
    return {q: all(flags) for q, flags in by_query.items()}


def both_tiers_fully_feasible(cells_10k: List[Cell], cells_100k: List[Cell],
                               F_star: float, delta: float) -> bool:
    f10 = fully_feasible_set(cells_10k, F_star, delta)
    f100 = fully_feasible_set(cells_100k, F_star, delta)
    return all(f10.values()) and all(f100.values())


def find_reach_maximizing_tau(cells_10k: List[Cell], cells_100k: List[Cell], delta: float,
                               tau_floor: float = TAU_FLOOR, hi: float = 0.95,
                               coarse_step: float = 0.01, fine_step: float = 0.001,
                               verbose: bool = True) -> float:
    """Largest tau (equivalently largest F_star = tau + delta) such that every query in
    the cohort is feasible at every grid cell, on BOTH tiers -- same "fixed delta, sweep
    F_star down from a high starting point, stop at the first tau that's fully feasible
    everywhere" procedure used for santosLarge and the original (superseded) WDC sweep.
    Feasibility is monotone non-decreasing as tau decreases, so a single descending scan
    (coarse, then refined) finds the exact breakpoint without needing to search the
    infeasible region at fine resolution.
    """
    tau = hi
    while tau > tau_floor:
        F_star = tau + delta
        if verbose:
            print("  tau={:.3f} (F*={:.3f}): checking...".format(tau, F_star), flush=True)
        if both_tiers_fully_feasible(cells_10k, cells_100k, F_star, delta):
            break
        tau = round(tau - coarse_step, 6)
    else:
        tau = tau_floor

    # refine upward from the last-known-infeasible coarse step, at fine resolution
    tau_fine = min(tau + coarse_step, hi)
    best = tau if tau >= tau_floor else tau_floor
    while tau_fine > best:
        F_star = tau_fine + delta
        if both_tiers_fully_feasible(cells_10k, cells_100k, F_star, delta):
            best = tau_fine
            break
        tau_fine = round(tau_fine - fine_step, 6)
    return max(best, tau_floor)


if __name__ == "__main__":
    candidates = _load_shared_candidates()
    print("{} shared (final-cohort) candidates".format(len(candidates)))
    t0 = time.perf_counter()
    cells_10k = build_cells_for_tier("tier_10k", candidates)
    cells_100k = build_cells_for_tier("tier_100k", candidates)
    print("built/loaded cells in {:.1f}s".format(time.perf_counter() - t0))

    DELTA = 0.10  # fixed, same choice as the original (superseded) WDC sweep, for continuity
    t0 = time.perf_counter()
    tau = find_reach_maximizing_tau(cells_10k, cells_100k, DELTA)
    F_star = round(tau + DELTA, 6)
    print("reach-maximizing: tau={:.4f}, delta={:.4f}, F*={:.4f} (found in {:.1f}s)".format(
        tau, DELTA, F_star, time.perf_counter() - t0))

    f10 = fully_feasible_set(cells_10k, F_star, DELTA)
    f100 = fully_feasible_set(cells_100k, F_star, DELTA)
    print("tier_10k feasible: {}/{}".format(sum(f10.values()), len(f10)))
    print("tier_100k feasible: {}/{}".format(sum(f100.values()), len(f100)))

    params_path = os.path.join(REPO, "experiments/results/wdc_shared_query_params.csv")
    with open(params_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "value"])
        w.writerow(["top_n", TOP_N])
        w.writerow(["ks", ";".join(str(k) for k in KS)])
        w.writerow(["alphas", ";".join(str(a) for a in ALPHAS)])
        w.writerow(["max_pool", max(int(k * a) for k in KS for a in ALPHAS)])
        w.writerow(["tau_floor", TAU_FLOOR])
        w.writerow(["delta", DELTA])
        w.writerow(["tau", round(tau, 4)])
        w.writerow(["F_star", F_star])
        w.writerow(["n_queries", len(candidates)])
        w.writerow(["tier_10k_feasible", "{}/{}".format(sum(f10.values()), len(f10))])
        w.writerow(["tier_100k_feasible", "{}/{}".format(sum(f100.values()), len(f100))])
    print("wrote params to {}".format(params_path))
