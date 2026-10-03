"""Empirical F*/tau sweep for santosLarge's maxD cohort (Step 3 of the
santosLarge redesign): find the largest F* (delta fixed) whose implied
tau = F* - delta still keeps reach/feasibility high across the (k,alpha) grid,
subject to a hard floor tau >= TAU_FLOOR (never let tau go arbitrarily low).

**Why this skips real unionability scoring entirely.** Stage 2's ILP
(``duts/stage2_ilp.py::solve``) uses U only as the objective coefficient
(``c=-U``) -- U never appears in either constraint (``dist`` uses only
``y_i(tau) = N_i - tau*n_i``; ``card`` is the cardinality equality). So
whether a size-k subset of P satisfying F>=tau EXISTS is completely
independent of U. This sweep only asks "does a feasible answer exist", not
"which one is best" -- so every candidate is scored U=1.0 (constant), and
the real pinned_match unionability adapter (which needs starmie_fair's
alignment machinery) is never built at all. Feasibility verdicts are
byte-identical to what the real scorer would produce; only the choice
*among* feasible answers would differ, and that choice isn't being made here.

**Why D/P are computed once per (query, k, alpha), not once per F* candidate.**
Neither retrieval (D depends only on q_table/attr/M/top_n) nor Stage 1
(P depends only on D and pool_size=alpha*k, never on F*/tau) changes across
the F* sweep. Only Stage 2's constraint changes. So D and P-with-constant-U
are cached once per (query, k, alpha) cell, and the sweep re-solves only the
(cheap) ILP feasibility check per F* candidate -- avoiding 1x retrieval per
sweep point instead of Nx.
"""
import csv
import os
import pickle
import time
from typing import Dict, List, NamedTuple, Tuple

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import drop_empty
from duts.types import CandidateStats
from dutsx.runner import retrieve_unscored_candidates

from . import context as ctx_mod
from .santoslarge_study import build_context

REPO = "/u6/bkassaie/DUTS"
DELTA = 0.15
TAU_FLOOR = 0.05
TOP_N = 5000
KS = (10, 15)
ALPHAS = (2.0, 3.0, 4.0)
COHORT_D_THRESHOLD = 60

MAXD_PROTECTED_CSV = os.path.join(REPO, "experiments/results/protected_attributes_santosLarge_maxD.csv")
MAXD_REPORT_CSV = os.path.join(REPO, "experiments/results/santoslarge_maxd_selection_report.csv")
CELLS_CACHE = os.path.join(REPO, "experiments/results/_cache_sl_tau_sweep_cells.pkl")


class Cell(NamedTuple):
    q_table: str
    k: int
    alpha: float
    N_Q: int
    n_Q: int
    P_scored: List[CandidateStats]   # U=1.0 constant -- see module docstring


def _cohort_tables() -> List[str]:
    with open(MAXD_REPORT_CSV) as f:
        rows = list(csv.DictReader(f))
    return [r["q_table"] for r in rows if float(r["new_n_D"]) >= COHORT_D_THRESHOLD]


def build_cells(verbose: bool = True) -> List[Cell]:
    paths = ctx_mod.paths_for("santosLarge", protected_csv=MAXD_PROTECTED_CSV)
    ctx, _paths, build_times = build_context()
    if verbose:
        print("context built in {:.1f}s".format(build_times.total_index_s))

    cohort = set(_cohort_tables())
    if verbose:
        print("cohort: {} queries (|D| >= {})".format(len(cohort), COHORT_D_THRESHOLD))

    # k value is irrelevant to retrieval/task identity beyond selecting which
    # rows load_base_tasks returns for a given q_table -- use KS[0] to fetch
    # the (q_table, attr, M) triples, then vary k/alpha ourselves per cell.
    tasks, _skipped = ctx_mod.load_base_tasks(
        k=KS[0], alpha=1.0, F_star=0.3, delta=DELTA, top_n=TOP_N, paths=paths)
    tasks = [t for t in tasks if t.q_table in cohort]
    if verbose:
        print("{} of {} cohort tasks resolved from protected CSV".format(len(tasks), len(cohort)))

    cells: List[Cell] = []
    t0 = time.perf_counter()
    for i, task in enumerate(tasks):
        D, N_Q, n_Q, _ = retrieve_unscored_candidates(task, ctx)
        D = drop_empty(D)
        for k in KS:
            for alpha in ALPHAS:
                pool_size = int(alpha * k)
                if len(D) < pool_size:
                    P_un = D
                else:
                    s1 = stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, True)
                    P_un = s1.selected
                P_scored = [CandidateStats(table=c.table, N=c.N, n=c.n, U=1.0) for c in P_un]
                cells.append(Cell(q_table=task.q_table, k=k, alpha=alpha,
                                   N_Q=N_Q, n_Q=n_Q, P_scored=P_scored))
        if verbose and (i + 1) % 10 == 0:
            print("  retrieved+pooled {}/{} queries ({:.1f}s elapsed)".format(
                i + 1, len(tasks), time.perf_counter() - t0))
    return cells


def sweep_by_cell(cells: List[Cell], f_star_grid: List[float], verbose: bool = True):
    """Returns {(k,alpha): [(F_star, tau, n_feasible, n_total), ...]} plus an
    'ALL' key for the aggregate -- so a bottleneck config isn't hidden by the
    overall average. Skips any F_star whose tau would fall below TAU_FLOOR."""
    by_config: Dict[Tuple, List[Cell]] = {}
    for c in cells:
        by_config.setdefault((c.k, c.alpha), []).append(c)
    by_config["ALL"] = cells

    out: Dict = {}
    for key, group in by_config.items():
        rows = []
        for f_star in f_star_grid:
            tau = f_star - DELTA
            if tau < TAU_FLOOR - 1e-12:
                continue
            n_feasible = sum(
                1 for c in group
                if stage2_ilp.solve(c.P_scored, c.k, tau, c.N_Q, c.n_Q, True).feasible
            )
            rows.append((f_star, tau, n_feasible, len(group)))
        out[key] = rows
        if verbose:
            label = "ALL" if key == "ALL" else "k={},alpha={}".format(*key)
            print(label, "n={}:".format(len(group)),
                  " ".join("{:.0f}%".format(100 * nf / nt) for _, _, nf, nt in rows))
    return out


if __name__ == "__main__":
    if os.path.exists(CELLS_CACHE):
        with open(CELLS_CACHE, "rb") as f:
            cells = pickle.load(f)
        print("loaded {} cached cells from {}".format(len(cells), CELLS_CACHE))
    else:
        cells = build_cells()
        with open(CELLS_CACHE, "wb") as f:
            pickle.dump(cells, f)
        print("cached {} cells to {}".format(len(cells), CELLS_CACHE))
    print("\n{} (query,k,alpha) cells total\n".format(len(cells)))

    grid = [round(0.20 + 0.05 * i, 2) for i in range(17)]  # 0.20 .. 1.00
    by_config = sweep_by_cell(cells, grid)

    out_path = os.path.join(REPO, "experiments/results/santoslarge_tau_sweep_by_config.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "alpha", "F_star", "tau", "n_feasible", "n_total", "reach_rate"])
        for key, rows in by_config.items():
            k, alpha = (None, None) if key == "ALL" else key
            for f_star, tau, n_feasible, n_total in rows:
                w.writerow([k, alpha, f_star, tau, n_feasible, n_total, n_feasible / n_total])
    print("\nwrote", out_path)
