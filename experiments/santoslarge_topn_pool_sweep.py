"""Dinkelbach convergence time/iterations vs. the *input* pool size |D|,
holding k=10, alpha=3 fixed (pool_size_requested=30) -- as opposed to
santoslarge_maxd_study.py's table, which fixes top_n=5000 and varies the
*output* pool size alpha*k across the (k,alpha) grid.

|D| is not itself a settable knob -- it's whatever retrieval returns after
the HNSW top_n probe + overlap-filter intersection + drop_empty. So, exactly
like the WDC report's top_n sweep (experiments/wdc/topn_alpha_k_sweep.py),
top_n is the dial and the ACTUAL resulting |D| is what gets reported (the
requested top_n is not the same as the retrieved/filtered count).

**Cohort.** The same 46 santosLarge queries used throughout
(santoslarge_maxd_study.py's cohort: |D|>=60 at top_n=5000, minus the one
tau=0.05 outlier, SpendoverC2A3500Apr17.csv), using the maxD-selected
protected attributes/values -- re-retrieved at each top_n in TOP_NS below.
Because |D| shrinks at smaller top_n, some queries clamp (pool_size_requested
= alpha*k = 30 >= |D|) at the low end; clamped rows are still included (with
dinkelbach_ran=False, iterations=0), matching the convention used everywhere
else in this repo -- the table reports what fraction clamped at each setting
rather than silently dropping those queries.
"""
import os
import statistics
import time
from typing import List, NamedTuple

from duts import stage1_dinkelbach
from duts.stats import drop_empty
from dutsx.runner import retrieve_unscored_candidates

from . import context as ctx_mod
from .santoslarge_maxd_study import MAXD_PROTECTED_CSV, _cohort_tables
from .santoslarge_study import build_context

REPO = "/u6/bkassaie/DUTS"
K = 10
ALPHA = 3.0
POOL_SIZE = int(K * ALPHA)  # 30 -- fixed throughout; NOT the swept variable
TOP_NS = (5000, 10000, 15000, 20000, 25000, 30000)


class SweepRow(NamedTuple):
    q_table: str
    top_n: int
    n_D: int                  # actual |D|, post overlap-filter + drop_empty
    clamped: bool              # pool_size (30) >= n_D -> Stage 1 is a no-op
    dinkelbach_ran: bool
    iterations: int
    stage1_time_s: float


def run(top_ns=TOP_NS, k: int = K, alpha: float = ALPHA, verbose: bool = True) -> List[SweepRow]:
    ctx, _paths, build_times = build_context()
    if verbose:
        print("index build: {:.1f}s over {} tables".format(
            build_times.total_index_s, build_times.n_datalake_tables), flush=True)

    cohort_tables = set(_cohort_tables())
    paths = ctx_mod.paths_for("santosLarge", protected_csv=MAXD_PROTECTED_CSV)

    rows: List[SweepRow] = []
    for top_n in top_ns:
        base, skipped = ctx_mod.load_base_tasks(
            k=k, alpha=alpha, F_star=0.122, delta=0.07, top_n=top_n, paths=paths)
        base = [t for t in base if t.q_table in cohort_tables]
        if verbose:
            print("top_n={}: {} cohort queries loaded ({} skipped)".format(
                top_n, len(base), len(skipped)), flush=True)

        for task in base:
            try:
                D, N_Q, n_Q, _ = retrieve_unscored_candidates(task, ctx)
            except Exception:
                continue
            D = drop_empty(D)
            n_D = len(D)
            pool_size = int(alpha * k)
            clamped = pool_size >= n_D
            if n_D == 0:
                continue

            t0 = time.perf_counter()
            if clamped:
                iters, dink_ran = 0, False
            else:
                s1 = stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, True)
                iters, dink_ran = s1.iterations, True
            stage1_t = time.perf_counter() - t0

            rows.append(SweepRow(
                q_table=task.q_table, top_n=top_n, n_D=n_D, clamped=clamped,
                dinkelbach_ran=dink_ran, iterations=iters, stage1_time_s=stage1_t,
            ))
        if verbose:
            print("  done top_n={}".format(top_n), flush=True)
    return rows


def summarize(rows: List[SweepRow]):
    by_topn = {}
    for r in rows:
        by_topn.setdefault(r.top_n, []).append(r)

    print("\n{:>8} {:>10} {:>8} {:>12} {:>10} {:>12}".format(
        "top_n", "mean|D|", "n_ran", "mean_ms", "mean_iter", "iter_range"))
    for top_n in sorted(by_topn):
        grp = by_topn[top_n]
        n_D_vals = [r.n_D for r in grp]
        ran = [r for r in grp if r.dinkelbach_ran]
        n_clamped = sum(1 for r in grp if r.clamped)
        if ran:
            times_ms = [r.stage1_time_s * 1000 for r in ran]
            iters = [r.iterations for r in ran]
            mean_ms = statistics.mean(times_ms)
            mean_it = statistics.mean(iters)
            it_range = "{}-{}".format(min(iters), max(iters))
        else:
            mean_ms, mean_it, it_range = float("nan"), float("nan"), "n/a"
        print("{:>8} {:>10.1f} {:>8} {:>12.4f} {:>10.2f} {:>12} (clamped {}/{})".format(
            top_n, statistics.mean(n_D_vals), len(ran), mean_ms, mean_it, it_range,
            n_clamped, len(grp)))


if __name__ == "__main__":
    rows = run()
    out_path = os.path.join(REPO, "experiments/results/santoslarge_topn_pool_sweep.csv")
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SweepRow._fields)
        for r in rows:
            w.writerow(r)
    print("\nwrote", out_path, "({} rows)".format(len(rows)))
    summarize(rows)
