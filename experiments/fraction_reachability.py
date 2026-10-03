"""Experiment 6 -- is the target fraction ``F*`` actually *reachable*?

Every other experiment reports what the pipeline **returned**. This one reports
what was **attainable**, which is a different question and the one that decides
whether a disappointing ``F_R`` is the optimizer's fault or the data's.

For each query it retrieves ``D`` exactly as the pipeline does, then enumerates
**every** ``k``-subset of ``D`` by brute force and computes

    F(S) = (N_Q + sum_{i in S} N_i) / (n_Q + sum_{i in S} n_i)      [include_query]
    F(S) =        sum_{i in S} N_i  /        sum_{i in S} n_i       [otherwise]

giving the exact attainable set ``{F(S) : |S| = k, S subseteq D}``. From that it
reports the attainable range, whether ``F*`` itself lies inside it, and the
single closest attainable value to ``F*``.

Three distinctions this draws that no other experiment can:

    tau_reachable   F_max >= tau           -- a feasible answer exists at all
    fstar_bracketed F_min <= F* <= F_max   -- F* is inside the attainable range,
                                              so *some* k-subset sits at or on
                                              both sides of the target
    best_gap        min |F(S) - F*|        -- how close ANY k-subset can get,
                                              which lower-bounds the error of
                                              every possible selection rule

``best_gap`` is the load-bearing one. If ``best_gap`` is large the target is
simply not attainable on this data and no objective function can fix it; if
``best_gap`` is small but the pipeline's ``|F* - F_R|`` is large, that gap is
attributable to C3's unbounded ``argmax F`` and IS fixable by changing the
objective. Reporting the two side by side is the point of this module.

**This is offline analysis, not a pipeline stage.** It deliberately does not
call ``duts.stage1_dinkelbach`` at all -- brute-force enumeration only, so the
numbers are independent of the solver they are used to judge, and nothing here
reintroduces the removed C4 certificate.

Cost: ``C(|D|, k)`` subsets per query. Retrieved ``|D|`` is <= 17 on santos3, so
the worst case is ``C(17, 8) = 24310`` -- microseconds. ``max_subsets`` guards
anything pathological by recording the query as skipped rather than hanging.
"""
import csv
import math
import os
from itertools import combinations
from typing import Dict, List, NamedTuple, Optional

from dutsx.runner import retrieve_unscored_candidates

from . import context as ctx_mod


class ReachabilityRow(NamedTuple):
    """One query's attainable-``F`` summary. Written as its own CSV rather than
    a ``ResultRow`` -- the shape is genuinely different (it describes a set of
    hypothetical selections, not one run)."""
    benchmark: str
    q_table: str
    attr: int
    k: int
    F_star: float
    delta: float
    tau: float
    n_D: int
    n_subsets: int
    N_Q: int
    n_Q: int
    F_Q: Optional[float]        # the query's own proportion -- the anchor include_query pins F to
    F_min: Optional[float]      # min over all k-subsets
    F_max: Optional[float]      # max over all k-subsets
    F_closest: Optional[float]  # attainable value nearest F*
    best_gap: Optional[float]   # |F_closest - F*|  -- the floor on any rule's error
    tau_reachable: Optional[bool]
    fstar_bracketed: Optional[bool]
    fstar_within_delta: Optional[bool]   # best_gap <= delta
    skipped: str                # "" when analysed; otherwise why not


def _f(sum_N: int, sum_n: int, N_Q: int, n_Q: int, include_query: bool) -> Optional[float]:
    num, den = (sum_N + N_Q, sum_n + n_Q) if include_query else (sum_N, sum_n)
    return None if den == 0 else num / den


def analyse_query(
    task, ctx, benchmark: str, max_subsets: int = 2_000_000,
) -> ReachabilityRow:
    """Enumerate every ``k``-subset of this query's retrieved ``D``."""
    D, N_Q, n_Q, _ = retrieve_unscored_candidates(task, ctx)
    k, tau = task.k, task.F_star - task.delta
    F_Q = (N_Q / n_Q) if n_Q else None
    base = dict(
        benchmark=benchmark, q_table=task.q_table, attr=int(task.attr), k=k,
        F_star=task.F_star, delta=task.delta, tau=tau, n_D=len(D),
        N_Q=N_Q, n_Q=n_Q, F_Q=F_Q,
    )
    blank = dict(
        n_subsets=0, F_min=None, F_max=None, F_closest=None, best_gap=None,
        tau_reachable=None, fstar_bracketed=None, fstar_within_delta=None,
    )
    if len(D) < k:
        return ReachabilityRow(skipped="insufficient_candidates", **base, **blank)

    n_subsets = math.comb(len(D), k) if hasattr(math, "comb") else None
    if n_subsets is not None and n_subsets > max_subsets:
        return ReachabilityRow(skipped="too_many_subsets", **base, **blank)

    Ns = [c.N for c in D]
    ns = [c.n for c in D]
    idx = range(len(D))
    f_min = f_max = f_closest = None
    best_gap = None
    count = 0
    for combo in combinations(idx, k):
        sum_N = sum(Ns[i] for i in combo)
        sum_n = sum(ns[i] for i in combo)
        f = _f(sum_N, sum_n, N_Q, n_Q, task.include_query)
        if f is None:
            continue
        count += 1
        if f_min is None or f < f_min:
            f_min = f
        if f_max is None or f > f_max:
            f_max = f
        gap = abs(f - task.F_star)
        if best_gap is None or gap < best_gap:
            best_gap, f_closest = gap, f

    if count == 0:
        return ReachabilityRow(skipped="no_valid_subset", **base, **blank)

    return ReachabilityRow(
        n_subsets=count, F_min=f_min, F_max=f_max, F_closest=f_closest, best_gap=best_gap,
        tau_reachable=f_max >= tau,
        fstar_bracketed=(f_min <= task.F_star <= f_max),
        fstar_within_delta=best_gap <= task.delta,
        skipped="", **base,
    )


def run_fraction_reachability(
    benchmark: str = "santos3",
    k: int = 5,
    alpha: float = 2.0,
    F_star: float = 0.3,
    delta: float = 0.15,
    include_query: bool = True,
    top_n: int = 100,
    unionability: str = "pinned_match",
    task_limit: Optional[int] = None,
    verbose: bool = True,
) -> List[ReachabilityRow]:
    paths = ctx_mod.paths_for(benchmark)
    ctx, _ = ctx_mod.build_santos_context(unionability=unionability, paths=paths)
    tasks, skipped = ctx_mod.load_base_tasks(
        k=k, alpha=alpha, F_star=F_star, delta=delta,
        include_query=include_query, top_n=top_n, paths=paths,
    )
    if task_limit is not None:
        tasks = tasks[:task_limit]
    if verbose:
        print("{} usable queries ({} skipped)".format(len(tasks), len(skipped)))
    return [analyse_query(t, ctx, benchmark) for t in tasks]


def write_rows(rows: List[ReachabilityRow], name: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "{}.csv".format(name))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ReachabilityRow._fields)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    return path


def summarise(rows: List[ReachabilityRow]) -> Dict[str, object]:
    ok = [r for r in rows if not r.skipped]
    n = len(ok)
    if not n:
        return {"analysed": 0}
    def frac(pred):
        return sum(1 for r in ok if pred(r)) / n
    gaps = sorted(r.best_gap for r in ok)
    return {
        "queries_total": len(rows),
        "analysed": n,
        "skipped_insufficient": sum(1 for r in rows if r.skipped == "insufficient_candidates"),
        "tau_reachable": frac(lambda r: r.tau_reachable),
        "fstar_bracketed": frac(lambda r: r.fstar_bracketed),
        "fstar_within_delta": frac(lambda r: r.fstar_within_delta),
        "median_best_gap": gaps[n // 2],
        "mean_best_gap": sum(gaps) / n,
        "median_F_min": sorted(r.F_min for r in ok)[n // 2],
        "median_F_max": sorted(r.F_max for r in ok)[n // 2],
        "median_F_Q": sorted(r.F_Q for r in ok if r.F_Q is not None)[
            len([r for r in ok if r.F_Q is not None]) // 2],
    }
