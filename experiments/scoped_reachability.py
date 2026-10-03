"""Attainable-``F`` ranges scoped to the cardinality and superset each stage
actually selects from -- a correction to ``fraction_reachability.py``, which
reports ``F_min``/``F_max`` only over ``k``-subsets of ``D``.

That's the right reference for nothing the pipeline does. Stage 1 (``F_P``)
selects a pool of size ``alpha*k`` (or all of ``D`` when the C5 clamp fires),
never ``k``. Stage 2 (``F_R``) selects ``k`` from ``P``, never from ``D``
directly -- once Stage 1 has pooled, candidates outside ``P`` are unreachable
by Stage 2 regardless of how attainable they'd have made ``F``.

This module enumerates two different attainable sets per retained query:

    pool range   {F(S) : |S| = min(alpha*k, |D|), S subseteq D}   -- F_P's ceiling
    result range {F(S) : |S| = k,                 S subseteq P}   -- F_R's ceiling

``P`` here is Stage 1's *actual* selection (from ``duts.stage1_dinkelbach``,
or ``P = D`` when clamped) -- not a re-derivation, the same object the
two-stage pipeline used to produce the ``F_P``/``F_R`` already reported in
``santos3_focused_k3_k5_alpha2.csv``. This module reruns retrieval and Stage 1
only (deterministic, no unionability/Stage 2 needed for a range over
``N``/``n`` alone) and joins onto that CSV's ``F_P``/``F_R`` for comparison.

Cost: two brute-force enumerations per query. Pool range is
``C(|D|, min(alpha*k,|D|))``; on santos3 (``|D|`` <= 17ish) this is at most
tens of thousands. Result range is ``C(|P|, k)`` with ``|P| <= alpha*k`` (<=10
here) -- always small.
"""
import csv
import os
from itertools import combinations
from typing import List, NamedTuple, Optional

from duts import stage1_dinkelbach
from duts.stats import drop_empty
from dutsx.runner import retrieve_unscored_candidates

from . import context as ctx_mod
from .fraction_reachability import analyse_query

REPO = "/u6/bkassaie/DUTS"


def _f_range(cands, size: int, N_Q: int, n_Q: int, include_query: bool):
    """Exact ``{F(S) : |S| = size, S subseteq cands}`` range by brute force."""
    if size <= 0 or size > len(cands):
        return None, None
    Ns = [c.N for c in cands]
    ns = [c.n for c in cands]
    f_min = f_max = None
    for combo in combinations(range(len(cands)), size):
        sum_N = sum(Ns[i] for i in combo)
        sum_n = sum(ns[i] for i in combo)
        num, den = (sum_N + N_Q, sum_n + n_Q) if include_query else (sum_N, sum_n)
        if den == 0:
            continue
        f = num / den
        if f_min is None or f < f_min:
            f_min = f
        if f_max is None or f > f_max:
            f_max = f
    return f_min, f_max


class ScopedRow(NamedTuple):
    q_table: str
    attr: int
    k: int
    alpha: float
    n_D: int
    pool_size_requested: int
    n_pool: int              # min(pool_size_requested, n_D) -- what was actually enumerated
    clamped: bool            # C5: pool_size_requested >= n_D, P = D, no Dinkelbach
    F_min_pool: Optional[float]   # ceiling/floor for F_P: range over n_pool-subsets of D
    F_max_pool: Optional[float]
    n_P: int
    F_min_P: Optional[float]      # ceiling/floor for F_R: range over k-subsets of P
    F_max_P: Optional[float]


def run(
    benchmark: str = "santos3",
    ks=(3, 5),
    alpha: float = 2.0,
    F_star: float = 0.3,
    delta: float = 0.15,
    top_n: int = 100,
    unionability: str = "pinned_match",
    verbose: bool = True,
    cohort_csv: Optional[str] = None,
) -> List[ScopedRow]:
    """``cohort_csv``, when given, pins the query set to exactly the
    ``(q_table, attr, k)`` triples already in that CSV (the deck's canonical
    solvable-only cohort, ``santos3_focused_k3_k5_alpha2.csv``) instead of
    re-deriving it via ``analyse_query``/``tau_reachable``. This matters
    because retrieval is NOT perfectly reproducible run-to-run here --
    ``dutsx/adapters/semantic.py``'s ``hnswlib.add_items`` runs multi-threaded
    by default, and parallel HNSW insertion order is not deterministic even
    with a fixed ``random_seed``, so re-deriving "which queries are
    tau-reachable" from scratch can retain a different query set (verified:
    two back-to-back runs of this module differed by 1 query at k=5, and 2 of
    85 shared rows had a different retrieved pool composition). Pinning to
    the existing CSV's query identities keeps this consistent with every
    other santos3 slide's stated "n=76" cohort; the *value* recomputed per
    query (via a fresh retrieval call) may still shift by the same small
    margin, but the query SET does not.
    """
    paths = ctx_mod.paths_for(benchmark)
    ctx, _ = ctx_mod.build_santos_context(unionability=unionability, paths=paths)
    rows: List[ScopedRow] = []

    cohort_keys = None
    if cohort_csv is not None:
        import pandas as pd
        cohort_df = pd.read_csv(cohort_csv)
        cohort_keys = set(zip(cohort_df.q_table, cohort_df.attr.astype(int), cohort_df.k.astype(int)))

    for k in ks:
        tasks, _ = ctx_mod.load_base_tasks(
            k=k, alpha=alpha, F_star=F_star, delta=delta, top_n=top_n, paths=paths,
        )
        kept = 0
        for task in tasks:
            if cohort_keys is not None:
                if (task.q_table, int(task.attr), k) not in cohort_keys:
                    continue
            else:
                reach = analyse_query(task, ctx, benchmark)
                if reach.skipped or not reach.tau_reachable:
                    continue  # same solvable-only cohort as focused_santos3.py
            kept += 1

            D, N_Q, n_Q, _ = retrieve_unscored_candidates(task, ctx)
            D = drop_empty(D)
            pool_size = int(alpha * k)
            n_pool = min(pool_size, len(D))
            clamped = pool_size >= len(D)

            F_min_pool, F_max_pool = _f_range(D, n_pool, N_Q, n_Q, task.include_query)

            if clamped:
                P_un = D
            else:
                s1 = stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, task.include_query)
                P_un = s1.selected

            F_min_P, F_max_P = _f_range(P_un, k, N_Q, n_Q, task.include_query)

            rows.append(ScopedRow(
                q_table=task.q_table, attr=int(task.attr), k=k, alpha=alpha,
                n_D=len(D), pool_size_requested=pool_size, n_pool=n_pool,
                clamped=clamped,
                F_min_pool=F_min_pool, F_max_pool=F_max_pool,
                n_P=len(P_un), F_min_P=F_min_P, F_max_P=F_max_P,
            ))
        if verbose:
            print("k={}: {} of {} queries retained".format(k, kept, len(tasks)))
    return rows


def write_rows(rows: List[ScopedRow], name: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name + ".csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ScopedRow._fields)
        for r in rows:
            w.writerow(r)
    return path


if __name__ == "__main__":
    # Pinned to the just-regenerated (post-2026-08-14-fix) focused CSV, so the
    # query set matches exactly -- not re-derived from a second fresh
    # analyse_query pass, which would reintroduce the run-to-run cohort drift
    # documented in run()'s docstring.
    cohort_csv = os.path.join(REPO, "experiments/results/santos3_focused_k3_k5_alpha2.csv")
    rows = run(cohort_csv=cohort_csv)
    out = write_rows(rows, "santos3_scoped_reachability_k3_k5_alpha2",
                      os.path.join(REPO, "experiments/results"))
    print("wrote", out, "({} rows)".format(len(rows)))
