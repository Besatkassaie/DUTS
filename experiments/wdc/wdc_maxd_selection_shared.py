"""Shared-query maxD selection across WDC tiers -- fixes a design flaw in
``wdc_maxd_selection.py``: that script drew and optimized query tables
INDEPENDENTLY per tier, so ``tier_10k`` and ``tier_100k`` could end up with
different query tables entirely. That breaks the point of nested tiers
(``tier_10k`` is exactly a 10,001-file SUBSET of ``tier_100k``, verified by
direct file-list comparison): a valid scale comparison holds the QUERY fixed
and observes how the surrounding datalake's growth changes retrieval/
feasibility outcomes, not a different query per tier (the original WDC report
already followed this "same 30 queries" convention for its union-retrieval
experiment -- this script restores it for the maxD redesign).

**Method**:
  1. Candidate query tables are drawn from ``tier_10k``'s own table universe
     ONLY (60 candidates, same seed=42) -- guaranteed present in every larger
     nested tier (``tier_100k`` now, ``tier_1m`` later) by construction.
  2. maxD attribute/value selection (rank-by-proxy, verify-top-20-by-real-
     retrieval, keep the best) runs against the LARGER ``tier_100k`` context
     -- more informative posting-list statistics at 10x the data, and since
     ``tier_10k`` is a subset, whatever value is "well-connected" in
     ``tier_100k`` is at least as informed a choice as picking from
     ``tier_10k`` alone.
  3. The resulting (table, attr, value) triple -- ONE per candidate, fixed --
     is then evaluated for real ``|D|`` against BOTH tiers' own separate
     indices. Since ``tier_10k`` is a literal subset of ``tier_100k``,
     ``|D|`` measured on ``tier_10k`` is <= ``|D|`` measured on ``tier_100k``
     for the identical query, always (more haystack can only add matches).

Output: ``experiments/results/wdc_shared_maxd_selection_report.csv`` -- one
row per candidate, with both tiers' ``|D|`` side by side.
"""
import csv
import os
import random
import time
from typing import List, NamedTuple, Optional, Tuple

from dutsx.retrieval import retrieve_candidates
from dutsx.runner import retrieve_unscored_candidates, QueryTask
from duts.stats import drop_empty

from .context import build_wdc_context, paths_for_tier

REPO = "/u6/bkassaie/DUTS"
TOP_N = 5000
TOP_K = 20
N_CANDIDATES = 60
EMPTY_VALUE = ""
SEED = 42
SELECTION_TIER = "tier_100k"    # larger context used to CHOOSE (attr, value)
CANDIDATE_SOURCE_TIER = "tier_10k"  # smaller tier's table universe -- guaranteed nested


class SharedMaxDRow(NamedTuple):
    q_table: str
    attr: int
    value: str
    n_D_tier_10k: int
    n_D_tier_100k: int
    n_candidates_considered: int
    n_candidates_verified: int


def _table_universe(tier: str) -> List[str]:
    paths = paths_for_tier(tier)
    return sorted(f for f in os.listdir(paths.csv_dir) if f.endswith(".csv"))


def _proxy_rank(synopsis, overlap, q_table: str) -> List[Tuple[int, str, int]]:
    postings = overlap._postings  # noqa: SLF001
    candidates = []
    for attr in synopsis.categorical_attrs(q_table):
        dist = synopsis.distribution(q_table, attr)
        for value in dist:
            if value == EMPTY_VALUE:
                continue
            candidates.append((attr, value, len(postings.get(value, ()))))
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates


def _real_n_D(ctx, q_table: str, attr: int, value: str, top_n: int) -> int:
    query_vec = ctx.query_vectors[q_table][attr]
    _, telemetry = retrieve_candidates(ctx.semantic, ctx.overlap, query_vec, {value}, top_n)
    return telemetry.n_d


def _exact_n_D(ctx, q_table: str, attr: int, value: str, top_n: int) -> int:
    task = QueryTask(q_table=q_table, attr=attr, M={value}, F_star=0.3, delta=0.15,
                      k=10, alpha=2.0, include_query=True, top_n=top_n)
    D, _, _, _ = retrieve_unscored_candidates(task, ctx)
    return len(drop_empty(D))


def run(n_candidates: int = N_CANDIDATES, top_k: int = TOP_K, top_n: int = TOP_N,
        seed: int = SEED, verbose: bool = True) -> List[SharedMaxDRow]:
    t0 = time.perf_counter()
    ctx_sel, bt_sel = build_wdc_context(SELECTION_TIER)
    if verbose:
        print("[{}] selection context built in {:.1f}s ({} tables)".format(
            SELECTION_TIER, time.perf_counter() - t0, bt_sel.n_tables), flush=True)

    t0 = time.perf_counter()
    ctx_10k, bt_10k = build_wdc_context(CANDIDATE_SOURCE_TIER)
    if verbose:
        print("[{}] small-tier context built in {:.1f}s ({} tables)".format(
            CANDIDATE_SOURCE_TIER, time.perf_counter() - t0, bt_10k.n_tables), flush=True)

    # candidate tables drawn from the SMALL tier's universe (nested -> present in both)
    table_ids = _table_universe(CANDIDATE_SOURCE_TIER)
    rng = random.Random(seed)
    eligible = [t for t in table_ids if ctx_sel.synopsis.categorical_attrs(t)]
    rng.shuffle(eligible)
    candidate_tables = eligible[:n_candidates]
    if verbose:
        print("{} eligible / {} total tier_10k tables; drew {} candidates".format(
            len(eligible), len(table_ids), len(candidate_tables)), flush=True)

    rows: List[SharedMaxDRow] = []
    for i, q_table in enumerate(candidate_tables):
        # rank + verify against the LARGER tier's context
        candidates = _proxy_rank(ctx_sel.synopsis, ctx_sel.overlap, q_table)
        top = candidates[:top_k]

        best_attr, best_value, best_n_D = None, None, -1
        for attr, value, _proxy in top:
            n_D = _real_n_D(ctx_sel, q_table, attr, value, top_n)
            if n_D > best_n_D:
                best_attr, best_value, best_n_D = attr, value, n_D
        if best_attr is None:
            continue

        n_D_100k = _exact_n_D(ctx_sel, q_table, best_attr, best_value, top_n)
        n_D_10k = _exact_n_D(ctx_10k, q_table, best_attr, best_value, top_n)

        rows.append(SharedMaxDRow(
            q_table=q_table, attr=best_attr, value=best_value,
            n_D_tier_10k=n_D_10k, n_D_tier_100k=n_D_100k,
            n_candidates_considered=len(candidates), n_candidates_verified=len(top),
        ))
        if verbose:
            print("[{}/{}] {}: |D|_10k={} |D|_100k={}".format(
                i + 1, len(candidate_tables), q_table, n_D_10k, n_D_100k), flush=True)

    return rows


def write_output(rows: List[SharedMaxDRow], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "wdc_shared_maxd_selection_report.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SharedMaxDRow._fields)
        for r in rows:
            w.writerow(r)
    return path


if __name__ == "__main__":
    rows = run()
    path = write_output(rows, os.path.join(REPO, "experiments/results"))
    print("\nwrote", path, "({} rows)".format(len(rows)))
    n10 = [r.n_D_tier_10k for r in rows]
    n100 = [r.n_D_tier_100k for r in rows]
    print("tier_10k |D|: min={} median={} max={}".format(min(n10), sorted(n10)[len(n10)//2], max(n10)))
    print("tier_100k |D|: min={} median={} max={}".format(min(n100), sorted(n100)[len(n100)//2], max(n100)))
    for thr in (50, 100, 200, 500):
        print("|D|_10k >= {}: {}/{}  |  |D|_100k >= {}: {}/{}".format(
            thr, sum(1 for n in n10 if n >= thr), len(rows),
            thr, sum(1 for n in n100 if n >= thr), len(rows)))
