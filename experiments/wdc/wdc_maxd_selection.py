"""Deliberately re-select each WDC query's ``(protected_attribute, protected_value)``
pair to maximize retrieved ``|D|``, replacing WDC's random/overlap-biased
``auto_select_queries`` assignment -- the direct WDC analogue of
``experiments/santoslarge_maxd_selection.py``, and part of the same effort:
stop inflating ``|D|`` via the union-mode retrieval relaxation
(``dutsx.retrieval``'s ``D = D_sem ∪ D_ovl``, a documented stress-test-only hack --
see ``experiments/wdc/union_retrieval.py``'s module docstring) and instead get a
larger, still paper-faithful (``D = D_sem ∩ D_ovl``) ``|D|`` by choosing better
attributes/values.

**Method, identical to santosLarge's**: per query table, enumerate every
``(categorical attr, value)`` pair from its own synopsis (excluding the empty-
string bucket), rank by the free proxy ``len(InvertedIndexOverlap._postings[value])``,
verify the top ``TOP_K`` by real (intersection) retrieval, keep whichever gives
the single largest real post-intersection ``|D|``. ``|M|=1`` throughout (one
attribute, one value per query) -- no value sets.

**Wider starting pool, WDC-specific**: WDC has no curated query/datalake split
and no groundtruth query list, so query tables are drawn at random in the first
place (``query_selection.auto_select_queries``). Per user instruction, this
script draws ``N_CANDIDATES=60`` random tables (not the eventual target of 30)
so that after the |D|-threshold + F*/delta/tau sweep (a separate script) filters
down to feasible queries, there is a safety margin to reach 30 feasible queries
without lowering the tau=0.05 floor -- mirroring santosLarge's "exclude/replace
rather than relax" resolution (Option 3).

Output, under ``experiments/results/``:
  - ``wdc_<tier>_maxd_selection_report.csv`` -- one row per candidate table,
    orig (random-baseline) vs. new (maxD) |D|, for transparency.
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


class WdcMaxDRow(NamedTuple):
    q_table: str
    orig_attr: int
    orig_value: str
    orig_n_D: int
    new_attr: int
    new_value: str
    new_n_D: int
    n_candidates_considered: int
    n_candidates_verified: int


def _table_universe(tier: str) -> List[str]:
    paths = paths_for_tier(tier)
    return sorted(f for f in os.listdir(paths.csv_dir) if f.endswith(".csv"))


def _proxy_rank(synopsis, overlap, q_table: str) -> List[Tuple[int, str, int]]:
    postings = overlap._postings  # noqa: SLF001 -- offline analysis script, not production path
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


def _random_baseline(synopsis, rng: random.Random, q_table: str) -> Optional[Tuple[int, str]]:
    """Plain-random (attr, value) -- the pre-maxD baseline for this table,
    |M|=1, no overlap-biasing (the fairest 'before' comparison)."""
    cat_attrs = synopsis.categorical_attrs(q_table)
    if not cat_attrs:
        return None
    attr = rng.choice(cat_attrs)
    dist = synopsis.distribution(q_table, attr)
    domain = [v for v in dist if v != EMPTY_VALUE]
    if not domain:
        return None
    return attr, rng.choice(domain)


def run(tier: str, n_candidates: int = N_CANDIDATES, top_k: int = TOP_K,
        top_n: int = TOP_N, seed: int = SEED, verbose: bool = True) -> List[WdcMaxDRow]:
    t0 = time.perf_counter()
    ctx, build_times = build_wdc_context(tier)
    if verbose:
        print("[{}] context built in {:.1f}s ({} tables)".format(
            tier, time.perf_counter() - t0, build_times.n_tables), flush=True)

    table_ids = _table_universe(tier)
    rng = random.Random(seed)
    eligible = [t for t in table_ids if ctx.synopsis.categorical_attrs(t)]
    rng.shuffle(eligible)
    candidate_tables = eligible[:n_candidates]
    if verbose:
        print("[{}] {} eligible / {} total tables; drew {} candidates".format(
            tier, len(eligible), len(table_ids), len(candidate_tables)), flush=True)

    rows: List[WdcMaxDRow] = []
    for i, q_table in enumerate(candidate_tables):
        baseline = _random_baseline(ctx.synopsis, rng, q_table)
        if baseline is None:
            continue
        orig_attr, orig_value = baseline
        orig_n_D = _real_n_D(ctx, q_table, orig_attr, orig_value, top_n)

        candidates = _proxy_rank(ctx.synopsis, ctx.overlap, q_table)
        top = candidates[:top_k]

        best_attr, best_value, best_n_D = orig_attr, orig_value, orig_n_D
        for attr, value, _proxy in top:
            n_D = _real_n_D(ctx, q_table, attr, value, top_n)
            if n_D > best_n_D:
                best_attr, best_value, best_n_D = attr, value, n_D

        exact_n_D = _exact_n_D(ctx, q_table, best_attr, best_value, top_n)

        rows.append(WdcMaxDRow(
            q_table=q_table, orig_attr=orig_attr, orig_value=orig_value, orig_n_D=orig_n_D,
            new_attr=best_attr, new_value=best_value, new_n_D=exact_n_D,
            n_candidates_considered=len(candidates), n_candidates_verified=len(top),
        ))
        if verbose:
            print("[{}/{}] {}: |D| {} -> {}".format(
                i + 1, len(candidate_tables), q_table, orig_n_D, exact_n_D), flush=True)

    return rows


def write_output(tier: str, rows: List[WdcMaxDRow], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "wdc_{}_maxd_selection_report.csv".format(tier))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(WdcMaxDRow._fields)
        for r in rows:
            w.writerow(r)
    return path


if __name__ == "__main__":
    import sys
    tier = sys.argv[1] if len(sys.argv) > 1 else "tier_10k"
    rows = run(tier)
    path = write_output(tier, rows, os.path.join(REPO, "experiments/results"))
    print("\nwrote", path, "({} rows)".format(len(rows)))
    ns = [r.new_n_D for r in rows]
    olds = [r.orig_n_D for r in rows]
    print("old |D|: min={} median={} max={}".format(min(olds), sorted(olds)[len(olds)//2], max(olds)))
    print("new |D|: min={} median={} max={}".format(min(ns), sorted(ns)[len(ns)//2], max(ns)))
    for thr in (50, 100, 200, 500, 1000):
        print("queries with new |D| >= {}: {}/{}".format(thr, sum(1 for n in ns if n >= thr), len(ns)))
