"""Dinkelbach convergence time/iterations vs. the *input* pool size |D|, for WDC -- the
direct analogue of santoslarge_topn_pool_sweep.py, run per tier since |D| behaves very
differently by corpus size at a fixed large output pool.

**Setting: k=20, alpha=10 fixed** (output pool alpha*k = 200) -- the largest cell of the
redesigned WDC grid (k in {5,10,20} x alpha in {5,10}, see experiments/wdc-report.md Sec 0).
``top_n`` -- the HNSW retrieval-count knob -- is swept over {5000,10000,15000,20000,25000,30000}
to produce different *actual* |D|, same convention as every other top_n sweep in this project:
top_n is the dial, the realized |D| is what's reported.

**Same 67-query unclamped-by-construction cohort throughout**
(experiments/results/wdc_unclamped_final_cohort.csv -- the SAME (table,attr,value) triples used
for the feasibility report, every one guaranteed to clear min(D_10k,D_100k) >= 200 = max(alpha*k)
for this grid), re-retrieved per (tier, top_n). Context is built ONCE per tier and reused across
every top_n/query -- retrieval is the expensive part, Dinkelbach itself is cheap.

Unlike the old k=50/alpha=50/pool=2500 sweep (dropped: incompatible with the finalized grid),
pool_size=200 here is the grid's own ceiling, so most (tier, top_n, query) cells should run
Dinkelbach for real rather than clamp -- clamping should mainly show up, if at all, at the
smallest top_n on tier_10k.
"""
import gc
import os
import pickle
import statistics
import time
from typing import List, NamedTuple, Tuple

from duts import stage1_dinkelbach
from duts.stats import drop_empty
from dutsx import registry
from dutsx.adapters.unionability import load_starmie_vectors
from dutsx.runner import RunnerContext, retrieve_unscored_candidates

from .context import (
    WDC_ROOT, build_wdc_context, paths_for_tier, tier_available, THETA_CAT, SIGMA,
)

REPO = "/u6/bkassaie/DUTS"
K = 10
ALPHA = 3.0
POOL_SIZE = int(K * ALPHA)  # 30 -- below the cohort's own min(D_10k,D_100k)>=200 guarantee,
                            # so this cell is unclamped on tier_10k/tier_100k by construction too
TOP_NS = (5000, 10000, 15000, 20000, 25000, 30000)
COHORT_CSV = os.path.join(REPO, "experiments/results/wdc_unclamped_final_cohort.csv")
OUT_CSV_NAME = "wdc_topn_pool_sweep_k{}_a{}.csv".format(K, int(ALPHA))


class SweepRow(NamedTuple):
    tier: str
    q_table: str
    top_n: int
    n_D: int
    clamped: bool
    dinkelbach_ran: bool
    iterations: int
    stage1_time_s: float


def _load_cohort():
    import csv
    with open(COHORT_CSV) as f:
        rows = list(csv.DictReader(f))
    return [(r["q_table"], int(r["attr"]), r["value"]) for r in rows]


def _build_semantic_index_streaming(vectors: dict, synopsis, tables, sigma: float,
                                     hnsw_m: int = 32, batch_rows: int = 100_000):
    """Builds an ``HnswRetriever`` without ever holding a full-size copy of the indexed
    data ALONGSIDE hnswlib's own internal storage. Three previous attempts still SIGKILLed
    tier_1m at this account's hard 32GB-per-process RSS ulimit, each closer than the last but
    still over: (1) ``experiments.context.categorical_only_vectors``-style zero-and-keep-
    full-shape, plus ``HnswRetriever.__init__``'s own ``_flatten()``+``vstack()`` double
    -- ~48GB; (2) a pop-as-you-go filtered dict -- ~32.3GB; (3) one preallocated ``data``
    buffer handed to ``add_items`` in a single call -- ~31.9-32GB+, right at hnswlib's own
    ``init_index``/``add_items`` allocation (reducing ``M`` 32->16 barely moved this peak,
    confirming the raw per-vector data size dominates over graph-link overhead). The actual
    problem: ``init_index(max_elements=...)`` allocates hnswlib's FULL internal storage
    (~14-16GB for tier_1m) in one block, and if the caller's own fully-built source array is
    still alive when ``add_items`` runs, both full-size blocks coexist. Fix: call
    ``init_index`` once (fixed cost, unavoidable), then feed it in small batches
    (``batch_rows``, ~100-300MB each), discarding each batch's temporary array immediately
    after ``add_items`` consumes it -- hnswlib's own storage is the only large block alive
    for the whole phase, not doubled by a second full-size source array."""
    import numpy as np
    import hnswlib
    from dutsx.adapters.semantic import HnswRetriever

    # Row counts from synopsis alone (no vectors touched yet).
    cat_attrs = {table: synopsis.categorical_attrs(table) for table in tables}
    upper_bound = sum(len(idxs) for idxs in cat_attrs.values())

    dim = 0
    for table in tables:
        if not cat_attrs[table]:
            continue
        arr = vectors.get(table)
        if arr is not None:
            dim = arr.shape[1]
            break

    retriever = HnswRetriever.__new__(HnswRetriever)
    retriever.sigma = sigma
    retriever._ef = 100
    retriever.index = hnswlib.Index(space="cosine", dim=max(dim, 1))
    labels: List[Tuple[str, int]] = []
    if upper_bound > 0 and dim > 0:
        retriever.index.init_index(
            max_elements=upper_bound, ef_construction=200, M=hnsw_m, random_seed=42,
        )

    batch_vecs: List["np.ndarray"] = []
    batch_labels: List[Tuple[str, int]] = []
    next_id = 0
    for table in tables:
        arr = vectors.pop(table, None)
        if arr is None:
            continue
        idxs = cat_attrs[table]
        if not idxs:
            continue
        for j in idxs:
            v = arr[j]
            if np.linalg.norm(v) == 0.0:
                continue
            batch_vecs.append(v)
            batch_labels.append((table, j))
        if len(batch_vecs) >= batch_rows:
            batch_data = np.vstack(batch_vecs).astype(np.float32)
            ids = np.arange(next_id, next_id + len(batch_vecs))
            retriever.index.add_items(batch_data, ids)
            next_id += len(batch_vecs)
            labels.extend(batch_labels)
            batch_vecs = []
            batch_labels = []
            del batch_data
    if batch_vecs:
        batch_data = np.vstack(batch_vecs).astype(np.float32)
        ids = np.arange(next_id, next_id + len(batch_vecs))
        retriever.index.add_items(batch_data, ids)
        next_id += len(batch_vecs)
        labels.extend(batch_labels)
        del batch_data

    n_indexed = next_id
    retriever._labels = labels
    retriever.n_indexed = n_indexed
    retriever._dim = dim
    if n_indexed > 0:
        retriever.index.set_ef(retriever._ef)
    gc.collect()
    return retriever


def _cache_paths(tier: str, hnsw_m: int):
    cache_dir = os.path.join(WDC_ROOT, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return (
        os.path.join(cache_dir, "{}_hnsw_lean_m{}.bin".format(tier, hnsw_m)),
        os.path.join(cache_dir, "{}_hnsw_lean_m{}.meta.pkl".format(tier, hnsw_m)),
        os.path.join(cache_dir, "{}_query_vectors_cohort.pkl".format(tier)),
    )


def _load_cached_semantic_index(index_path: str, meta_path: str, sigma: float):
    """Loads a previously ``save_index``d ``HnswRetriever`` -- seconds, vs. the ~20+ minute,
    memory-marginal build every fresh process otherwise has to redo. ``hnswlib.Index.load_index``
    still needs the full ``max_elements``/``dim`` up front (same as a fresh ``init_index``), so
    this is fast wall-clock but not magically free of the underlying index's resident size --
    just free of the SOURCE-DATA-COEXISTING-WITH-DESTINATION peak that made building it once so
    hard (see ``_build_semantic_index_streaming``)."""
    import hnswlib
    from dutsx.adapters.semantic import HnswRetriever
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    retriever = HnswRetriever.__new__(HnswRetriever)
    retriever.sigma = sigma
    retriever._ef = 100
    retriever._labels = meta["labels"]
    retriever.n_indexed = meta["n_indexed"]
    retriever._dim = meta["dim"]
    retriever.index = hnswlib.Index(space="cosine", dim=max(meta["dim"], 1))
    if meta["n_indexed"] > 0:
        retriever.index.load_index(index_path, max_elements=meta["n_indexed"])
        retriever.index.set_ef(retriever._ef)
    return retriever


def _build_context_lean(tier: str, needed_q_tables, theta_cat: int = THETA_CAT,
                         sigma: float = SIGMA, hnsw_m: int = 16,
                         verbose: bool = True) -> RunnerContext:
    """Memory-frugal ``RunnerContext`` builder for tier_1m. Even with
    ``_build_semantic_index_streaming``'s batched ``add_items`` (which fixed the earlier
    ~48GB/~32.3GB/~32GB triple- and double-copy crashes), a full ``M=32`` run still died at
    ~33.1GB RSS, only ~450MB short of this account's hard 33,554,432 KB per-process RSS
    ulimit -- batching removed the source-side duplication, but hnswlib's own per-element
    graph-link storage (``~2*M`` links/element at the base layer) is still large enough at
    tier_1m's ~5-6M indexed columns to tip it over. ``hnsw_m=16`` (half of the 32 used for
    tier_10k/tier_100k and every other WDC/santos context in this project) is the necessary
    tier_1m-only exception -- flag this honestly in any report: it's a memory-driven
    implementation constraint, not a scalability finding, and can shift tier_1m's realized
    ``|D|`` somewhat vs. what ``M=32`` would have given (a construction/recall difference),
    though it does not change how Dinkelbach itself behaves on whatever ``|D|`` it receives.
    Also pulls out only the cohort's own query vectors before ``vectors`` is consumed
    (``retrieve_unscored_candidates`` only ever does ``ctx.query_vectors.get(q_table)``, never
    iterates the whole dict) and uses a ``"constant"`` unionability object, since this sweep
    never scores U. Caches the built HNSW index + a small query-vectors sidecar to disk
    (``_cache_paths``) so a second run of this sweep -- or a rerun after a crash elsewhere in
    the pipeline -- never has to repeat the ~20+ minute, memory-marginal build."""
    paths = paths_for_tier(tier)
    if not tier_available(paths):
        raise FileNotFoundError("tier %r not available" % tier)

    index_path, meta_path, qv_path = _cache_paths(tier, hnsw_m)
    have_index_cache = os.path.isfile(index_path) and os.path.isfile(meta_path)

    query_vectors_small = None
    if os.path.isfile(qv_path):
        with open(qv_path, "rb") as f:
            cached_qv = pickle.load(f)
        if needed_q_tables <= set(cached_qv.keys()):
            query_vectors_small = {q: cached_qv[q] for q in needed_q_tables}

    if have_index_cache and query_vectors_small is not None:
        if verbose:
            print("[{}] loading cached lean semantic index from {}".format(
                tier, index_path), flush=True)
        semantic_obj = _load_cached_semantic_index(index_path, meta_path, sigma)
        synopsis = registry.build(
            "synopsis", "metadata_store", pkl_path=paths.metadata_pkl, theta_cat=theta_cat,
        )
        tables = sorted(f for f in os.listdir(paths.csv_dir) if f.endswith(".csv"))
        if verbose:
            print("[{}] lean semantic index loaded from cache ({} items indexed)".format(
                tier, semantic_obj.n_indexed), flush=True)
    else:
        tables = sorted(f for f in os.listdir(paths.csv_dir) if f.endswith(".csv"))
        vectors = load_starmie_vectors(paths.vectors_pkl)

        synopsis = registry.build(
            "synopsis", "metadata_store", pkl_path=paths.metadata_pkl, theta_cat=theta_cat,
        )

        query_vectors_small = {q: vectors[q].copy() for q in needed_q_tables if q in vectors}
        with open(qv_path, "wb") as f:
            pickle.dump(query_vectors_small, f)

        semantic_obj = _build_semantic_index_streaming(vectors, synopsis, tables, sigma, hnsw_m=hnsw_m)
        del vectors  # already emptied by the streaming pop, but drop the (now-empty) dict itself
        gc.collect()
        if verbose:
            print("[{}] lean semantic index built ({} items indexed)".format(
                tier, semantic_obj.n_indexed), flush=True)

        semantic_obj.index.save_index(index_path)
        with open(meta_path, "wb") as f:
            pickle.dump({
                "labels": semantic_obj._labels,
                "n_indexed": semantic_obj.n_indexed,
                "dim": semantic_obj._dim,
            }, f)
        if verbose:
            print("[{}] cached lean semantic index to {}".format(tier, index_path), flush=True)

    overlap_obj = registry.build("overlap", "inverted_index", synopsis=synopsis, tables=tables)
    unionability_obj = registry.build("unionability", "constant")

    return RunnerContext(
        synopsis=synopsis, semantic=semantic_obj, overlap=overlap_obj,
        unionability=unionability_obj, query_vectors=query_vectors_small,
    )


def run_tier(tier: str, top_ns=TOP_NS, k: int = K, alpha: float = ALPHA,
             verbose: bool = True) -> List[SweepRow]:
    from dutsx.runner import QueryTask

    cohort = _load_cohort()
    t0 = time.perf_counter()
    if tier == "tier_1m":
        ctx = _build_context_lean(tier, needed_q_tables={c[0] for c in cohort}, verbose=verbose)
        if verbose:
            print("[{}] lean context built in {:.1f}s".format(
                tier, time.perf_counter() - t0), flush=True)
    else:
        ctx, build_times = build_wdc_context(tier)
        if verbose:
            print("[{}] context built in {:.1f}s ({} tables)".format(
                tier, time.perf_counter() - t0, build_times.n_tables), flush=True)

    pool_size = int(alpha * k)
    rows: List[SweepRow] = []
    for top_n in top_ns:
        n_ran, n_clamped = 0, 0
        for q_table, attr, value in cohort:
            task = QueryTask(q_table=q_table, attr=attr, M={value}, F_star=0.20, delta=0.10,
                              k=k, alpha=alpha, include_query=True, top_n=top_n)
            try:
                D, N_Q, n_Q, _ = retrieve_unscored_candidates(task, ctx)
            except Exception:
                continue
            D = drop_empty(D)
            n_D = len(D)
            if n_D == 0:
                continue
            clamped = pool_size >= n_D

            t1 = time.perf_counter()
            if clamped:
                iters, dink_ran = 0, False
                n_clamped += 1
            else:
                s1 = stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, True)
                iters, dink_ran = s1.iterations, True
                n_ran += 1
            stage1_t = time.perf_counter() - t1

            rows.append(SweepRow(
                tier=tier, q_table=q_table, top_n=top_n, n_D=n_D, clamped=clamped,
                dinkelbach_ran=dink_ran, iterations=iters, stage1_time_s=stage1_t,
            ))
        if verbose:
            print("  [{}] top_n={}: {} ran, {} clamped (of {})".format(
                tier, top_n, n_ran, n_clamped, len(cohort)), flush=True)
    return rows


def summarize(rows: List[SweepRow], tier: str):
    by_topn = {}
    for r in rows:
        if r.tier == tier:
            by_topn.setdefault(r.top_n, []).append(r)

    print("\n=== {} (k={}, alpha={}, pool={}) ===".format(tier, K, ALPHA, POOL_SIZE))
    print("{:>8} {:>10} {:>10} {:>12} {:>10} {:>12}".format(
        "top_n", "mean|D|", "n_ran", "mean_ms", "mean_iter", "iter_range"))
    for top_n in sorted(by_topn):
        grp = by_topn[top_n]
        n_D_vals = [r.n_D for r in grp]
        ran = [r for r in grp if r.dinkelbach_ran]
        n_clamped = sum(1 for r in grp if r.clamped)
        if ran:
            times_ms = [r.stage1_time_s * 1000 for r in ran]
            iters = [r.iterations for r in ran]
            mean_ms, mean_it = statistics.mean(times_ms), statistics.mean(iters)
            it_range = "{}-{}".format(min(iters), max(iters))
        else:
            mean_ms, mean_it, it_range = float("nan"), float("nan"), "n/a"
        print("{:>8} {:>10.1f} {:>10} {:>12.4f} {:>10.2f} {:>12} (clamped {}/{})".format(
            top_n, statistics.mean(n_D_vals), len(ran), mean_ms, mean_it, it_range,
            n_clamped, len(grp)))


if __name__ == "__main__":
    import sys
    tiers = sys.argv[1:] or ["tier_10k", "tier_100k", "tier_1m"]
    all_rows: List[SweepRow] = []
    for tier in tiers:
        all_rows.extend(run_tier(tier))

    import csv
    out_path = os.path.join(REPO, "experiments/results", OUT_CSV_NAME)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SweepRow._fields)
        for r in all_rows:
            w.writerow(r)
    print("\nwrote", out_path, "({} rows)".format(len(all_rows)))

    for tier in tiers:
        summarize(all_rows, tier)
