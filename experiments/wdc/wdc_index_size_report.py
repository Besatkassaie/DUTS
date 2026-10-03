"""Index build time + on-disk size, HNSW and inverted index, across all three WDC tiers --
the WDC analogue of the santosLarge index-cost verification (same approach: build the context
once per tier, read the already-measured build times off ``WdcBuildTimes``, then separately
serialize each index to measure its real on-disk footprint -- neither index is persisted by the
normal pipeline, both are rebuilt in-memory on every ``build_wdc_context`` call).

HNSW size: ``hnswlib.Index.save_index(path)`` (native serialization) -- ``ctx.semantic.index`` is
the raw ``hnswlib.Index`` object (dutsx/adapters/semantic.py::HnswRetriever.index).
Inverted index size: ``pickle.dump(ctx.overlap._postings, ...)`` -- the postings dict
(dutsx/adapters/overlap.py::InvertedIndexOverlap._postings).

Usage:
    python -m experiments.wdc.wdc_index_size_report tier_10k tier_100k tier_1m
"""
import gc
import os
import pickle
import sys
import time

from dutsx import registry
from dutsx.adapters.unionability import load_starmie_vectors

from .context import THETA_CAT, SIGMA, WdcBuildTimes, paths_for_tier, tier_available

REPO = "/u6/bkassaie/DUTS"
TMP_DIR = "/tmp/wdc_index_size_report"


def measure_tier_lean(tier: str, theta_cat: int = THETA_CAT, sigma: float = SIGMA,
                       verbose: bool = True) -> dict:
    """Like ``measure_tier`` but frees each large intermediate as soon as it's no longer
    needed, instead of assembling a full ``RunnerContext`` (which keeps the raw vectors dict,
    the categorical-filtered COPY of it, and hnswlib's own internal copy all alive at once --
    three ~16GB copies for tier_1m, comfortably over this account's hard 32GB per-process RSS
    ulimit (``ulimit -m`` -- confirmed via ``ulimit -Hm``/``-Sm``, not adjustable). Two
    consecutive tier_1m runs were silently killed (no traceback -- consistent with a
    ulimit-triggered SIGKILL) at exactly this point before this fix. Never needed at tier_10k/
    tier_100k scale (~1.6GB/16GB of vector data, nowhere near the ceiling), which is why this
    wasn't caught until tier_1m.

    Doesn't build a ``unionability`` object or return a ``RunnerContext`` -- this script only
    ever needs the semantic/overlap indices themselves, never real retrieval."""
    from experiments.context import categorical_only_vectors, column_coverage_stats

    paths = paths_for_tier(tier)
    if not tier_available(paths):
        raise FileNotFoundError("tier %r not available" % tier)

    t0 = time.time()
    tables = sorted(f for f in os.listdir(paths.csv_dir) if f.endswith(".csv"))
    vectors = load_starmie_vectors(paths.vectors_pkl)
    load_resources_s = time.time() - t0
    if verbose:
        print("[{}] loaded {} tables' vectors in {:.1f}s".format(
            tier, len(tables), load_resources_s), flush=True)

    t0 = time.time()
    synopsis = registry.build(
        "synopsis", "metadata_store", pkl_path=paths.metadata_pkl, theta_cat=theta_cat,
    )
    synopsis_build_s = time.time() - t0

    t0 = time.time()
    semantic_vectors = categorical_only_vectors(vectors, synopsis, tables)
    del vectors  # not needed again -- no unionability object built in this lean path
    gc.collect()
    semantic_obj = registry.build("semantic", "hnsw", vectors=semantic_vectors, sigma=sigma)
    del semantic_vectors  # hnswlib.add_items already copied what it needs internally
    gc.collect()
    hnsw_build_s = time.time() - t0
    if verbose:
        print("[{}] HNSW built in {:.1f}s".format(tier, hnsw_build_s), flush=True)

    os.makedirs(TMP_DIR, exist_ok=True)
    hnsw_path = os.path.join(TMP_DIR, "{}_hnsw.bin".format(tier))
    t0 = time.time()
    semantic_obj.index.save_index(hnsw_path)
    hnsw_save_s = time.time() - t0
    hnsw_size = os.path.getsize(hnsw_path)
    os.remove(hnsw_path)
    del semantic_obj
    gc.collect()
    if verbose:
        print("[{}] HNSW index: {:.2f} MB (save took {:.1f}s)".format(
            tier, hnsw_size / 1e6, hnsw_save_s), flush=True)

    t0 = time.time()
    overlap_obj = registry.build("overlap", "inverted_index", synopsis=synopsis, tables=tables)
    inverted_index_build_s = time.time() - t0

    n_distinct_values = len(overlap_obj._postings)
    n_total_postings = sum(len(v) for v in overlap_obj._postings.values())
    ovl_path = os.path.join(TMP_DIR, "{}_postings.pkl".format(tier))
    t0 = time.time()
    with open(ovl_path, "wb") as f:
        pickle.dump(dict(overlap_obj._postings), f)
    ovl_save_s = time.time() - t0
    ovl_size = os.path.getsize(ovl_path)
    os.remove(ovl_path)
    if verbose:
        print("[{}] Inverted index: {:.2f} MB (save took {:.1f}s, {} distinct values, "
              "{} total postings)".format(
                  tier, ovl_size / 1e6, ovl_save_s, n_distinct_values, n_total_postings), flush=True)

    return {
        "tier": tier,
        "n_tables": len(tables),
        "load_resources_s": round(load_resources_s, 3),
        "hnsw_build_s": round(hnsw_build_s, 3),
        "hnsw_size_mb": round(hnsw_size / 1e6, 2),
        "inverted_index_build_s": round(inverted_index_build_s, 3),
        "inverted_index_size_mb": round(ovl_size / 1e6, 2),
        "n_distinct_values": n_distinct_values,
        "n_total_postings": n_total_postings,
        "total_context_build_s": round(
            load_resources_s + synopsis_build_s + hnsw_build_s + inverted_index_build_s, 1),
    }


def measure_tier(tier: str, verbose: bool = True) -> dict:
    from .context import build_wdc_context
    os.makedirs(TMP_DIR, exist_ok=True)
    t0 = time.time()
    ctx, build_times = build_wdc_context(tier)
    total_build_s = time.time() - t0
    if verbose:
        print("[{}] context built in {:.1f}s ({} tables)".format(
            tier, total_build_s, build_times.n_tables), flush=True)

    hnsw_path = os.path.join(TMP_DIR, "{}_hnsw.bin".format(tier))
    t0 = time.time()
    ctx.semantic.index.save_index(hnsw_path)
    hnsw_save_s = time.time() - t0
    hnsw_size = os.path.getsize(hnsw_path)
    os.remove(hnsw_path)
    if verbose:
        print("[{}] HNSW index: {:.2f} MB (save took {:.1f}s)".format(
            tier, hnsw_size / 1e6, hnsw_save_s), flush=True)

    n_distinct_values = len(ctx.overlap._postings)
    n_total_postings = sum(len(v) for v in ctx.overlap._postings.values())
    ovl_path = os.path.join(TMP_DIR, "{}_postings.pkl".format(tier))
    t0 = time.time()
    with open(ovl_path, "wb") as f:
        pickle.dump(dict(ctx.overlap._postings), f)
    ovl_save_s = time.time() - t0
    ovl_size = os.path.getsize(ovl_path)
    os.remove(ovl_path)
    if verbose:
        print("[{}] Inverted index: {:.2f} MB (save took {:.1f}s, {} distinct values, "
              "{} total postings)".format(
                  tier, ovl_size / 1e6, ovl_save_s, n_distinct_values, n_total_postings), flush=True)

    return {
        "tier": tier,
        "n_tables": build_times.n_tables,
        "load_resources_s": build_times.load_resources_s,
        "hnsw_build_s": build_times.hnsw_build_s,
        "hnsw_size_mb": round(hnsw_size / 1e6, 2),
        "inverted_index_build_s": build_times.inverted_index_build_s,
        "inverted_index_size_mb": round(ovl_size / 1e6, 2),
        "n_distinct_values": n_distinct_values,
        "n_total_postings": n_total_postings,
        "total_context_build_s": round(total_build_s, 1),
    }


if __name__ == "__main__":
    tiers = sys.argv[1:] or ["tier_10k", "tier_100k", "tier_1m"]
    rows = []
    for tier in tiers:
        fn = measure_tier_lean if tier == "tier_1m" else measure_tier
        rows.append(fn(tier))
        print(flush=True)

    print("\n{:>10} {:>10} {:>12} {:>10} {:>14} {:>10}".format(
        "tier", "n_tables", "hnsw_build_s", "hnsw_MB", "inv_build_s", "inv_MB"))
    for r in rows:
        print("{:>10} {:>10} {:>12.2f} {:>10.2f} {:>14.2f} {:>10.2f}".format(
            r["tier"], r["n_tables"], r["hnsw_build_s"], r["hnsw_size_mb"],
            r["inverted_index_build_s"], r["inverted_index_size_mb"]))

    import csv
    out_path = os.path.join(REPO, "experiments/results/wdc_index_size_report.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(rows[0].keys())
        for r in rows:
            w.writerow(r.values())
    print("\nwrote", out_path)
