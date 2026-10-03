"""Disk cache for built HNSW semantic indexes, so retrieval is reproducible across runs.

Why this exists: ``hnswlib.add_items`` is multi-threaded, so a fresh build produces a
slightly different graph every time. For most queries that moves ``|D|`` by +/-1, but it is
not always benign -- on tier_100k, six duplicate cohort queries sharing (attr=4, value=2)
flipped between ``|D|=234`` and ``|D|=3102`` across two builds, shifting that tier's mean
``|D|`` by 14%. Caching the graph makes every downstream number reproducible.

tier_1m already had its own cache (``wdc_topn_pool_sweep._build_context_lean``, M=16, driven
by the 32GB RSS ceiling); this module is the general M=32 equivalent for every other dataset.

The cache key includes ``sigma`` and ``theta_cat`` because both change WHICH columns are
indexed, and ``n_indexed``/``dim`` are re-checked on load so a stale cache is rebuilt rather
than silently used.
"""
import os
import pickle

CACHE_DIR = os.environ.get("DUTS_HNSW_CACHE", "/u6/bkassaie/wdc_data/cache")


def cache_paths(key: str, m: int, sigma: float, theta_cat: int):
    os.makedirs(CACHE_DIR, exist_ok=True)
    stem = "{}_hnsw_m{}_tc{}_sig{}".format(key, m, theta_cat, str(sigma).replace(".", ""))
    return (os.path.join(CACHE_DIR, stem + ".bin"),
            os.path.join(CACHE_DIR, stem + ".meta.pkl"))


def _load(index_path, meta_path, sigma, expect_n):
    import hnswlib
    from dutsx.adapters.semantic import HnswRetriever
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # Compare against the SOURCE vector count, not n_indexed: HnswRetriever._flatten drops
    # zero-norm vectors, so n_indexed is legitimately smaller (e.g. 487,728 indexed from
    # 518,131 source columns on tier_100k). Comparing to n_indexed rejected every cache.
    if meta.get("n_source") != expect_n:
        return None  # stale (or written before n_source was recorded): rebuild
    r = HnswRetriever.__new__(HnswRetriever)
    r.sigma = sigma
    r._ef = 100
    r._labels = meta["labels"]
    r.n_indexed = meta["n_indexed"]
    r._dim = meta["dim"]
    r.index = hnswlib.Index(space="cosine", dim=max(meta["dim"], 1))
    if meta["n_indexed"] > 0:
        r.index.load_index(index_path, max_elements=meta["n_indexed"])
        r.index.set_ef(r._ef)
    return r


def get_or_build(key, semantic_vectors, sigma, theta_cat, m=32, verbose=True):
    """Load the cached HNSW index for ``key``, or build it once and cache it.

    ``semantic_vectors`` is the already-filtered {table: ndarray} passed to HnswRetriever.
    Returns an object satisfying ``ports.SemanticRetriever``.
    """
    from dutsx import registry
    index_path, meta_path = cache_paths(key, m, sigma, theta_cat)
    expect_n = sum(len(v) for v in semantic_vectors.values())

    if os.path.isfile(index_path) and os.path.isfile(meta_path):
        r = _load(index_path, meta_path, sigma, expect_n)
        if r is not None:
            if verbose:
                print("[{}] loaded cached HNSW ({} items) from {}".format(
                    key, r.n_indexed, index_path), flush=True)
            return r
        if verbose:
            print("[{}] cached HNSW is stale (expected {} items) -- rebuilding".format(
                key, expect_n), flush=True)

    r = registry.build("semantic", "hnsw", vectors=semantic_vectors, sigma=sigma)
    try:
        r.index.save_index(index_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"labels": r._labels, "n_indexed": r.n_indexed, "dim": r._dim,
                         "n_source": expect_n}, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print("[{}] cached HNSW ({} items) to {}".format(key, r.n_indexed, index_path),
                  flush=True)
    except Exception as e:                      # caching is an optimization, never fatal
        print("[{}] WARNING could not cache HNSW: {}".format(key, e), flush=True)
    return r
