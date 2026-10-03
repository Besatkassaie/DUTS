"""Builds a ``dutsx.runner.RunnerContext`` from named adapters (PLAN-integration.md
§5's config shape), and loads the santos query set. One place that knows the
real data's paths, so every experiment driver builds its adapters the same
way and shares a single set of on-disk constants.

Adapter construction (index building, pickle loading) is the expensive,
query-independent part -- callers should build one context per distinct
``(synopsis, semantic, overlap, unionability)`` combination and reuse it
across every ``QueryTask``, exactly as ``tests/test_dutsx_runner.py``'s
``real_ctx`` fixture does.
"""
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from dutsx import registry
from dutsx.adapters.unionability import load_starmie_vectors
from dutsx.runner import QueryTask, RunnerContext, load_queries_from_csv

STARMIE_FAIR_ROOT = os.environ.get("STARMIE_FAIR_ROOT", "/u6/bkassaie/starmie_fair")
_DATA_ROOT = os.path.join(STARMIE_FAIR_ROOT, "data")

_VEC_DATALAKE = "cl_datalake_drop_col_tfidf_entity_column_0.pkl"
_VEC_QUERY = "cl_query_drop_col_tfidf_entity_column_0.pkl"


class BenchmarkPaths(NamedTuple):
    """On-disk locations for one benchmark under ``starmie_fair/data``.

    The available benchmarks are not interchangeable -- they differ in both
    query set and datalake size, which matters for any claim about scaling
    with ``|T|``:

        santos   50 queries /  550 datalake   (48 resolve; see below)
        santos2  48 queries /  550 datalake
        santos3  48 queries /  931 datalake
        santos4  50 queries /  931 datalake

    ``protected_attributes_santos*.csv`` each carry 96 rows: 48 base queries
    plus 48 ``*_fair.csv`` fair-rebalanced variants of those same queries
    (identical ``protected_attribute_id`` and ``protected_value``, produced by
    starmie_fair's ``fair_dataset_builder.py``). The ``_fair`` variants live in
    ``santos2/query`` and ``santos3/query``, NOT in ``santos/query`` -- which is
    why the santos benchmark yields 48 usable tasks from a 96-row CSV.
    ``load_queries_from_csv`` reports the rest as ``skipped``.
    """
    name: str
    datalake_dir: str
    query_dir: str
    datalake_vec_pkl: str
    query_vec_pkl: str
    protected_csv: str
    groundtruth_csv: str   # santos-style table-union-search groundtruth (see experiments/groundtruth.py)


def paths_for(benchmark: str = "santos", protected_csv: Optional[str] = None) -> BenchmarkPaths:
    """Standard layout for ``data/<benchmark>/``. ``protected_csv`` defaults to
    ``data/protected_attributes_<benchmark>.csv``; override it to pair a query
    set with a different protected-attribute file. ``groundtruth_csv`` defaults
    to ``data/<benchmark>/<benchmark>_small_benchmark_groundtruth.csv``, present
    for all four santos variants (verified: santos, santos2, santos3, santos4)."""
    root = os.path.join(_DATA_ROOT, benchmark)
    vectors = os.path.join(root, "vectors")
    return BenchmarkPaths(
        name=benchmark,
        datalake_dir=os.path.join(root, "datalake"),
        query_dir=os.path.join(root, "query"),
        datalake_vec_pkl=os.path.join(vectors, _VEC_DATALAKE),
        query_vec_pkl=os.path.join(vectors, _VEC_QUERY),
        protected_csv=protected_csv or os.path.join(
            _DATA_ROOT, "protected_attributes_{}.csv".format(benchmark)),
        groundtruth_csv=os.path.join(
            root, "{}_small_benchmark_groundtruth.csv".format(benchmark)),
    )


SANTOS = paths_for("santos")

# Back-compat module constants -- every existing caller defaults to santos.
SANTOS_ROOT = os.path.join(_DATA_ROOT, "santos")
DATALAKE_DIR = SANTOS.datalake_dir
QUERY_DIR = SANTOS.query_dir
VECTORS_DIR = os.path.join(SANTOS_ROOT, "vectors")
DATALAKE_VEC_PKL = SANTOS.datalake_vec_pkl
QUERY_VEC_PKL = SANTOS.query_vec_pkl
PROTECTED_CSV = SANTOS.protected_csv

THETA_CAT = 50   # the CONFIGURED theta_cat (CLAUDE.md/PLAN-integration.md Phase A) -- CsvSynopsis, not
                  # MetadataStoreSynopsis, is required to actually honor it (dutsx/adapters/synopsis.py).
SIGMA = 0.6


def benchmark_available(paths: Optional[BenchmarkPaths] = None) -> bool:
    p = paths or SANTOS
    return (os.path.isdir(p.datalake_dir) and os.path.isfile(p.datalake_vec_pkl)
            and os.path.isfile(p.query_vec_pkl) and os.path.isfile(p.protected_csv))


def santos_data_available() -> bool:
    """Back-compat alias for ``benchmark_available()`` on santos."""
    return benchmark_available(SANTOS)


def load_shared_resources(theta_cat: int = THETA_CAT, paths: Optional[BenchmarkPaths] = None):
    """Load the query-independent, adapter-independent inputs once: the
    santos file list, both vector pickles, and a ``CsvSynopsis`` (whose
    per-table DataFrame cache is what actually makes reuse worthwhile --
    ``InvertedIndexOverlap``'s construction calls ``categorical_attrs`` for
    every one of 550 tables, i.e. 550 CSV reads, and a fresh ``CsvSynopsis``
    per adapter combination pays that cost again from cold).

    Returns ``(synopsis, datalake_files, datalake_vectors, query_vectors)``.
    Pass these into ``build_santos_context`` across multiple adapter
    combinations (as ``experiments/retrieval_ablation.py`` does) to avoid
    redundant disk I/O; ``build_santos_context`` still works standalone
    (loads its own copies) when only one combination is needed.
    """
    p = paths or SANTOS
    datalake_files = sorted(os.listdir(p.datalake_dir))
    datalake_vectors = load_starmie_vectors(p.datalake_vec_pkl)
    query_vectors = load_starmie_vectors(p.query_vec_pkl)
    synopsis = registry.build("synopsis", "csv",
                               table_dirs=[p.datalake_dir, p.query_dir], theta_cat=theta_cat)
    return synopsis, datalake_files, datalake_vectors, query_vectors


def build_santos_context(
    semantic: str = "hnsw",
    overlap: str = "inverted_index",
    unionability: str = "pinned_match",
    theta_cat: int = THETA_CAT,
    sigma: float = SIGMA,
    synopsis=None,
    datalake_files: Optional[List[str]] = None,
    datalake_vectors: Optional[Dict] = None,
    query_vectors: Optional[Dict] = None,
    paths: Optional[BenchmarkPaths] = None,
) -> Tuple[RunnerContext, Dict[str, str]]:
    """Build one ``RunnerContext`` against the real santos benchmark, using
    ``dutsx.registry`` to resolve adapter names -- the same config-driven
    selection PLAN-integration.md §5 specifies. Returns ``(ctx, adapter_names)``
    so callers/rows can record exactly which adapters produced a result.

    Pass ``synopsis``/``datalake_files``/``datalake_vectors``/``query_vectors``
    (from ``load_shared_resources``) to build several adapter combinations
    without re-reading the CSVs/pickles from disk each time; omit them for a
    single one-off build, which loads its own copies.
    """
    p = paths or SANTOS
    if datalake_files is None:
        datalake_files = sorted(os.listdir(p.datalake_dir))
    if datalake_vectors is None:
        datalake_vectors = load_starmie_vectors(p.datalake_vec_pkl)
    if query_vectors is None:
        query_vectors = load_starmie_vectors(p.query_vec_pkl)
    if synopsis is None:
        synopsis = registry.build("synopsis", "csv",
                                   table_dirs=[p.datalake_dir, p.query_dir], theta_cat=theta_cat)

    semantic_vectors = categorical_only_vectors(datalake_vectors, synopsis, datalake_files)
    semantic_obj = registry.build("semantic", semantic, vectors=semantic_vectors, sigma=sigma)
    overlap_obj = registry.build("overlap", overlap, synopsis=synopsis, tables=datalake_files)

    if unionability == "constant":
        unionability_obj = registry.build("unionability", "constant")
    elif unionability == "starmie_verify":
        unionability_obj = registry.build(
            "unionability", "starmie_verify", query_vectors=query_vectors,
            candidate_vectors=datalake_vectors, threshold=sigma, starmie_root=STARMIE_FAIR_ROOT,
        )
    else:
        unionability_obj = registry.build(
            "unionability", unionability, query_vectors=query_vectors,
            candidate_vectors=datalake_vectors, threshold=sigma,
        )

    ctx = RunnerContext(
        synopsis=synopsis, semantic=semantic_obj, overlap=overlap_obj,
        unionability=unionability_obj, query_vectors=query_vectors,
    )
    names = {"synopsis": "csv", "semantic": semantic, "overlap": overlap,
              "unionability": unionability, "benchmark": p.name}
    return ctx, names


def categorical_only_vectors(
    vectors: Dict[str, np.ndarray], synopsis, tables: List[str]
) -> Dict[str, np.ndarray]:
    """Zero out every non-categorical column's embedding, so a semantic index built from the
    result indexes exactly the same ``(table, attr)`` population as an overlap index built from
    the same ``synopsis`` (categorical-only, ``theta_cat``-bounded).

    Necessary because ``HnswRetriever`` has no categorical filter of its own -- it indexes every
    column with a nonzero embedding (``dutsx/adapters/semantic.py::_flatten`` skips zero-norm rows
    only), while ``InvertedIndexOverlap`` only ever indexes ``synopsis.categorical_attrs`` columns.
    Left unfiltered, semantic retrieval can surface a candidate ``(table, attr)`` pair whose
    ``attr`` is not categorical -- not usable as a comparable ``V_D`` for that candidate table,
    since only categorical attributes have a value distribution / ``N_i`` at all.

    Zeroing rather than dropping columns preserves each row's position as the real column index,
    which retrieval and unionability rely on downstream as ``attr_idx`` -- found and fixed in the
    WDC context builder first (2026-08-14), then generalized here so every context builder that
    constructs a semantic index (santos-family included) gets the same fix, not just WDC.

    Applies to the CANDIDATE/datalake side only -- callers should filter whatever vectors dict
    feeds ``registry.build("semantic", ...)``, never the query side (a query's own attribute
    embedding is used as-is; only the indexed population needs to match the overlap filter's).
    """
    out: Dict[str, np.ndarray] = {}
    for table in tables:
        arr = vectors.get(table)
        if arr is None:
            continue
        arr = np.array(arr, dtype=np.float32, copy=True)
        cat_idx = set(synopsis.categorical_attrs(table))
        for j in range(arr.shape[0]):
            if j not in cat_idx:
                arr[j] = 0.0
        out[table] = arr
    return out


def column_coverage_stats(synopsis, tables: List[str]) -> Dict[str, float]:
    """Fraction of columns ``theta_cat`` classifies as categorical, summed
    across ``tables``. Generic instrumentation (not benchmark-specific) --
    added while investigating the WDC scalability study, where median table
    size (~6 rows) makes ``nunique(col) <= theta_cat`` classify nearly every
    column as categorical (``nunique`` is capped by ``n_rows`` itself at that
    table size), unlike santos-family tables (~5,000 rows), where the same
    absolute cap is a real, selective filter. Deliberately a standalone
    function rather than folded into ``build_santos_context``'s return value
    -- that would change its signature and every one of its 7 existing call
    sites; this is opt-in, purely additive instrumentation instead.

    Requires a ``CsvSynopsis`` (or anything exposing its ``n_columns``
    extension beyond the ``SynopsisSource`` protocol), not any adapter.
    """
    n_total = 0
    n_categorical = 0
    for table in tables:
        n_total += synopsis.n_columns(table)
        n_categorical += len(synopsis.categorical_attrs(table))
    return {
        "n_tables": len(tables),
        "n_columns_total": n_total,
        "n_columns_categorical": n_categorical,
        "categorical_coverage_ratio": (n_categorical / n_total) if n_total else 0.0,
    }


def load_base_tasks(
    k: int, alpha: float, F_star: float, delta: float,
    include_query: bool = True, top_n: int = 100,
    paths: Optional[BenchmarkPaths] = None,
) -> Tuple[List[QueryTask], List[str]]:
    """``load_queries_from_csv`` against a benchmark's real paths -- returns
    ``(tasks, skipped_q_names)``. On santos that is 48 usable tasks out of 96
    CSV rows (the other 48 are ``*_fair.csv`` variants living in
    santos2/santos3; see ``BenchmarkPaths``). Callers vary ``k``/``alpha``/etc.
    per condition via ``task._replace(...)``."""
    p = paths or SANTOS
    return load_queries_from_csv(
        p.protected_csv, p.query_dir, k=k, alpha=alpha, F_star=F_star, delta=delta,
        include_query=include_query, top_n=top_n,
    )


def load_santos_base_tasks(*args, **kwargs) -> Tuple[List[QueryTask], List[str]]:
    """Back-compat alias for ``load_base_tasks`` on santos."""
    return load_base_tasks(*args, **kwargs)
