"""Builds a ``dutsx.runner.RunnerContext`` for one WDC tier (PLAN's WDC scalability study, Phase 5).

Modeled on ``experiments/context.py::build_santos_context``, but with its own path resolution
rooted at ``/u6/bkassaie/wdc_data/`` rather than ``context.py::paths_for``'s
``starmie_fair/data/<benchmark>/`` layout -- WDC's raw archives/tiers are new DUTS-side acquisition,
not an existing upstream benchmark. Wires Phases 1-3's artifacts through ``dutsx.registry`` exactly
as ``build_santos_context`` does -- zero changes to ``registry.py``/``runner.py`` needed anywhere.

``query_vectors`` is deliberately the SAME dict as ``candidate_vectors``: WDC has no query/datalake
split (queries are just tier-drawn tables, per ``experiments.wdc.query_selection``), and Phase 3
embeds the whole tier once as "datalake" -- that single pickle already covers every table, including
whichever ones later get selected as queries.
"""
import os
import time
from typing import NamedTuple, Optional, Tuple

from dutsx import registry
from dutsx.adapters.unionability import load_starmie_vectors
from dutsx.runner import RunnerContext

WDC_ROOT = os.environ.get("WDC_DATA_ROOT", "/u6/bkassaie/wdc_data")
THETA_CAT = 50
SIGMA = 0.6


class WdcPaths(NamedTuple):
    tier: str
    csv_dir: str
    vectors_pkl: str
    metadata_pkl: str


def paths_for_tier(tier: str, root: str = WDC_ROOT) -> WdcPaths:
    return WdcPaths(
        tier=tier,
        csv_dir=os.path.join(root, "tiers", tier, "csv"),
        vectors_pkl=os.path.join(root, "vectors", "%s_roberta.pkl" % tier),
        metadata_pkl=os.path.join(root, "indexes", "%s_metadata.pkl" % tier),
    )


def tier_available(paths: WdcPaths) -> bool:
    return os.path.isdir(paths.csv_dir) and os.path.isfile(paths.vectors_pkl)


class WdcBuildTimes(NamedTuple):
    """Index-build cost breakdown for one tier -- same shape as
    ``santoslarge_study.py``'s ``BuildTimes``, extended with the two phases
    santos-family never needed (embeddings were always prebuilt there):
    ``embedding_extraction_s`` (Phase 3, read from its own timing sidecar,
    not re-timed here since it's a separate offline GPU step) and
    ``csv_conversion_s`` (Phase 1, read from its own stats file)."""
    tier: str
    n_tables: int
    synopsis_source: str
    csv_conversion_s: Optional[float]
    embedding_extraction_s: Optional[float]
    load_resources_s: float
    synopsis_build_s: float
    hnsw_build_s: float
    inverted_index_build_s: float
    total_index_s: float
    n_columns_total: int
    n_columns_categorical: int
    categorical_coverage_ratio: float
    parse_failed: int
    parse_recovered: int


def _read_json_stat(path: str, key: str):
    import json
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(key)


def build_wdc_context(
    tier: str,
    semantic: str = "hnsw",
    overlap: str = "inverted_index",
    unionability: str = "pinned_match",
    theta_cat: int = THETA_CAT,
    sigma: float = SIGMA,
    root: str = WDC_ROOT,
) -> Tuple[RunnerContext, WdcBuildTimes]:
    """Build one ``RunnerContext`` over a WDC tier, timing each index-build
    phase separately -- the direct WDC analogue of
    ``santoslarge_study.py::build_context``."""
    from experiments.context import categorical_only_vectors, column_coverage_stats

    paths = paths_for_tier(tier, root)
    if not tier_available(paths):
        raise FileNotFoundError(
            "tier %r not available: expected %s and %s" % (tier, paths.csv_dir, paths.vectors_pkl)
        )

    t0 = time.time()
    tables = sorted(f for f in os.listdir(paths.csv_dir) if f.endswith(".csv"))
    vectors = load_starmie_vectors(paths.vectors_pkl)
    load_resources_s = time.time() - t0

    t0 = time.time()
    if os.path.isfile(paths.metadata_pkl):
        # Prebuilt MetadataStore pickle (scripts/wdc_build_metadata_store.py) -- O(1) dict
        # lookups per N_i/n_i call instead of CsvSynopsis's uncached pandas value_counts()
        # recompute on every call (RESULTS-wdc-sweep.md §6: that recompute was 72-90% of
        # end-to-end time). Verified against CsvSynopsis as an independent oracle before this
        # was wired in (scripts/wdc_verify_metadata_store.py) -- exhaustive n_rows agreement on
        # tier_10k, sampled agreement on tier_100k, one documented narrow divergence on 2/100,000
        # corrupted-byte tables (see that script's module docstring), not a systematic issue.
        synopsis_source = "metadata_store"
        synopsis = registry.build(
            "synopsis", "metadata_store", pkl_path=paths.metadata_pkl, theta_cat=theta_cat,
        )
    else:
        # Fallback for any tier without a prebuilt pickle yet (e.g. tier_1m, if run before its
        # own pickle is built) -- recomputes from CSVs directly, no setup required.
        synopsis_source = "csv"
        synopsis = registry.build("synopsis", "csv", table_dirs=[paths.csv_dir], theta_cat=theta_cat)
    synopsis_build_s = time.time() - t0

    t0 = time.time()
    semantic_vectors = categorical_only_vectors(vectors, synopsis, tables)
    if semantic == "hnsw":
        # Cached graph -> reproducible |D| across runs (experiments/semantic_cache.py)
        from experiments.semantic_cache import get_or_build
        semantic_obj = get_or_build(tier, semantic_vectors, sigma, theta_cat)
    else:
        semantic_obj = registry.build("semantic", semantic, vectors=semantic_vectors, sigma=sigma)
    hnsw_build_s = time.time() - t0

    t0 = time.time()
    overlap_obj = registry.build("overlap", overlap, synopsis=synopsis, tables=tables)
    inverted_index_build_s = time.time() - t0

    coverage = column_coverage_stats(synopsis, tables)

    if unionability == "constant":
        unionability_obj = registry.build("unionability", "constant")
    else:
        unionability_obj = registry.build(
            "unionability", unionability, query_vectors=vectors,
            candidate_vectors=vectors, threshold=sigma,
        )

    ctx = RunnerContext(
        synopsis=synopsis, semantic=semantic_obj, overlap=overlap_obj,
        unionability=unionability_obj, query_vectors=vectors,
    )

    conversion_stats_path = os.path.join(paths.csv_dir, "_conversion_stats.json")
    timing_sidecar_path = paths.vectors_pkl.rsplit(".", 1)[0] + ".timing.json"

    build_times = WdcBuildTimes(
        tier=tier,
        n_tables=len(tables),
        synopsis_source=synopsis_source,
        csv_conversion_s=_read_json_stat(conversion_stats_path, "elapsed_s"),
        embedding_extraction_s=_read_json_stat(timing_sidecar_path, "elapsed_s"),
        load_resources_s=round(load_resources_s, 3),
        synopsis_build_s=round(synopsis_build_s, 3),
        hnsw_build_s=round(hnsw_build_s, 3),
        inverted_index_build_s=round(inverted_index_build_s, 3),
        total_index_s=round(
            load_resources_s + synopsis_build_s + hnsw_build_s + inverted_index_build_s, 3
        ),
        n_columns_total=coverage["n_columns_total"],
        n_columns_categorical=coverage["n_columns_categorical"],
        categorical_coverage_ratio=coverage["categorical_coverage_ratio"],
        parse_failed=_read_json_stat(conversion_stats_path, "n_failed") or 0,
        parse_recovered=_read_json_stat(conversion_stats_path, "n_recovered") or 0,
    )
    return ctx, build_times
