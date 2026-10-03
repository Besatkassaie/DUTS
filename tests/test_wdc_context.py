"""Tests for ``experiments.wdc.context`` (WDC scalability study Phase 5).

Real-tier smoke test only, skip cleanly if WDC tier_10k data isn't on disk -- matches every other
``test_dutsx_*``/context real-data test's convention. No stub-based half exists here: unlike
Phase 4's selection logic, there's nothing in this module worth testing without real adapters --
it's pure wiring over ``dutsx.registry``, already covered by that registry's own tests.

Module-scoped fixture: building a tier_10k context costs real time (~30s for the overlap index
alone) -- share one build across every test in this module rather than paying that four times.
"""
import os

import pytest

from dutsx.ports import OverlapFilter, SemanticRetriever, SynopsisSource, UnionabilityScorer
from dutsx.runner import run_query
from experiments.wdc.context import build_wdc_context, paths_for_tier, tier_available
from experiments.wdc.query_selection import auto_select_queries

pytestmark = pytest.mark.skipif(
    not tier_available(paths_for_tier("tier_10k")),
    reason="WDC tier_10k data not available (run Phases 0-3 first)",
)


@pytest.fixture(scope="module")
def tier_10k_ctx():
    return build_wdc_context("tier_10k")


def test_build_wdc_context_adapters_satisfy_protocols(tier_10k_ctx):
    ctx, _build_times = tier_10k_ctx
    assert isinstance(ctx.synopsis, SynopsisSource)
    assert isinstance(ctx.semantic, SemanticRetriever)
    assert isinstance(ctx.overlap, OverlapFilter)
    assert isinstance(ctx.unionability, UnionabilityScorer)


def test_build_wdc_context_build_times_are_sane(tier_10k_ctx):
    _ctx, build_times = tier_10k_ctx
    assert build_times.tier == "tier_10k"
    assert build_times.n_tables == 10000
    assert build_times.total_index_s > 0
    assert build_times.n_columns_total >= build_times.n_columns_categorical > 0
    assert 0.0 <= build_times.categorical_coverage_ratio <= 1.0
    assert build_times.embedding_extraction_s is not None  # sidecar found
    assert build_times.csv_conversion_s is not None


def test_query_vectors_same_object_as_candidate_vectors_no_split(tier_10k_ctx):
    ctx, _build_times = tier_10k_ctx
    # WDC has no query/datalake split -- both sides of PinnedMatchScorer were
    # built from the identical vectors dict (see module docstring).
    assert ctx.unionability.query_vectors is ctx.unionability.candidate_vectors
    assert ctx.unionability.query_vectors is ctx.query_vectors


def test_real_run_query_end_to_end_on_tier_10k_smoke(tier_10k_ctx):
    ctx, _build_times = tier_10k_ctx
    tables = sorted(f for f in os.listdir(paths_for_tier("tier_10k").csv_dir) if f.endswith(".csv"))

    tasks, report = auto_select_queries(
        ctx, tables, n_queries=3, seed=42, k=5, alpha=2.0,
        F_star=0.3, delta=0.15, top_n=200, measure_yield=False,
    )
    assert len(tasks) == 3

    for task in tasks:
        result = run_query(task, ctx)
        assert result is not None
        assert result.telemetry.n_D >= 0
        assert result.telemetry.retrieval_time_s is not None
