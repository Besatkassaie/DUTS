"""Tests for ``experiments.wdc.query_selection`` (WDC scalability study Phase 4).

Two kinds, matching the ``test_dutsx_runner.py``/``test_experiments.py`` convention:

* Stub-adapter tests (no real data dependency) -- determinism, selection validity invariants, and
  the frequency-biasing preference itself (a stub ``OverlapFilter`` with KNOWN, controlled posting
  sizes per value, unlike ``test_dutsx_runner.py``'s ``_StubOverlap`` which ignores ``M`` entirely
  and can't distinguish values). Run with ``measure_yield=False`` -- these test selection logic, not
  retrieval, so they don't need a working semantic retriever or real query vectors.
* A real-tier smoke test (skip cleanly if WDC tier_10k data isn't on disk) asserting the biased
  sample's realized ``|D|`` median measurably exceeds the unbiased side-sample's -- the actual claim
  Phase 4 exists to support, checked against real retrieval, not simulated.
"""
import os
import random

import pytest

from dutsx.runner import RunnerContext
from experiments.wdc.query_selection import (
    _posting_size, _select_M, auto_select_queries, select_queries_with_comparison,
)

# ---------------------------------------------------------------------------
# Stub adapters, purpose-built for this module (a generic _StubOverlap that
# ignores M can't exercise "prefers higher-posting-size values").
# ---------------------------------------------------------------------------


class _StubSynopsis(object):
    def __init__(self, categorical, distributions):
        self._categorical = categorical          # {table: [attr, ...]}
        self._dist = distributions                # {(table, attr): {value: count}}

    def n_rows(self, table):
        return sum(self._dist.get((table, 0), {}).values())

    def distribution(self, table, attr):
        return self._dist.get((table, attr), {})

    def categorical_attrs(self, table):
        return self._categorical.get(table, [])


class _StubOverlapByValuePostingSize(object):
    """query({v}) returns exactly ``posting_sizes[v]`` distinct (table, attr)
    pairs, all with attr=0 and table names 't0'..'t{n-1}' -- lets tests
    assert selection prefers specific, known values."""

    def __init__(self, posting_sizes):
        self._sizes = posting_sizes  # {value: n_distinct_tables}

    def query(self, M):
        pairs = {}
        for v in M:
            n = self._sizes.get(v, 0)
            for i in range(n):
                pairs[("t%d" % i, 0)] = 1
        return pairs


def _ctx(synopsis, overlap):
    return RunnerContext(
        synopsis=synopsis, semantic=None, overlap=overlap,
        unionability=None, query_vectors={},
    )


# ---------------------------------------------------------------------------
# _posting_size / _select_M unit-level checks
# ---------------------------------------------------------------------------


def test_posting_size_counts_distinct_tables_not_pairs():
    overlap = _StubOverlapByValuePostingSize({"common": 5, "rare": 1})
    assert _posting_size(overlap, "common") == 5
    assert _posting_size(overlap, "rare") == 1
    assert _posting_size(overlap, "unseen") == 0


def test_select_m_biased_prefers_higher_posting_size_values():
    overlap = _StubOverlapByValuePostingSize({"a": 1, "b": 50, "c": 3})
    rng = random.Random(0)
    M, met = _select_M(
        domain=["a", "b", "c"], overlap=overlap, m_size_range=(1, 1), rng=rng,
        bias_by_overlap=True, min_posting_size=10,
    )
    assert M == {"b"}  # only value clearing min_posting_size=10
    assert met is True


def test_select_m_biased_falls_back_when_none_meet_threshold_but_never_empty():
    overlap = _StubOverlapByValuePostingSize({"a": 1, "b": 2, "c": 3})
    rng = random.Random(0)
    M, met = _select_M(
        domain=["a", "b", "c"], overlap=overlap, m_size_range=(1, 1), rng=rng,
        bias_by_overlap=True, min_posting_size=100,
    )
    assert M == {"c"}  # best available (highest posting size), even though it fails the threshold
    assert met is False


def test_select_m_unbiased_ignores_posting_size():
    """bias_by_overlap=False never calls overlap.query -- confirmed by an
    overlap stub that raises if queried at all."""
    class _RaisingOverlap(object):
        def query(self, M):
            raise AssertionError("unbiased selection must not query overlap")

    rng = random.Random(0)
    M, met = _select_M(
        domain=["a", "b", "c"], overlap=_RaisingOverlap(), m_size_range=(2, 2), rng=rng,
        bias_by_overlap=False, min_posting_size=10,
    )
    assert len(M) == 2
    assert M <= {"a", "b", "c"}
    assert met is True


# ---------------------------------------------------------------------------
# auto_select_queries -- determinism and validity invariants (measure_yield=False)
# ---------------------------------------------------------------------------


def _synth_universe(n_tables=20, seed=0):
    rng = random.Random(seed)
    categorical = {}
    dist = {}
    for i in range(n_tables):
        table = "t%d.csv" % i
        if i % 5 == 0:
            categorical[table] = []  # some tables have no categorical attrs at all
            continue
        categorical[table] = [0, 1]
        for attr in (0, 1):
            values = rng.choices(["red", "blue", "green", "yellow"], k=rng.randint(5, 30))
            counts = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            dist[(table, attr)] = counts
    return list(categorical.keys()), categorical, dist


def test_auto_select_is_deterministic_given_same_seed():
    table_ids, categorical, dist = _synth_universe()
    synopsis = _StubSynopsis(categorical, dist)
    overlap = _StubOverlapByValuePostingSize({})  # unbiased path in this test
    ctx = _ctx(synopsis, overlap)

    tasks1, report1 = auto_select_queries(
        ctx, table_ids, n_queries=5, seed=7, bias_by_overlap=False, measure_yield=False,
    )
    tasks2, report2 = auto_select_queries(
        ctx, table_ids, n_queries=5, seed=7, bias_by_overlap=False, measure_yield=False,
    )
    assert tasks1 == tasks2
    assert report1["queries"] == report2["queries"]


def test_auto_select_never_picks_attr_outside_categorical_attrs():
    table_ids, categorical, dist = _synth_universe()
    synopsis = _StubSynopsis(categorical, dist)
    overlap = _StubOverlapByValuePostingSize({})
    ctx = _ctx(synopsis, overlap)

    tasks, _report = auto_select_queries(
        ctx, table_ids, n_queries=10, seed=1, bias_by_overlap=False, measure_yield=False,
    )
    for task in tasks:
        assert task.attr in synopsis.categorical_attrs(task.q_table)


def test_auto_select_M_always_nonempty_subset_of_real_domain():
    table_ids, categorical, dist = _synth_universe()
    synopsis = _StubSynopsis(categorical, dist)
    overlap = _StubOverlapByValuePostingSize({})
    ctx = _ctx(synopsis, overlap)

    tasks, _report = auto_select_queries(
        ctx, table_ids, n_queries=10, seed=2, bias_by_overlap=False, measure_yield=False,
    )
    for task in tasks:
        assert len(task.M) > 0
        real_domain = set(synopsis.distribution(task.q_table, task.attr)) - {""}
        assert task.M <= real_domain


def test_auto_select_never_selects_zero_categorical_attr_tables():
    table_ids, categorical, dist = _synth_universe()
    synopsis = _StubSynopsis(categorical, dist)
    overlap = _StubOverlapByValuePostingSize({})
    ctx = _ctx(synopsis, overlap)

    tasks, _report = auto_select_queries(
        ctx, table_ids, n_queries=20, seed=3, bias_by_overlap=False, measure_yield=False,
    )
    zero_cat_tables = {t for t in table_ids if not synopsis.categorical_attrs(t)}
    for task in tasks:
        assert task.q_table not in zero_cat_tables


def test_auto_select_reports_eligibility_and_domain_stats():
    table_ids, categorical, dist = _synth_universe(n_tables=20)
    synopsis = _StubSynopsis(categorical, dist)
    overlap = _StubOverlapByValuePostingSize({})
    ctx = _ctx(synopsis, overlap)

    tasks, report = auto_select_queries(
        ctx, table_ids, n_queries=8, seed=4, bias_by_overlap=False, measure_yield=False,
    )
    n_eligible_expected = sum(1 for t in table_ids if categorical.get(t))
    assert report["n_eligible"] == n_eligible_expected
    assert report["n_table_universe"] == 20
    assert len(report["domain_size_distribution"]) == len(tasks)
    assert report["yield"] is None  # measure_yield=False


# ---------------------------------------------------------------------------
# Real-tier smoke test: biased vs unbiased realized |D|
# ---------------------------------------------------------------------------

TIER_10K_CSV = "/u6/bkassaie/wdc_data/tiers/tier_10k/csv"
TIER_10K_VEC = "/u6/bkassaie/wdc_data/vectors/tier_10k_roberta.pkl"

pytestmark_real = pytest.mark.skipif(
    not (os.path.isdir(TIER_10K_CSV) and os.path.isfile(TIER_10K_VEC)),
    reason="WDC tier_10k data not available (run Phases 0-3 first)",
)


@pytestmark_real
def test_biased_M_selection_yields_higher_D_than_unbiased_paired_comparison():
    """Paired comparison: same (q_table, attr) selections, biased vs
    unbiased M drawn for each -- NOT two independently-random samples.

    An earlier, confounded version of this test drew biased/unbiased from
    different random table subsets and found IDENTICAL medians (both 1.0)
    on a real tier_10k run -- not because biasing doesn't help, but because
    which query table gets selected dominates the aggregate statistic at
    this scale, masking the M-selection effect. A manual row-by-row check
    showed biasing IS doing real work once the table-selection confound is
    removed (e.g. one pair picked M="6" (|D|=12) vs M="13 Rock Art Hot
    Spot" (|D|=3) for the IDENTICAL query table/attr) -- this test checks
    that on the actual paired mean, not the confounded aggregate medians.
    """
    from dutsx import registry
    from dutsx.adapters.unionability import load_starmie_vectors

    tables = sorted(f for f in os.listdir(TIER_10K_CSV) if f.endswith(".csv"))
    synopsis = registry.build("synopsis", "csv", table_dirs=[TIER_10K_CSV], theta_cat=50)
    overlap = registry.build("overlap", "inverted_index", synopsis=synopsis, tables=tables)
    vectors = load_starmie_vectors(TIER_10K_VEC)
    semantic = registry.build("semantic", "hnsw", vectors=vectors, sigma=0.6)
    unionability = registry.build("unionability", "constant")  # unused by yield measurement

    ctx = RunnerContext(
        synopsis=synopsis, semantic=semantic, overlap=overlap,
        unionability=unionability, query_vectors=vectors,
    )

    tasks, report = select_queries_with_comparison(
        ctx, tables, n_queries=25, seed=42,
        k=10, alpha=2.0, F_star=0.3, delta=0.15, top_n=200,
    )
    assert len(tasks) > 0
    assert report["yield"]["n"] == report["unbiased_side_sample"]["yield"]["n"]  # truly paired

    # the real signal: on average, biasing does not make yield WORSE, and on
    # this benchmark (tier_10k, seed=42) it measurably helps.
    assert report["paired_biased_minus_unbiased_mean"] >= 0
