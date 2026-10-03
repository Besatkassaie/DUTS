"""Tests for the Phase D end-to-end runner (PLAN-integration.md §3, §4 Phase D).

Two kinds of tests, matching the existing Phase A/B/C convention:

* Stub-adapter tests (no real data dependency) -- exercise the composition
  logic itself: the structural "score P only, never D" guarantee, degenerate
  cardinalities, all four infeasibility causes, and the CSV loader. These
  run unconditionally.
* Real-data tests against ``/u6/bkassaie/starmie_fair`` (read-only upstream)
  -- the actual 50-query santos run and the ``n_unionability_computations``
  independence-from-|T| demonstration. Skip cleanly if that data isn't
  mounted, same as the other ``test_dutsx_*.py`` modules.
"""
import csv
import os
import random

import numpy as np
import pytest

from dutsx.adapters.synopsis import CsvSynopsis
from dutsx.adapters.semantic import HnswRetriever
from dutsx.adapters.overlap import InvertedIndexOverlap
from dutsx.adapters.unionability import PinnedMatchScorer, load_starmie_vectors
from dutsx.runner import (
    QueryTask,
    RunnerContext,
    UnscoredCandidate,
    load_queries_from_csv,
    retrieve_unscored_candidates,
    run_query,
)
from duts.types import CandidateStats

# ---------------------------------------------------------------------------
# Stub adapters -- deterministic, no real data. Satisfy the four Protocols
# structurally (duck typing); see dutsx/ports.py.
# ---------------------------------------------------------------------------


class _StubSynopsis(object):
    """``n_rows``/``distribution`` from explicit dicts; ``categorical_attrs``
    is unused by the runner directly (only by ``InvertedIndexOverlap``, which
    these tests don't build), so it's a stub."""

    def __init__(self, n_rows, distributions):
        self._n_rows = n_rows
        self._dist = distributions

    def n_rows(self, table):
        return self._n_rows[table]

    def distribution(self, table, attr):
        return self._dist.get((table, attr), {})

    def categorical_attrs(self, table):
        return [0]


class _StubSemantic(object):
    """Ignores the probe vector entirely -- returns a fixed candidate list,
    exactly what every stub-adapter test needs (retrieval mechanics are
    Phase B's concern, already tested in test_dutsx_retrieval.py)."""

    def __init__(self, results):
        self._results = results  # [(table, attr, sim)]

    def query(self, vec, top_n):
        return self._results[:top_n]


class _StubOverlap(object):
    def __init__(self, pairs):
        self._pairs = pairs  # {(table, attr): overlap}

    def query(self, M):
        return dict(self._pairs)


class _StubUnionability(object):
    """Counts every call (mirrors each real adapter's own ``n_scored``) and
    records the exact (q_table, c_table, pin) triples seen -- lets tests
    assert *which* tables got scored, not just how many."""

    def __init__(self, value=1.0):
        self.value = value
        self.n_scored = 0
        self.calls = []

    def score(self, q_table, c_table, pin):
        self.n_scored += 1
        self.calls.append((q_table, c_table, pin))
        return self.value


def _make_uniform_ctx(n_tables, n_i=10, N_i=5, q_n=10, q_N=5, u_value=1.0):
    """``n_tables`` candidates, all identical (N=N_i, n=n_i), plus a query
    table with its own (q_N, q_n). Every candidate is attr 0, overlap 1,
    similarity descending by table id so retrieval order is deterministic."""
    tables = [f"t{i}" for i in range(n_tables)]
    n_rows = {t: n_i for t in tables}
    n_rows["Q"] = q_n
    dist = {}
    for t in tables:
        dist[(t, 0)] = {"m": N_i, "x": max(n_i - N_i, 0)}
    dist[("Q", 0)] = {"m": q_N, "x": max(q_n - q_N, 0)}
    synopsis = _StubSynopsis(n_rows, dist)
    semantic = _StubSemantic([(t, 0, 1.0 - i * 1e-6) for i, t in enumerate(tables)])
    overlap = _StubOverlap({(t, 0): 1 for t in tables})
    unionability = _StubUnionability(value=u_value)
    query_vectors = {"Q": np.zeros((1, 4))}
    ctx = RunnerContext(
        synopsis=synopsis, semantic=semantic, overlap=overlap,
        unionability=unionability, query_vectors=query_vectors,
    )
    return ctx, unionability


# ---------------------------------------------------------------------------
# The structural guarantee: score P only, never D.
# ---------------------------------------------------------------------------


def test_unscored_candidate_has_no_U_field():
    """The type-level half of the guarantee: UnscoredCandidate literally
    cannot carry a U, so Stage 2 (which reads .U) cannot silently run on it."""
    assert "U" not in UnscoredCandidate._fields
    assert set(UnscoredCandidate._fields) == {"table", "N", "n", "attr"}


def test_n_unionability_computations_equals_pool_size_not_candidate_size():
    """|D| = 30, alpha*k = 6 -- exactly 6 scoring calls, not 30."""
    ctx, unionability = _make_uniform_ctx(n_tables=30)
    task = QueryTask(q_table="Q", attr=0, M={"m"}, F_star=0.0, delta=1.0, k=3, alpha=2)
    result = run_query(task, ctx)

    assert len(result.candidates) == 30
    assert result.telemetry.n_unionability_computations == 6
    assert unionability.n_scored == 6
    assert len(result.pool) == 6


def test_scoring_calls_are_exactly_the_pool_tables():
    """The scorer is called on precisely P's tables -- not a superset, not a
    subset -- demonstrated by inspecting the recorded call list directly."""
    ctx, unionability = _make_uniform_ctx(n_tables=20)
    task = QueryTask(q_table="Q", attr=0, M={"m"}, F_star=0.0, delta=1.0, k=4, alpha=2)
    result = run_query(task, ctx)

    scored_tables = {c_table for (_, c_table, _) in unionability.calls}
    pool_tables = {c.table for c in result.pool}
    assert scored_tables == pool_tables
    assert len(scored_tables) == 8  # alpha*k


def test_n_unionability_computations_independent_of_candidate_pool_growth():
    """Growing |D| from 20 to 2000 (at fixed k, alpha) does not change the
    number of scoring calls -- the headline claim, isolated from retrieval
    mechanics entirely (stub retrieval returns everything it's given)."""
    counts = []
    for n_tables in (20, 200, 2000):
        ctx, unionability = _make_uniform_ctx(n_tables=n_tables)
        task = QueryTask(q_table="Q", attr=0, M={"m"}, F_star=0.0, delta=1.0, k=5, alpha=2, top_n=3000)
        result = run_query(task, ctx)
        counts.append((n_tables, len(result.candidates), unionability.n_scored))

    assert [c[1] for c in counts] == [20, 200, 2000]     # |D| really did grow
    assert [c[2] for c in counts] == [10, 10, 10]          # scoring calls did NOT


# ---------------------------------------------------------------------------
# Degenerate cardinalities and R subseteq P subseteq D (C5)
# ---------------------------------------------------------------------------


def test_insufficient_candidates_short_circuits_before_any_scoring():
    ctx, unionability = _make_uniform_ctx(n_tables=3)
    task = QueryTask(q_table="Q", attr=0, M={"m"}, F_star=0.0, delta=1.0, k=5, alpha=2)
    result = run_query(task, ctx)

    assert result.selected == []
    assert result.pool == []
    assert len(result.candidates) == 3
    assert result.telemetry.infeasibility_cause == "insufficient_candidates"
    assert unionability.n_scored == 0  # never scored anything -- short-circuited pre-Stage-1


def test_pool_clamped_to_D_when_D_smaller_than_alpha_times_k():
    """k <= |D| < alpha*k -> P == D, never padded (C5); n_unionability ==
    |D|, not alpha*k."""
    ctx, unionability = _make_uniform_ctx(n_tables=7)
    task = QueryTask(q_table="Q", attr=0, M={"m"}, F_star=0.0, delta=1.0, k=5, alpha=3)
    result = run_query(task, ctx)

    assert len(result.pool) == 7  # clamped to |D|, not alpha*k=15
    assert result.telemetry.n_unionability_computations == 7
    assert unionability.n_scored == 7
    assert result.telemetry.effective_alpha == pytest.approx(7 / 5)


def test_R_subseteq_P_subseteq_D_by_table_on_feasible_run():
    ctx, _ = _make_uniform_ctx(n_tables=30)
    task = QueryTask(q_table="Q", attr=0, M={"m"}, F_star=0.0, delta=1.0, k=4, alpha=2)
    result = run_query(task, ctx)

    d_tables = {c.table for c in result.candidates}
    p_tables = {c.table for c in result.pool}
    r_tables = {c.table for c in result.selected}
    assert r_tables <= p_tables <= d_tables
    assert len(result.selected) == 4
    assert all(isinstance(c, CandidateStats) for c in result.pool)
    assert all(isinstance(c, CandidateStats) for c in result.selected)


def test_delta_constraint_holds_on_every_nonempty_result():
    """Delta(F*, F_R) <= delta on the feasible path -- the signed
    duts.stats.delta convention (overshoot admissible, only undershoot past
    delta is a violation)."""
    ctx, _ = _make_uniform_ctx(n_tables=30, n_i=10, N_i=6, q_n=10, q_N=6)
    task = QueryTask(q_table="Q", attr=0, M={"m"}, F_star=0.5, delta=0.2, k=4, alpha=2)
    result = run_query(task, ctx)

    assert result.telemetry.feasible
    assert result.telemetry.delta_R <= task.delta + 1e-9


# ---------------------------------------------------------------------------
# Infeasibility causes: insufficient_candidates, stage2_infeasible
#
# The C4 certificate was removed from this path on 2026-08-10 (user
# instruction), so "intrinsic" and "alpha_induced" no longer exist as causes
# and their two tests are gone with them. Every infeasible instance that
# clears |D| >= k is now reported as "stage2_infeasible".
# ---------------------------------------------------------------------------


def test_stage2_infeasible_cause():
    """tau unreachable by ANY k-subset of P. Scoring DID happen (the pool was
    scored before Stage 2 discovered infeasibility) -- that's still alpha*k
    calls, not 0."""
    ctx, unionability = _make_uniform_ctx(n_tables=30, n_i=10, N_i=0, q_n=10, q_N=0)
    task = QueryTask(q_table="Q", attr=0, M={"m"}, F_star=0.9, delta=0.01, k=4, alpha=2)
    result = run_query(task, ctx)

    assert result.selected == []
    assert result.telemetry.feasible is False
    assert result.telemetry.infeasibility_cause == "stage2_infeasible"
    assert result.telemetry.n_unionability_computations == 8  # pool WAS scored
    assert unionability.n_scored == 8


def test_former_alpha_induced_witness_now_reports_stage2_infeasible():
    """The exact numeric witness from
    tests/test_pipeline.py::test_infeasibility_cause_alpha_induced, replayed
    through the runner's retrieve -> Stage1 -> score -> Stage2 composition.

    Kept as a regression after the C4 certificate's removal precisely BECAUSE
    it is the interesting case: some k-subset of all of D clears tau, but
    Stage 1's pool -- optimizing the AGGREGATE ratio over alpha*k tables, a
    different combinatorial object -- doesn't happen to contain one. With no
    certificate to tell those apart, the runner must report the conservative
    "stage2_infeasible" here, and must still have scored the pool."""
    raw = [
        ("t0", 2, 3), ("t1", 2, 8), ("t2", 3, 7), ("t3", 0, 18), ("t4", 2, 2),
        ("t5", 1, 4), ("t6", 3, 15), ("t7", 9, 12), ("t8", 0, 4), ("t9", 1, 7),
        ("t10", 4, 19), ("t11", 9, 12), ("t12", 13, 17), ("t13", 0, 1),
    ]
    tables = [r[0] for r in raw]
    n_rows = {t: n for (t, _, n) in raw}
    n_rows["Q"] = 4  # n_Q=4 (N_Q=7 below -- an oracle-fixture edge case, N_Q>n_Q is
                      # arithmetically fine for this stub since n_rows and distribution
                      # are independent lookups here, not required to sum-agree)
    dist = {(t, 0): {"m": N, "x": max(n - N, 0)} for (t, N, n) in raw}
    dist[("Q", 0)] = {"m": 7}
    synopsis = _StubSynopsis(n_rows, dist)
    semantic = _StubSemantic([(t, 0, 1.0 - i * 1e-6) for i, t in enumerate(tables)])
    overlap = _StubOverlap({(t, 0): 1 for t in tables})
    unionability = _StubUnionability(value=1.0)
    ctx = RunnerContext(
        synopsis=synopsis, semantic=semantic, overlap=overlap,
        unionability=unionability, query_vectors={"Q": np.zeros((1, 4))},
    )
    task = QueryTask(
        q_table="Q", attr=0, M={"m"}, F_star=0.870758246125924,
        delta=0.07302128918681021, k=2, alpha=2, include_query=False,
    )
    result = run_query(task, ctx)

    assert result.telemetry.infeasibility_cause == "stage2_infeasible"
    assert result.telemetry.feasible is False
    assert result.selected == []
    assert len(result.pool) == 4  # alpha*k -- the pool that starved Stage 2
    assert unionability.n_scored == 4  # scoring happened; no certificate short-circuit


# retrieve_unscored_candidates: N_i is exact integer arithmetic over M (C2)
# ---------------------------------------------------------------------------


def test_retrieved_candidates_have_exact_integer_N_over_value_set():
    n_rows = {"Q": 10, "t0": 20}
    dist = {
        ("Q", 0): {"a": 3, "b": 4, "c": 3},
        ("t0", 0): {"a": 5, "b": 5, "c": 5, "d": 5},
    }
    synopsis = _StubSynopsis(n_rows, dist)
    semantic = _StubSemantic([("t0", 0, 0.9)])
    overlap = _StubOverlap({("t0", 0): 1})
    ctx = RunnerContext(
        synopsis=synopsis, semantic=semantic, overlap=overlap,
        unionability=_StubUnionability(), query_vectors={"Q": np.zeros((1, 4))},
    )
    task = QueryTask(q_table="Q", attr=0, M={"a", "c"}, F_star=0.0, delta=1.0, k=1, alpha=1)
    D, N_Q, n_Q, _telemetry = retrieve_unscored_candidates(task, ctx)

    assert n_Q == 10 and N_Q == 6  # 3 + 3, integer, over the SET {"a","c"}
    assert len(D) == 1
    assert D[0].table == "t0" and D[0].N == 10 and D[0].n == 20  # 5+5 over {"a","c"}


def test_zero_row_candidates_are_dropped():
    n_rows = {"Q": 10, "t_empty": 0, "t_ok": 5}
    dist = {("Q", 0): {"a": 5}, ("t_empty", 0): {}, ("t_ok", 0): {"a": 2}}
    synopsis = _StubSynopsis(n_rows, dist)
    semantic = _StubSemantic([("t_empty", 0, 0.9), ("t_ok", 0, 0.8)])
    overlap = _StubOverlap({("t_empty", 0): 1, ("t_ok", 0): 1})
    ctx = RunnerContext(
        synopsis=synopsis, semantic=semantic, overlap=overlap,
        unionability=_StubUnionability(), query_vectors={"Q": np.zeros((1, 4))},
    )
    task = QueryTask(q_table="Q", attr=0, M={"a"}, F_star=0.0, delta=1.0, k=1, alpha=1)
    D, _N_Q, _n_Q, _telemetry = retrieve_unscored_candidates(task, ctx)

    assert [c.table for c in D] == ["t_ok"]


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def test_load_queries_from_csv_wraps_scalar_value_as_singleton_set(tmp_path):
    query_dir = tmp_path / "query"
    query_dir.mkdir()
    (query_dir / "t1.csv").write_text("a,b\n1,2\n")
    (query_dir / "t2.csv").write_text("a,b\n1,2\n")

    csv_path = tmp_path / "protected.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_name", "protected_attribute_id", "protected_value"])
        w.writerow(["t1.csv", "2", "hello"])
        w.writerow(["t2.csv", "0", "42.0"])

    tasks, skipped = load_queries_from_csv(
        str(csv_path), str(query_dir), k=5, alpha=2, F_star=0.5, delta=0.05,
    )

    assert skipped == []
    assert len(tasks) == 2
    by_name = {t.q_table: t for t in tasks}
    assert by_name["t1.csv"].attr == 2
    assert by_name["t1.csv"].M == {"hello"}
    assert isinstance(by_name["t1.csv"].M, set)
    assert by_name["t2.csv"].attr == 0
    assert by_name["t2.csv"].M == {"42.0"}
    # shared config params applied uniformly
    assert all(t.k == 5 and t.alpha == 2 and t.F_star == 0.5 and t.delta == 0.05 for t in tasks)


def test_load_queries_from_csv_skips_rows_with_no_matching_file(tmp_path):
    """The real protected_attributes_santos.csv has rows (the "_fair" ones)
    that reference files in a DIFFERENT benchmark's query directory
    (santos3, not santos) -- this is the general case that finding exercises
    with a minimal synthetic fixture: any row whose q_name isn't an actual
    file in query_dir is skipped, not an error."""
    query_dir = tmp_path / "query"
    query_dir.mkdir()
    (query_dir / "present.csv").write_text("a\n1\n")

    csv_path = tmp_path / "protected.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_name", "protected_attribute_id", "protected_value"])
        w.writerow(["present.csv", "0", "x"])
        w.writerow(["absent_fair.csv", "0", "y"])
        w.writerow(["also_absent.csv", "1", "z"])

    tasks, skipped = load_queries_from_csv(
        str(csv_path), str(query_dir), k=1, alpha=1, F_star=0.5, delta=0.05,
    )

    assert [t.q_table for t in tasks] == ["present.csv"]
    assert set(skipped) == {"absent_fair.csv", "also_absent.csv"}


# ---------------------------------------------------------------------------
# Real-data tests: full santos benchmark run.
# ---------------------------------------------------------------------------

STARMIE_FAIR_ROOT = "/u6/bkassaie/starmie_fair"
SANTOS_ROOT = os.path.join(STARMIE_FAIR_ROOT, "data", "santos")
DATALAKE_DIR = os.path.join(SANTOS_ROOT, "datalake")
QUERY_DIR = os.path.join(SANTOS_ROOT, "query")
VECTORS_DIR = os.path.join(SANTOS_ROOT, "vectors")
DATALAKE_VEC_PKL = os.path.join(VECTORS_DIR, "cl_datalake_drop_col_tfidf_entity_column_0.pkl")
QUERY_VEC_PKL = os.path.join(VECTORS_DIR, "cl_query_drop_col_tfidf_entity_column_0.pkl")
PROTECTED_CSV = os.path.join(STARMIE_FAIR_ROOT, "data", "protected_attributes_santos.csv")

pytestmark_real = pytest.mark.skipif(
    not (os.path.isdir(DATALAKE_DIR) and os.path.isfile(DATALAKE_VEC_PKL)
         and os.path.isfile(PROTECTED_CSV)),
    reason="starmie_fair santos data not available at the expected read-only path",
)


@pytest.fixture(scope="module")
def real_datalake_files():
    return sorted(os.listdir(DATALAKE_DIR))


@pytest.fixture(scope="module")
def real_datalake_vectors():
    return load_starmie_vectors(DATALAKE_VEC_PKL)


@pytest.fixture(scope="module")
def real_query_vectors():
    return load_starmie_vectors(QUERY_VEC_PKL)


@pytest.fixture(scope="module")
def real_synopsis():
    return CsvSynopsis([DATALAKE_DIR, QUERY_DIR], theta_cat=50)


@pytest.fixture(scope="module")
def real_ctx(real_synopsis, real_datalake_files, real_datalake_vectors, real_query_vectors):
    semantic = HnswRetriever(real_datalake_vectors, sigma=0.6)
    overlap = InvertedIndexOverlap(real_synopsis, real_datalake_files)
    unionability = PinnedMatchScorer(real_query_vectors, real_datalake_vectors, threshold=0.6)
    return RunnerContext(
        synopsis=real_synopsis, semantic=semantic, overlap=overlap,
        unionability=unionability, query_vectors=real_query_vectors,
    )


@pytest.mark.skipif(
    not (os.path.isdir(DATALAKE_DIR) and os.path.isfile(DATALAKE_VEC_PKL)
         and os.path.isfile(PROTECTED_CSV)),
    reason="starmie_fair santos data not available at the expected read-only path",
)
def test_all_santos_queries_end_to_end(real_ctx):
    """Phase D's accept criterion, run against real data with real output:
    load every usable row of protected_attributes_santos.csv, run each
    QueryTask end to end, and check R subseteq P subseteq D, |R| in {0,k},
    Delta(F*,F_R) <= delta on every non-empty result, and
    n_unionability_computations == len(P) on every single query -- not just
    in aggregate.

    Only 48 of protected_attributes_santos.csv's 96 rows resolve to an
    actual santos/query/ file (see load_queries_from_csv's docstring for
    why) -- this test reports that count rather than assuming 50.
    """
    tasks, skipped = load_queries_from_csv(
        PROTECTED_CSV, QUERY_DIR, k=3, alpha=2, F_star=0.3, delta=0.15, top_n=100,
    )
    print(f"\nloaded {len(tasks)} QueryTasks from {len(tasks) + len(skipped)} CSV rows "
          f"({len(skipped)} skipped -- no matching file in {QUERY_DIR})")
    assert len(tasks) + len(skipped) == 96
    assert len(tasks) >= 40  # sanity floor; exact count (48) documented above

    causes = {}
    rows = []
    for task in tasks:
        result = run_query(task, real_ctx)
        tel = result.telemetry
        key = "feasible" if tel.feasible else tel.infeasibility_cause
        causes[key] = causes.get(key, 0) + 1
        rows.append((task.q_table, tel))

        d_tables = {c.table for c in result.candidates}
        p_tables = {c.table for c in result.pool}
        r_tables = {c.table for c in result.selected}
        assert r_tables <= p_tables <= d_tables, task.q_table
        assert len(result.selected) in (0, task.k), task.q_table
        assert tel.n_unionability_computations == len(result.pool) == len(p_tables), task.q_table
        if tel.feasible:
            assert tel.delta_R <= task.delta + 1e-9, task.q_table

    print(f"\n50-santos-query run (k=3, alpha=2, F*=0.3, delta=0.15): "
          f"{len(tasks)} queries, causes={causes}")
    for q_table, tel in rows:
        print(f"  {q_table:60s} n_D={tel.n_D:4d} n_P={tel.n_P:3d} n_R={tel.n_R:2d} "
              f"n_unionability={tel.n_unionability_computations:3d} "
              f"feasible={tel.feasible} cause={tel.infeasibility_cause}")

    assert causes.get("feasible", 0) >= 1  # at least some real queries succeed end to end


def test_n_unionability_computations_independent_of_datalake_size_real_data(
    real_synopsis, real_datalake_files, real_datalake_vectors,
):
    """The headline claim (PLAN-integration.md §3), demonstrated directly
    against real santos data rather than stubs: the SAME query, at fixed
    (k, alpha), scored against a datalake shrunk from 550 tables to 38,
    produces the SAME number of unionability computations (alpha*k),
    because that count depends only on |P| = min(alpha*k, |D|), never on
    |T|. |D| itself is allowed to differ slightly (approximate HNSW over a
    much smaller index can surface one extra/fewer neighbour) -- that's
    retrieval noise, not the claim under test.
    """
    table = "analytics-iris-externe-termes-de-recherche-20170901-20170930.csv"
    attr = 1
    M = {"16", "17", "18"}
    assert table in real_datalake_files  # guard: the fixture table must actually exist

    query_vectors = {table: real_datalake_vectors[table]}
    # alpha*k=6, comfortably below this probe's observed ~7-8 matching
    # candidates (HNSW's construction RNG can shift the exact count by 1
    # run to run -- Phase B's documented finding -- so this leaves headroom
    # rather than pinning to the single largest count ever observed).
    task = QueryTask(q_table=table, attr=attr, M=M, F_star=0.1, delta=0.9, k=2, alpha=3, top_n=200)

    # First pass over the FULL datalake to discover which tables this query
    # actually needs, so the shrunk universe below is guaranteed to still
    # contain them (rather than shrinking blind and hoping).
    semantic_full = HnswRetriever(real_datalake_vectors, sigma=0.6)
    overlap_full = InvertedIndexOverlap(real_synopsis, real_datalake_files)
    ctx_full = RunnerContext(
        synopsis=real_synopsis, semantic=semantic_full, overlap=overlap_full,
        unionability=PinnedMatchScorer(query_vectors, real_datalake_vectors, threshold=0.6),
        query_vectors=query_vectors,
    )
    r_full = run_query(task, ctx_full)
    needed = {c.table for c in r_full.candidates}
    assert len(needed) >= task.alpha * task.k  # otherwise this fixture can't demonstrate an unclamped pool

    rng = random.Random(3)
    filler = [t for t in real_datalake_files if t not in needed]
    small_subset = list(needed) + rng.sample(filler, 30)

    observed = []
    for subset, label in [(small_subset, "shrunk"), (real_datalake_files, "full")]:
        sub_vectors = {t: real_datalake_vectors[t] for t in subset}
        semantic = HnswRetriever(sub_vectors, sigma=0.6)
        overlap = InvertedIndexOverlap(real_synopsis, subset)
        unionability = PinnedMatchScorer(query_vectors, real_datalake_vectors, threshold=0.6)
        ctx = RunnerContext(
            synopsis=real_synopsis, semantic=semantic, overlap=overlap,
            unionability=unionability, query_vectors=query_vectors,
        )
        result = run_query(task, ctx)
        observed.append((label, len(subset), len(result.candidates),
                          result.telemetry.n_unionability_computations, unionability.n_scored))

    print("\n|T|-independence of n_unionability_computations (real santos data):")
    for label, n_T, n_D, n_uc, n_scored in observed:
        print(f"  {label:8s} |T|={n_T:4d}  |D|={n_D:3d}  n_unionability_computations={n_uc:2d}  "
              f"scorer.n_scored={n_scored:2d}")

    pool_sizes = [row[3] for row in observed]
    scorer_counts = [row[4] for row in observed]
    t_sizes = [row[1] for row in observed]

    assert t_sizes[0] != t_sizes[1]           # |T| really did change (30-ish vs 550)
    assert pool_sizes[0] == pool_sizes[1] == task.alpha * task.k  # scoring count did NOT
    assert scorer_counts[0] == scorer_counts[1] == task.alpha * task.k
