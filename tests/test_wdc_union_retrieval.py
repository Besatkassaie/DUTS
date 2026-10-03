"""Tests for ``experiments/wdc/union_retrieval.py`` -- the union-mode retrieval
variant used only by the WDC union-pool scalability experiment (see that
module's docstring for why it exists and why it isn't in ``dutsx/``).

Stub-adapter only, no real data dependency -- same convention as
``tests/test_dutsx_runner.py``.
"""
import numpy as np

from dutsx.retrieval import retrieve_candidates
from dutsx.runner import QueryTask, RunnerContext
from experiments.wdc.union_retrieval import (
    retrieve_union_candidates,
    retrieve_unscored_candidates_union,
    retrieve_unscored_candidates_union_and_intersection,
)


class _StubSynopsis(object):
    def __init__(self, n_rows, distributions):
        self._n_rows = n_rows
        self._dist = distributions

    def n_rows(self, table):
        return self._n_rows.get(table, 0)

    def distribution(self, table, attr):
        return self._dist.get((table, attr), {})

    def categorical_attrs(self, table):
        return [0]


class _StubSemantic(object):
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
    def score(self, q_table, c_table, pin):
        return 1.0


# ---------------------------------------------------------------------------
# retrieve_union_candidates -- pure pair-combination logic
# ---------------------------------------------------------------------------

def test_union_includes_a_pair_present_in_only_one_signal():
    semantic = _StubSemantic([("t_sem_only", 0, 0.9)])
    overlap = _StubOverlap({("t_ovl_only", 0): 3})
    d, telem = retrieve_union_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)
    tables = {c.table for c in d}
    assert tables == {"t_sem_only", "t_ovl_only"}
    assert telem.n_sem == 1
    assert telem.n_ovl == 1
    assert telem.n_pair == 2
    assert telem.n_d == 2


def test_intersection_would_have_dropped_both_semantic_only_and_overlap_only_pairs():
    """Same fixture as above, run through the real intersection function --
    confirms the union variant is a genuine relaxation, not a no-op."""
    from dutsx.retrieval import retrieve_candidates
    semantic = _StubSemantic([("t_sem_only", 0, 0.9)])
    overlap = _StubOverlap({("t_ovl_only", 0): 3})
    d, telem = retrieve_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)
    assert d == []
    assert telem.n_pair == 0


def test_semantic_only_pair_gets_overlap_zero_placeholder():
    semantic = _StubSemantic([("t1", 0, 0.5)])
    overlap = _StubOverlap({})
    d, _ = retrieve_union_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)
    assert len(d) == 1
    assert d[0].table == "t1"
    assert d[0].sim == 0.5
    assert d[0].overlap == 0


def test_overlap_only_pair_gets_sim_zero_placeholder():
    semantic = _StubSemantic([])
    overlap = _StubOverlap({("t1", 0): 7})
    d, _ = retrieve_union_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)
    assert len(d) == 1
    assert d[0].table == "t1"
    assert d[0].sim == 0.0
    assert d[0].overlap == 7


def test_per_table_argmax_prefers_real_similarity_over_overlap_only_placeholder():
    """Table t1 clears BOTH signals via different attributes (attr 0 semantic,
    attr 1 overlap-only). The per-table reduction must keep the real-sim pair
    (attr 0), not the sim=0.0 placeholder (attr 1)."""
    semantic = _StubSemantic([("t1", 0, 0.7)])
    overlap = _StubOverlap({("t1", 1): 2})
    d, telem = retrieve_union_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)
    assert telem.n_pair == 2   # both pairs survive the union...
    assert len(d) == 1          # ...but only one table-level candidate remains
    assert d[0].attr == 0
    assert d[0].sim == 0.7


def test_union_pair_count_is_never_smaller_than_intersection():
    from dutsx.retrieval import retrieve_candidates
    semantic = _StubSemantic([("t1", 0, 0.9), ("t2", 0, 0.5), ("t_sem_only", 0, 0.1)])
    overlap = _StubOverlap({("t1", 0): 5, ("t2", 0): 1, ("t_ovl_only", 0): 9})
    d_union, telem_union = retrieve_union_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)
    d_inter, telem_inter = retrieve_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)
    assert telem_union.n_pair >= telem_inter.n_pair
    assert len(d_union) >= len(d_inter)
    assert {c.table for c in d_inter} <= {c.table for c in d_union}


# ---------------------------------------------------------------------------
# retrieve_unscored_candidates_union -- N_i/n_i lookup + C5 drop
# ---------------------------------------------------------------------------

def _make_ctx(n_rows, distributions, sem_results, ovl_pairs):
    synopsis = _StubSynopsis(n_rows, distributions)
    semantic = _StubSemantic(sem_results)
    overlap = _StubOverlap(ovl_pairs)
    q_vecs = {"q": np.zeros((1, 4))}
    return RunnerContext(
        synopsis=synopsis, semantic=semantic, overlap=overlap,
        unionability=_StubUnionability(), query_vectors=q_vecs,
    )


def test_n_i_and_N_i_computed_correctly_for_union_candidates():
    ctx = _make_ctx(
        n_rows={"q": 10, "t1": 20, "t2": 30},
        distributions={
            ("q", 0): {"v": 4},
            ("t1", 0): {"v": 6, "w": 14},
            ("t2", 1): {"v": 9},
        },
        sem_results=[("t1", 0, 0.8)],
        ovl_pairs={("t2", 1): 1},
    )
    task = QueryTask(q_table="q", attr=0, M={"v"}, F_star=0.3, delta=0.15, k=5, alpha=2.0, top_n=10)
    D, N_Q, n_Q, telem = retrieve_unscored_candidates_union(task, ctx)
    assert N_Q == 4 and n_Q == 10
    by_table = {c.table: c for c in D}
    assert by_table["t1"].N == 6 and by_table["t1"].n == 20 and by_table["t1"].attr == 0
    assert by_table["t2"].N == 9 and by_table["t2"].n == 30 and by_table["t2"].attr == 1
    assert telem.n_d == 2


def test_n_i_zero_candidate_is_dropped():
    ctx = _make_ctx(
        n_rows={"q": 10, "t_empty": 0, "t_ok": 5},
        distributions={("q", 0): {"v": 1}, ("t_ok", 0): {"v": 2}},
        sem_results=[("t_empty", 0, 0.9), ("t_ok", 0, 0.5)],
        ovl_pairs={},
    )
    task = QueryTask(q_table="q", attr=0, M={"v"}, F_star=0.3, delta=0.15, k=5, alpha=2.0, top_n=10)
    D, _, _, _ = retrieve_unscored_candidates_union(task, ctx)
    tables = {c.table for c in D}
    assert tables == {"t_ok"}


# ---------------------------------------------------------------------------
# retrieve_unscored_candidates_union_and_intersection -- the combined,
# single-probe function that replaced calling the union and intersection
# paths separately (a real measurement bug: the original two-call version
# probed semantic/overlap TWICE and reported their summed cost as "the union
# path's retrieval time").
# ---------------------------------------------------------------------------

class _CountingSemantic(object):
    def __init__(self, results):
        self._results = results
        self.n_calls = 0

    def query(self, vec, top_n):
        self.n_calls += 1
        return self._results[:top_n]


class _CountingOverlap(object):
    def __init__(self, pairs):
        self._pairs = pairs
        self.n_calls = 0

    def query(self, M):
        self.n_calls += 1
        return dict(self._pairs)


def test_combined_function_probes_semantic_and_overlap_exactly_once():
    """The regression this test guards: an earlier version called two
    separate retrieval functions back-to-back, each doing its own probe --
    semantic/overlap would have been queried twice per QueryTask."""
    semantic = _CountingSemantic([("t1", 0, 0.9), ("t2", 0, 0.4)])
    overlap = _CountingOverlap({("t2", 0): 3, ("t3", 0): 1})
    ctx = RunnerContext(
        synopsis=_StubSynopsis(
            n_rows={"q": 10, "t1": 5, "t2": 5, "t3": 5},
            distributions={("q", 0): {"v": 1}},
        ),
        semantic=semantic, overlap=overlap, unionability=_StubUnionability(),
        query_vectors={"q": np.zeros((1, 4))},
    )
    task = QueryTask(q_table="q", attr=0, M={"v"}, F_star=0.3, delta=0.15, k=5, alpha=2.0, top_n=10)
    retrieve_unscored_candidates_union_and_intersection(task, ctx)
    assert semantic.n_calls == 1
    assert overlap.n_calls == 1


def test_combined_function_matches_separate_union_and_intersection_functions():
    """D_union/D_intersection from the combined, single-probe function must
    be IDENTICAL (as sets of tables, and per-table N/n/attr) to what the
    separate union function and the real dutsx intersection function
    produce independently -- the fix must not change any answer, only how
    many times the expensive calls run."""
    ctx = _make_ctx(
        n_rows={"q": 10, "t1": 20, "t2": 30, "t3": 15},
        distributions={
            ("q", 0): {"v": 4},
            ("t1", 0): {"v": 6},
            ("t2", 0): {"v": 9},
            ("t3", 1): {"v": 2},
        },
        sem_results=[("t1", 0, 0.9), ("t2", 0, 0.4)],
        ovl_pairs={("t2", 0): 3, ("t3", 1): 1},
    )
    task = QueryTask(q_table="q", attr=0, M={"v"}, F_star=0.3, delta=0.15, k=5, alpha=2.0, top_n=10)

    D_union_ref, _, _, _ = retrieve_unscored_candidates_union(task, ctx)
    D_inter_ref, _, _, _ = _unscored_from_retrieve_candidates(task, ctx)

    D_union, D_inter, N_Q, n_Q, telem_u, telem_i, timing = \
        retrieve_unscored_candidates_union_and_intersection(task, ctx)

    assert sorted(D_union) == sorted(D_union_ref)
    assert sorted(D_inter) == sorted(D_inter_ref)
    assert telem_u.n_d == len(D_union)
    assert telem_i.n_d == len(D_inter)
    # timing: all phases non-negative, and the probe genuinely ran (>= 0 is
    # the only safe assertion on wall-clock time, but it must be present).
    assert timing.probe_time_s >= 0
    assert timing.union_combine_time_s >= 0
    assert timing.intersection_combine_time_s >= 0
    assert timing.union_lookup_time_s >= 0
    assert timing.intersection_lookup_time_s >= 0


def test_dedupe_by_table_false_keeps_every_column_of_the_same_table():
    """Stress-test mode, requested explicitly for the WDC scalability
    experiment: skip the per-table argmax reduction entirely so a table
    matched on multiple columns contributes one row PER column, not one."""
    ctx = _make_ctx(
        n_rows={"q": 10, "t1": 20},
        distributions={
            ("q", 0): {"v": 4},
            ("t1", 0): {"v": 6},
            ("t1", 1): {"v": 8},
            ("t1", 2): {"v": 2},
        },
        sem_results=[("t1", 0, 0.9), ("t1", 1, 0.4)],
        ovl_pairs={("t1", 2): 5},
    )
    task = QueryTask(q_table="q", attr=0, M={"v"}, F_star=0.3, delta=0.15, k=5, alpha=2.0, top_n=10)

    D_union, D_inter, N_Q, n_Q, telem_u, telem_i, timing = \
        retrieve_unscored_candidates_union_and_intersection(task, ctx, dedupe_by_table=False)

    # all three columns of t1 survive as separate candidates
    assert len(D_union) == 3
    attrs = sorted(c.attr for c in D_union)
    assert attrs == [0, 1, 2]
    assert all(c.table == "t1" for c in D_union)

    # deduped (default) mode collapses the same fixture to one row
    D_union_deduped, _, _, _, _, _, _ = \
        retrieve_unscored_candidates_union_and_intersection(task, ctx, dedupe_by_table=True)
    assert len(D_union_deduped) == 1


def _unscored_from_retrieve_candidates(task, ctx):
    """Mirrors dutsx.runner.retrieve_unscored_candidates's N_i/n_i lookup,
    but driven off dutsx.retrieval.retrieve_candidates directly -- avoids
    importing the runner's own function just to build a reference value."""
    from duts.stats import N_of
    from dutsx.runner import UnscoredCandidate
    q_vecs = ctx.query_vectors[task.q_table]
    query_vec = q_vecs[task.attr]
    retrieved, telem = retrieve_candidates(ctx.semantic, ctx.overlap, query_vec, task.M, task.top_n)
    n_Q = ctx.synopsis.n_rows(task.q_table)
    N_Q = N_of(ctx.synopsis.distribution(task.q_table, task.attr), task.M)
    D = []
    for rc in retrieved:
        n_i = ctx.synopsis.n_rows(rc.table)
        if n_i == 0:
            continue
        N_i = N_of(ctx.synopsis.distribution(rc.table, rc.attr), task.M)
        D.append(UnscoredCandidate(table=rc.table, N=N_i, n=n_i, attr=rc.attr))
    return D, N_Q, n_Q, telem
