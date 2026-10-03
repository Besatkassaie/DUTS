"""Tests for ``experiments/wdc/union_scalability_study.py``'s ``_score`` --
specifically the synthetic-id logic added for the ``dedupe_by_table=False``
stress test (multiple columns of the same physical table must not collapse
to one ``CandidateStats`` under Stage 1's table-keyed identity/tie-break).

The driver loop itself (``run_tier_study_union``) has no dedicated test,
matching the existing project convention for this class of script
(``santoslarge_study.py``/``scalability_study.py`` have none either --
correctness rests on the already-tested building blocks it composes).
"""
from experiments.wdc.union_scalability_study import _score


class _StubUnionability(object):
    def score(self, q_table, c_table, pin):
        return 1.0


class _Cand(object):
    def __init__(self, table, N, n, attr):
        self.table = table
        self.N = N
        self.n = n
        self.attr = attr


class _Ctx(object):
    def __init__(self):
        self.unionability = _StubUnionability()


def test_dedupe_by_table_true_keeps_real_table_name():
    cands = [_Cand("t1", 3, 10, 0)]
    out = _score(cands, _Ctx(), "q", 0, dedupe_by_table=True)
    assert out[0].table == "t1"


def test_dedupe_by_table_false_synthesizes_unique_ids_per_column():
    cands = [_Cand("t1", 3, 10, 0), _Cand("t1", 5, 10, 1), _Cand("t1", 1, 10, 2)]
    out = _score(cands, _Ctx(), "q", 0, dedupe_by_table=False)
    ids = [c.table for c in out]
    assert len(set(ids)) == 3, "each column must get a distinct candidate id"
    assert ids == ["t1::0", "t1::1", "t1::2"]
    # N/n/U carried through unchanged, only the id field changed
    assert [c.N for c in out] == [3, 5, 1]
    assert all(c.n == 10 for c in out)
