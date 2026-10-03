import pytest

from duts.stats import (F, F_of_set, N_of, aggregate, delta,
                         drop_empty, weighted_average_identity)
from duts.types import CandidateStats


def test_N_of_sums_over_value_set():
    counts = {"a": 3, "b": 5, "c": 2}
    assert N_of(counts, {"a", "c"}) == 5
    assert N_of(counts, {"a", "b", "c"}) == 10
    assert N_of(counts, {"missing"}) == 0


def test_F_basic():
    assert F(5, 10) == 0.5


def test_F_zero_denominator_raises():
    with pytest.raises(ZeroDivisionError):
        F(0, 0)


def test_drop_empty():
    cs = [CandidateStats("a", 1, 0, 0.0), CandidateStats("b", 2, 5, 0.0)]
    assert drop_empty(cs) == [CandidateStats("b", 2, 5, 0.0)]


def test_aggregate_integer_arithmetic():
    stats = [CandidateStats("t1", N=3, n=10, U=0.1), CandidateStats("t2", N=7, n=20, U=0.2)]
    N_S, n_S = aggregate(stats, N_Q=1, n_Q=5, include_query=True)
    assert (N_S, n_S) == (11, 35)
    assert isinstance(N_S, int) and isinstance(n_S, int)


def test_aggregate_query_excluded_when_include_query_false():
    stats = [CandidateStats("t1", N=3, n=10, U=0.1)]
    N_S, n_S = aggregate(stats, N_Q=100, n_Q=100, include_query=False)
    assert (N_S, n_S) == (3, 10)


def test_delta_signed_overshoot_admissible():
    # F* = 0.5 but achieved F = 0.7 -- overshoot -> delta is negative
    assert delta(0.5, 0.7) == pytest.approx(-0.2)


def test_weighted_average_identity_exact_q_free():
    """§5's identity: F_S = sum_i w_i F_i, w_i = n_i / sum(n_j) -- Q-free."""
    stats = [
        CandidateStats("t1", N=5, n=10, U=0.0),   # F_1 = 0.5
        CandidateStats("t2", N=2, n=20, U=0.0),   # F_2 = 0.1
        CandidateStats("t3", N=9, n=30, U=0.0),   # F_3 = 0.3
    ]
    direct = F_of_set(stats, include_query=False)
    weighted = weighted_average_identity(stats)
    assert direct == pytest.approx(weighted)

    n_total = 60
    expected = (10 / n_total) * 0.5 + (20 / n_total) * 0.1 + (30 / n_total) * 0.3
    assert direct == pytest.approx(expected)


def test_weighted_average_identity_many_random_instances():
    import random
    rng = random.Random(0)
    for _ in range(200):
        m = rng.randint(1, 12)
        stats = []
        for i in range(m):
            n_i = rng.randint(1, 50)
            N_i = rng.randint(0, n_i)
            stats.append(CandidateStats(f"t{i}", N_i, n_i, rng.random()))
        assert F_of_set(stats, include_query=False) == pytest.approx(
            weighted_average_identity(stats)
        )


def test_aggregation_sensitivity_witness():
    """Two instances with IDENTICAL per-table {F_i} but DIFFERENT F_S
    (PLAN.md Phase 1 accept criterion; §3's aggregation-sensitivity property).
    """
    a = [CandidateStats("small", N=9, n=10, U=0.0), CandidateStats("large", N=10, n=100, U=0.0)]
    b = [CandidateStats("small", N=90, n=100, U=0.0), CandidateStats("large", N=1, n=10, U=0.0)]

    Fi_a = sorted(round(c.N / c.n, 9) for c in a)
    Fi_b = sorted(round(c.N / c.n, 9) for c in b)
    assert Fi_a == Fi_b  # identical per-table F_i sets: {0.1, 0.9} in both

    F_a = F_of_set(a, include_query=False)
    F_b = F_of_set(b, include_query=False)
    assert F_a != pytest.approx(F_b)  # yet the aggregate proportions differ
