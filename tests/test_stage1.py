import random

import pytest

from duts.stage1_dinkelbach import certify_feasible, iterate, solve
from duts.types import CandidateStats
from tests.generators import random_candidates
from tests.oracle import brute_force_stage1


@pytest.mark.parametrize("seed", range(500))
def test_stage1_exact_vs_brute_force(seed):
    """PLAN.md Phase 2 accept: exact vs brute force over all C(m, size) for
    m <= 16, >= 500 random instances including ties, equal n_i, N_i=0,
    N_i=n_i."""
    rng = random.Random(seed)
    m = rng.randint(2, 16)
    candidates = random_candidates(
        m, rng, n_range=(1, 20),
        force_ties=(seed % 3 == 0),
        force_boundary=(seed % 4 == 0),
    )
    candidates = [c for c in candidates if c.n > 0]
    if not candidates:
        pytest.skip("degenerate: all candidates had n_i=0")
    size = rng.randint(1, len(candidates))

    N_Q = rng.randint(0, 20)
    n_Q = rng.randint(1, 20)
    include_query = rng.random() < 0.5

    result = solve(candidates, size, N_Q, n_Q, include_query)
    _, brute_F = brute_force_stage1(candidates, size, N_Q, n_Q, include_query)

    assert result.objective_value == pytest.approx(brute_F, abs=1e-9)
    assert len(result.selected) == size
    assert result.iterations <= 100
    assert result.iterations >= 1


@pytest.mark.parametrize("seed", range(200))
def test_certify_feasible_matches_brute_force(seed):
    """PLAN.md Phase 2 accept: certify_feasible agrees with brute-force
    max_{|S|=k, S subseteq D} F on every instance."""
    rng = random.Random(1000 + seed)
    m = rng.randint(2, 14)
    candidates = random_candidates(m, rng, n_range=(1, 15))
    candidates = [c for c in candidates if c.n > 0]
    if not candidates:
        pytest.skip("degenerate")
    k = rng.randint(1, len(candidates))
    N_Q, n_Q = rng.randint(0, 15), rng.randint(1, 15)
    include_query = rng.random() < 0.5
    tau = rng.uniform(0, 1)

    feasible, f_max_k, _ = certify_feasible(candidates, k, tau, N_Q, n_Q, include_query)
    _, brute_f_max_k = brute_force_stage1(candidates, k, N_Q, n_Q, include_query)

    assert f_max_k == pytest.approx(brute_f_max_k, abs=1e-9)
    assert feasible == (brute_f_max_k >= tau)


def test_certify_feasible_insufficient_candidates():
    candidates = [CandidateStats("a", 1, 1, 0.0)]
    feasible, f_max_k, n_iter = certify_feasible(candidates, k=5, tau=0.1)
    assert feasible is False
    assert f_max_k == float("-inf")
    assert n_iter == 0


def test_lambda_monotone_from_iteration_1():
    """C5: monotonicity is only guaranteed from iteration 1 onward with the
    real-subset seeding -- see test_c4_counterexample_regression for why the
    full-pool-ratio seed cannot be used as the baseline under include_query=True."""
    rng = random.Random(0)
    for _ in range(50):
        m = rng.randint(2, 12)
        candidates = random_candidates(m, rng, n_range=(1, 20))
        candidates = [c for c in candidates if c.n > 0]
        if not candidates:
            continue
        size = rng.randint(1, len(candidates))
        N_Q, n_Q = rng.randint(0, 20), rng.randint(1, 20)

        trace = [lam for _, lam, _ in iterate(candidates, size, N_Q, n_Q, True)]
        for a, b in zip(trace, trace[1:]):
            assert b >= a - 1e-9, f"lambda decreased after iteration 1: {trace}"


def test_deterministic_tie_breaking():
    """Same input, called twice, must return the identical selection -- C5's
    (-n_i, table_id) tie rule is what makes the set-equality termination test
    meaningful in the first place."""
    candidates = [
        CandidateStats("b", N=5, n=10, U=0.0),
        CandidateStats("a", N=5, n=10, U=0.0),
        CandidateStats("c", N=3, n=6, U=0.0),
    ]
    r1 = solve(candidates, 2, 0, 1, True)
    r2 = solve(candidates, 2, 0, 1, True)
    assert [c.table for c in r1.selected] == [c.table for c in r2.selected]


def test_raises_on_insufficient_candidates():
    candidates = [CandidateStats("a", 1, 5, 0.0)]
    with pytest.raises(ValueError):
        solve(candidates, size=3)


def test_raises_when_exceeding_iteration_cap():
    rng = random.Random(7)
    candidates = random_candidates(10, rng, n_range=(1, 20))
    candidates = [c for c in candidates if c.n > 0]
    with pytest.raises(RuntimeError):
        solve(candidates, 3, 0, 5, True, max_iterations=0)


def test_c4_counterexample_regression():
    """PLAN.md C4: the RETIRED monotonicity claim
        max_{|S|=k} F >= max_{|S|=alpha*k} F
    is FALSE under include_query=True. This is a permanent regression fixture:
    if it ever starts failing because someone "fixed" the solver to make it
    pass, that fix is wrong -- the claim itself does not hold under C1. The
    certificate (certify_feasible) never relied on this claim; it is sound
    directly from Problem (1)'s |R|=k constraint, and that soundness is what
    the second half of this test checks.
    """
    t1 = CandidateStats("t1", N=5, n=10, U=0.0)
    t2 = CandidateStats("t2", N=4, n=10, U=0.0)
    candidates = [t1, t2]
    N_Q, n_Q = 0, 1000

    _, f_k_true = brute_force_stage1(candidates, 1, N_Q, n_Q, include_query=True)
    _, f_ak_true = brute_force_stage1(candidates, 2, N_Q, n_Q, include_query=True)
    assert f_k_true < f_ak_true  # the claim FAILS under Q -- expected and correct

    _, f_k_false = brute_force_stage1(candidates, 1, N_Q, n_Q, include_query=False)
    _, f_ak_false = brute_force_stage1(candidates, 2, N_Q, n_Q, include_query=False)
    assert f_k_false >= f_ak_false  # Q-free: the claim holds

    # the certificate is sound regardless -- it only needs |R|=k, R subseteq D
    feasible, f_max_k, _ = certify_feasible(
        candidates, k=1, tau=0.003, N_Q=N_Q, n_Q=n_Q, include_query=True
    )
    assert f_max_k == pytest.approx(f_k_true)
    assert feasible is True


def test_lambda_seed_overshoots_under_include_query():
    """Companion to the C4 fixture: demonstrates the lambda^(0) overshoot
    directly (PLAN.md C5). 100 identical tables against a large, zero-match
    query: the full-pool ratio (0.25) is above lambda* (~0.0098), so a solver
    seeded from it would need its first step to DECREASE."""
    candidates = [CandidateStats(f"t{i}", N=5, n=10, U=0.0) for i in range(100)]
    N_Q, n_Q = 0, 1000
    alpha_k = 2

    full_pool_ratio = (N_Q + sum(c.N for c in candidates)) / (n_Q + sum(c.n for c in candidates))
    result = solve(candidates, alpha_k, N_Q, n_Q, include_query=True)

    assert full_pool_ratio > result.objective_value  # the overshoot PLAN.md documents
    _, brute_F = brute_force_stage1(candidates, alpha_k, N_Q, n_Q, include_query=True)
    assert result.objective_value == pytest.approx(brute_F, abs=1e-9)  # solver is still exact
