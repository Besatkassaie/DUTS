import random

import pytest

from duts import pipeline
from duts.types import CandidateStats, QuerySpec
from tests.generators import (knapsack_to_tables, random_candidates,
                               tau_from_knapsack_capacity)


def make_query(N_Q=0, n_Q=10, k=5, alpha=2, F_star=0.5, delta=0.1, include_query=True):
    return QuerySpec(N_Q=N_Q, n_Q=n_Q, k=k, alpha=alpha, F_star=F_star, delta=delta,
                      include_query=include_query)


@pytest.mark.parametrize("seed", range(300))
def test_pipeline_properties_over_random_instances(seed):
    """PLAN.md Phase 4 accept: Delta(F*,F_R) <= delta on every non-empty
    result; |R| = k or empty; R subseteq P subseteq D. Exercised with
    `certify` both on and off, since it changes the pipeline's shape (an
    extra Dinkelbach run before Stage 1) but must never change these
    invariants."""
    rng = random.Random(seed)
    m = rng.randint(1, 40)
    D = random_candidates(m, rng, n_range=(1, 30), force_ties=(seed % 3 == 0),
                           force_boundary=(seed % 4 == 0))
    k = rng.randint(1, 10)
    alpha = rng.choice([1, 1.5, 2, 3, 5])
    F_star = rng.uniform(0, 1)
    delta = rng.uniform(0, 0.5)
    N_Q, n_Q = rng.randint(0, 20), rng.randint(1, 20)
    include_query = rng.random() < 0.5
    certify = seed % 2 == 0

    query = QuerySpec(N_Q, n_Q, k, alpha, F_star, delta, include_query)
    result = pipeline.run(D, query, certify=certify)

    D_nonempty_ids = {c.table for c in D if c.n > 0}
    P_ids = {c.table for c in result.pool}
    R_ids = {c.table for c in result.selected}

    assert P_ids <= D_nonempty_ids  # P subseteq D
    assert R_ids <= P_ids           # R subseteq P

    if result.telemetry.feasible:
        assert len(result.selected) == k
        assert result.telemetry.delta_R is not None
        assert result.telemetry.delta_R <= delta + 1e-9  # Delta(F*,F_R) <= delta
    else:
        assert result.selected == []
        if certify:
            assert result.telemetry.infeasibility_cause in (
                "insufficient_candidates", "intrinsic", "alpha_induced",
            )
        else:
            assert result.telemetry.infeasibility_cause in (
                "insufficient_candidates", "stage2_infeasible",
            )


def test_infeasibility_cause_insufficient_candidates():
    D = [CandidateStats("a", 5, 10, 0.0)]
    query = make_query(k=5)
    result = pipeline.run(D, query)
    assert result.telemetry.infeasibility_cause == "insufficient_candidates"
    assert result.selected == [] and result.pool == []


def test_infeasibility_cause_intrinsic():
    """The whole lake can't reach tau even using every table (C4): certify
    fails, so the ILP is never invoked (dinkelbach_iterations_stage1 stays 0
    and no LP/ILP telemetry is populated)."""
    D = [CandidateStats(f"t{i}", N=0, n=10, U=0.5) for i in range(10)]  # every F_i = 0
    query = make_query(N_Q=0, n_Q=0, k=3, alpha=2, F_star=0.9, delta=0.0, include_query=False)
    result = pipeline.run(D, query, certify=True)
    assert result.telemetry.infeasibility_cause == "intrinsic"
    assert result.telemetry.F_max_k == pytest.approx(0.0)
    assert result.selected == [] and result.pool == []
    assert result.telemetry.lp_precheck_ran is False


def test_infeasibility_cause_alpha_induced():
    """Certificate says the lake CAN reach tau at cardinality k (some k-subset
    of ALL of D achieves F >= tau), but Stage 1's pool -- which maximizes the
    AGGREGATE ratio over alpha*k tables, a different combinatorial choice from
    'the k tables that individually form the best k-subset' -- doesn't happen
    to contain a k-subset that also clears tau. Must be reported as
    alpha_induced, not intrinsic. (alpha=1 can never produce this: then
    Stage 1's solve(D, k, ...) call is *identical* to certify's, so the pool
    IS exactly the certifying subset. This needed alpha=2 and was found by
    random search -- PLAN.md flags this interaction as the key diagnostic for
    tuning alpha, not something obvious to hand-construct.)"""
    D = [
        CandidateStats("t0", N=2, n=3, U=0.6100139536102268),
        CandidateStats("t1", N=2, n=8, U=0.5411656900574272),
        CandidateStats("t2", N=3, n=7, U=0.25264568028365286),
        CandidateStats("t3", N=0, n=18, U=0.6199092735430697),
        CandidateStats("t4", N=2, n=2, U=0.5485023660870102),
        CandidateStats("t5", N=1, n=4, U=0.15745892702159203),
        CandidateStats("t6", N=3, n=15, U=0.42564990019801463),
        CandidateStats("t7", N=9, n=12, U=0.3779088850886737),
        CandidateStats("t8", N=0, n=4, U=0.12012027515997548),
        CandidateStats("t9", N=1, n=7, U=0.417770032346621),
        CandidateStats("t10", N=4, n=19, U=0.8856899728058276),
        CandidateStats("t11", N=9, n=12, U=0.43406071292605697),
        CandidateStats("t12", N=13, n=17, U=0.016633809024522273),
        CandidateStats("t13", N=0, n=1, U=0.6554639742216479),
    ]
    query = QuerySpec(N_Q=7, n_Q=4, k=2, alpha=2, F_star=0.870758246125924,
                       delta=0.07302128918681021, include_query=False)
    result = pipeline.run(D, query, certify=True)

    assert result.telemetry.F_max_k == pytest.approx(0.8)  # certify: feasible over all of D
    assert result.telemetry.F_max_k >= query.F_star - query.delta
    assert result.telemetry.infeasibility_cause == "alpha_induced"
    assert result.telemetry.feasible is False
    assert result.selected == []
    assert len(result.pool) == 4  # alpha*k = 4 -- the pool that starved Stage 2


def test_certify_defaults_to_false():
    """certify=False is the default (for now): no certificate run, no
    certify-related telemetry populated, and the same instance that reports
    'alpha_induced' under certify=True reports 'stage2_infeasible' under the
    default -- since without the certificate we can't tell alpha_induced from
    intrinsic, only that Stage 2 itself failed."""
    D = [
        CandidateStats("t0", N=2, n=3, U=0.6100139536102268),
        CandidateStats("t1", N=2, n=8, U=0.5411656900574272),
        CandidateStats("t2", N=3, n=7, U=0.25264568028365286),
        CandidateStats("t3", N=0, n=18, U=0.6199092735430697),
        CandidateStats("t4", N=2, n=2, U=0.5485023660870102),
        CandidateStats("t5", N=1, n=4, U=0.15745892702159203),
        CandidateStats("t6", N=3, n=15, U=0.42564990019801463),
        CandidateStats("t7", N=9, n=12, U=0.3779088850886737),
        CandidateStats("t8", N=0, n=4, U=0.12012027515997548),
        CandidateStats("t9", N=1, n=7, U=0.417770032346621),
        CandidateStats("t10", N=4, n=19, U=0.8856899728058276),
        CandidateStats("t11", N=9, n=12, U=0.43406071292605697),
        CandidateStats("t12", N=13, n=17, U=0.016633809024522273),
        CandidateStats("t13", N=0, n=1, U=0.6554639742216479),
    ]
    query = QuerySpec(N_Q=7, n_Q=4, k=2, alpha=2, F_star=0.870758246125924,
                       delta=0.07302128918681021, include_query=False)

    result = pipeline.run(D, query)  # certify not passed -> defaults to False
    assert result.telemetry.infeasibility_cause == "stage2_infeasible"
    assert result.telemetry.F_max_k is None
    assert result.telemetry.dinkelbach_iterations_certify == 0
    assert result.telemetry.certify_time_s is None

    explicit = pipeline.run(D, query, certify=False)
    assert explicit.telemetry.infeasibility_cause == "stage2_infeasible"


def test_clamped_pool_when_D_smaller_than_alpha_k():
    """C5: k <= |D| < alpha*k -> P clamped to D, effective_alpha = |D|/k logged,
    never padded."""
    D = [CandidateStats(f"t{i}", N=10, n=10, U=1.0) for i in range(4)]
    query = make_query(N_Q=0, n_Q=0, k=3, alpha=5, F_star=0.5, delta=0.5, include_query=False)
    result = pipeline.run(D, query)
    assert len(result.pool) == 4  # clamped to |D|, not padded to alpha*k=15
    assert result.telemetry.effective_alpha == pytest.approx(4 / 3)


def test_alpha_sweep_quality_cost_tradeoff():
    """PLAN.md Phase 4 accept: alpha sweep alpha in {1,2,3,5,10} x k in
    {5,10,20} on synthetic D, showing the quality/cost trade-off. (The C3
    max_f vs satisfice ablation is excluded: 'satisfice' has no defined
    objective yet -- see PLAN.md C3 open question -- so this sweep covers
    max_f only, which is all that's implemented.)
    """
    rng = random.Random(42)
    D = random_candidates(200, rng, n_range=(5, 50))
    N_Q, n_Q = 5, 50

    rows = []
    for k in (5, 10, 20):
        for alpha in (1, 2, 3, 5, 10):
            query = QuerySpec(N_Q, n_Q, k, alpha, F_star=0.5, delta=0.1, include_query=True)
            result = pipeline.run(D, query)
            rows.append((k, alpha, result.telemetry.feasible,
                         result.telemetry.sum_U, result.telemetry.n_P))
            # regardless of outcome, the core invariants must hold
            if result.telemetry.feasible:
                assert result.telemetry.delta_R <= 0.1 + 1e-9
                assert len(result.selected) == k
            assert result.telemetry.n_P <= int(alpha * k) or result.telemetry.n_P == len(D)

    # sanity: at least some (k, alpha) combinations were feasible, and pool
    # size is non-decreasing in alpha for fixed k (the actual "cost" axis
    # the trade-off is about)
    assert any(r[2] for r in rows)
    for k in (5, 10, 20):
        pool_sizes = [r[4] for r in sorted([row for row in rows if row[0] == k], key=lambda r: r[1])]
        assert pool_sizes == sorted(pool_sizes)


@pytest.mark.parametrize("seed", range(40))
def test_ekkp_reduction_matches_knapsack_optimum(seed):
    """Theorem 3.2's E-kKP reduction as an instance generator (PLAN.md §6):
    build tables from a knapsack instance, solve with Stage 2 directly
    (Q-free, per the corrected tau), and check the objective matches the
    true knapsack optimum found by brute force. Validates the reduction
    itself, not just the solver."""
    import itertools

    from duts.stage2_ilp import solve as stage2_solve

    rng = random.Random(seed)
    n_items = rng.randint(3, 10)
    c = rng.randint(10, 30)
    profits = [rng.uniform(1, 20) for _ in range(n_items)]
    weights = [rng.randint(1, c) for _ in range(n_items)]
    k = rng.randint(1, n_items)

    # true knapsack optimum: exactly k items, total weight <= c, max profit
    best_profit = None
    for combo in itertools.combinations(range(n_items), k):
        total_w = sum(weights[i] for i in combo)
        if total_w <= c:
            p = sum(profits[i] for i in combo)
            if best_profit is None or p > best_profit:
                best_profit = p
    if best_profit is None:
        pytest.skip("no feasible knapsack solution at this cardinality")

    tables = knapsack_to_tables(profits, weights, c)
    tau = tau_from_knapsack_capacity(k, c, capacity=c)

    result = stage2_solve(tables, k, tau, N_Q=0, n_Q=0, include_query=False)
    assert result.feasible is True
    assert result.objective_value == pytest.approx(best_profit, abs=1e-6)
