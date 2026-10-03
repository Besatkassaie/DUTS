import random

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from duts.stage2_ilp import _y_tau, solve
from duts.types import CandidateStats
from tests.generators import random_candidates
from tests.oracle import brute_force_stage2


@pytest.mark.parametrize("seed", range(400))
def test_stage2_exact_vs_brute_force(seed):
    """PLAN.md Phase 3 accept: exact vs brute force over all C(|P|, k) for
    |P| <= 18."""
    rng = random.Random(seed)
    m = rng.randint(2, 18)
    pool = random_candidates(m, rng, n_range=(1, 20),
                              force_ties=(seed % 3 == 0),
                              force_boundary=(seed % 4 == 0))
    pool = [c for c in pool if c.n > 0]
    if not pool:
        pytest.skip("degenerate")
    k = rng.randint(1, len(pool))
    N_Q, n_Q = rng.randint(0, 20), rng.randint(1, 20)
    include_query = rng.random() < 0.5
    tau = rng.uniform(0, 1)

    result = solve(pool, k, tau, N_Q, n_Q, include_query)
    brute_selected, brute_U = brute_force_stage2(pool, k, tau, N_Q, n_Q, include_query)

    if brute_selected is None:
        assert result.feasible is False
        assert result.selected == []
    else:
        assert result.feasible is True
        assert result.objective_value == pytest.approx(brute_U, abs=1e-6)
        assert len(result.selected) == k


@pytest.mark.parametrize("seed", range(300))
def test_every_solution_is_binary_and_cardinality_correct(seed):
    """Guards the Bounds(0,1) trap directly: without it, milp can select a
    table more than once. Every feasible result must have exactly k distinct
    tables and satisfy the distribution constraint."""
    rng = random.Random(2000 + seed)
    m = rng.randint(2, 15)
    pool = random_candidates(m, rng, n_range=(1, 20))
    pool = [c for c in pool if c.n > 0]
    if not pool:
        pytest.skip("degenerate")
    k = rng.randint(1, len(pool))
    N_Q, n_Q = rng.randint(0, 20), rng.randint(1, 20)
    include_query = rng.random() < 0.5
    tau = rng.uniform(0, 1)

    result = solve(pool, k, tau, N_Q, n_Q, include_query)
    if not result.feasible:
        assert result.selected == []
    else:
        tables = [c.table for c in result.selected]
        assert len(tables) == k
        assert len(set(tables)) == k  # no table selected twice

        from duts.stats import aggregate, F as F_ratio
        N_S, n_S = aggregate(result.selected, N_Q, n_Q, include_query)
        assert F_ratio(N_S, n_S) >= tau - 1e-9  # never a constraint-violating set


def test_infeasible_returns_empty_never_violating():
    """An impossibly high tau must return infeasible, not a violating set."""
    pool = [CandidateStats(f"t{i}", N=1, n=10, U=float(i)) for i in range(5)]
    result = solve(pool, k=3, tau=0.99, N_Q=0, n_Q=0, include_query=False)
    assert result.feasible is False
    assert result.selected == []


def test_insufficient_pool_size():
    pool = [CandidateStats("a", 1, 1, 0.0)]
    result = solve(pool, k=5, tau=0.0)
    assert result.feasible is False
    assert result.info["lp_precheck_ran"] is False


def test_c1_rhs_is_negative_yq_not_zero():
    """C1: Stage 2's constraint RHS is -y_Q(tau), not 0 -- the paper's printed
    '0' is only the Q-free special case. Construct an instance that is
    feasible Q-free but infeasible once a large low-F query is folded in via
    include_query=True, demonstrating the RHS actually shifts."""
    pool = [CandidateStats("a", N=5, n=10, U=1.0), CandidateStats("b", N=4, n=10, U=0.5)]
    tau = 0.45
    N_Q, n_Q = 0, 100  # large, zero-matching query: pulls F down hard

    r_free = solve(pool, k=1, tau=tau, N_Q=N_Q, n_Q=n_Q, include_query=False)
    r_incl = solve(pool, k=1, tau=tau, N_Q=N_Q, n_Q=n_Q, include_query=True)

    assert r_free.feasible is True     # best single table alone clears tau=0.45
    assert r_incl.feasible is False    # but folded in with a huge zero-match query, it can't


def test_lp_precheck_matches_ilp_feasibility_verdict():
    """PLAN.md Phase 3 (revised): for this constraint shape (one cardinality
    equality + one linear inequality), LP-feasible <=> ILP-feasible -- proven
    by an exchange argument (see stage2_ilp.py docstring) and checked here
    across many random instances. This replaces the originally-planned
    'LP-feasible-but-ILP-infeasible' fixture, which cannot be constructed for
    this formulation.

    presolve is disabled on both solver calls (trap 3, stage2_ilp.py
    docstring): with scipy's default presolve, this exact loop originally
    caught `milp` reporting success=True on a genuinely infeasible instance
    (see test_milp_presolve_trap_regression) -- an artifact of the solver,
    not a counterexample to the exchange argument. A non-zero objective is
    used deliberately (a zero objective was part of what triggered trap 3).
    """
    rng = random.Random(99)
    for _ in range(1000):
        m = rng.randint(2, 10)
        k = rng.randint(1, m)
        y = np.array([rng.uniform(-10, 10) for _ in range(m)])
        rhs = rng.uniform(-10, 10)

        lp = linprog(
            c=-y,
            A_ub=np.array([-y]), b_ub=np.array([-rhs]),
            A_eq=np.array([np.ones(m)]), b_eq=np.array([float(k)]),
            bounds=[(0, 1)] * m, method="highs", options={"presolve": False},
        )
        ilp = milp(
            c=-y, integrality=np.ones(m), bounds=Bounds(0, 1),
            constraints=[LinearConstraint(y, lb=rhs), LinearConstraint(np.ones(m), lb=k, ub=k)],
            options={"presolve": False},
        )
        assert lp.success == ilp.success, (m, k, y, rhs)
        if ilp.success:
            x = np.round(ilp.x)
            assert y @ x >= rhs - 1e-6 and abs(x.sum() - k) < 1e-6, (
                "ilp.success=True but the returned solution violates a constraint", m, k, y, rhs
            )


def test_milp_presolve_trap_regression():
    """Permanent regression fixture for trap 3 (stage2_ilp.py docstring): a
    concrete instance where scipy 1.10.1's `milp`, with its default
    presolve=True, reports success=True and returns x=[0,1,1,1,1,0,0] with
    y.x = -0.772 against rhs = 2.165 -- a solution that VIOLATES the
    constraint it was asked to satisfy. Brute force over all C(7,4)=35
    subsets confirms the instance is genuinely infeasible. presolve=False
    (what duts.stage2_ilp.solve actually uses) reports the correct verdict.
    """
    import itertools

    m, k = 7, 4
    y = np.array([-7.15415005, -3.89880619, -4.79243782, -0.95212265,
                  8.87097597, -9.45564444, -5.36800353])
    rhs = 2.1653325972437703

    truly_feasible = any(
        sum(y[i] for i in combo) >= rhs for combo in itertools.combinations(range(m), k)
    )
    assert truly_feasible is False

    buggy = milp(
        c=-y, integrality=np.ones(m), bounds=Bounds(0, 1),
        constraints=[LinearConstraint(y, lb=rhs), LinearConstraint(np.ones(m), lb=k, ub=k)],
    )  # default presolve=True
    assert buggy.success is True  # the bug: scipy claims success on an infeasible instance
    bad_x = np.round(buggy.x)
    assert not (y @ bad_x >= rhs - 1e-6)  # ...and the "solution" it returns is invalid

    fixed = milp(
        c=-y, integrality=np.ones(m), bounds=Bounds(0, 1),
        constraints=[LinearConstraint(y, lb=rhs), LinearConstraint(np.ones(m), lb=k, ub=k)],
        options={"presolve": False},
    )
    assert fixed.success is False  # presolve=False -- what duts.stage2_ilp.solve uses -- is correct

    # duts.stage2_ilp.solve is exercised end-to-end against this exact bug class in
    # test_stage2_exact_vs_brute_force and test_every_solution_is_binary_and_cardinality_correct
    # (700 random instances total, independently re-verified against brute force and
    # against a fresh F/aggregate recomputation, not against milp's own success flag).


def test_y_tau_precomputation():
    c = CandidateStats("t", N=7, n=10, U=0.0)
    assert _y_tau(c, tau=0.5) == pytest.approx(7 - 0.5 * 10)
