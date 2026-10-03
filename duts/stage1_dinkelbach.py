"""Stage 1: distribution-aware candidate selection via Dinkelbach's parametric
method (PLAN.md §5; C1, C3, C4, C5).

    P = argmax_{S subseteq D, |S| = alpha*k}  F(T(S u Q))

Each Dinkelbach iteration is one deterministic top-(alpha*k) selection on
``y_i(lambda) = N_i - lambda*n_i`` (linear in the decision variables, hence
cheap); the fixed point is the exact optimum (Phi(lambda*) = 0 characterizes
lambda* = F*, the governing theorem behind Dinkelbach's method).

C1: Q enters as a constant offset in the Phi(lambda) evaluation and in the
lambda update -- the inner top-(alpha*k) maximizer itself is unaffected by Q.

C5 (seeding): lambda^(0) is seeded from an ACTUAL size-`size` subset (the
top-`size` by per-table F_i), not the full-pool ratio over all of D. Under
C1/include_query=True the full-pool ratio is not guaranteed <= lambda*, so
seeding from it can make the first step *decrease* -- see PLAN.md C5 and the
regression fixture in tests/test_stage1.py::test_c4_counterexample_regression.
Monotonicity is only guaranteed from iteration 1 onward with this seeding.
"""
from typing import Iterator, List, Tuple

from .stats import aggregate, drop_empty
from .types import CandidateStats, StageResult

MAX_ITERATIONS = 100
CONVERGENCE_TOL = 1e-12


def _y(c: CandidateStats, lam: float) -> float:
    """y_i(lambda) = N_i - lambda * n_i (§5)."""
    return c.N - lam * c.n


def _tie_key(c: CandidateStats, lam: float):
    """Deterministic ranking key: descending y_i(lambda), ties broken by
    (-n_i, table_id) (C5) -- required for the set-equality termination test
    to mean anything. Python's sort is ascending, hence the negations."""
    return (-_y(c, lam), -c.n, c.table)


def top_by_y(candidates: List[CandidateStats], lam: float, size: int) -> List[CandidateStats]:
    """The exact-`size` subset maximizing sum(y_i(lambda)), by sort
    (O(m log m), matching §4.1's stated bound)."""
    return sorted(candidates, key=lambda c: _tie_key(c, lam))[:size]


def _seed_lambda(candidates: List[CandidateStats], size: int, N_Q: int, n_Q: int,
                  include_query: bool) -> float:
    """C5: seed lambda^(0) from the top-`size` subset by per-table F_i -- a real
    achievable point -- rather than the full-pool ratio."""
    seed = sorted(candidates, key=lambda c: (-(c.N / c.n), -c.n, c.table))[:size]
    N_seed, n_seed = aggregate(seed, N_Q, n_Q, include_query)
    return N_seed / n_seed


def iterate(
    candidates: List[CandidateStats],
    size: int,
    N_Q: int = 0,
    n_Q: int = 0,
    include_query: bool = True,
    max_iterations: int = MAX_ITERATIONS,
    tol: float = CONVERGENCE_TOL,
) -> Iterator[Tuple[int, float, List[CandidateStats]]]:
    """Yield (iteration, lambda_next, selected) at each Dinkelbach step.

    The last yielded (lambda_next, selected) is the exact optimum. Exposed
    separately from `solve` so tests can inspect the lambda trace directly
    (PLAN.md C5: "log iteration counts"; the monotonicity property is
    asserted on this trace, starting at iteration 1).

    Raises ValueError if fewer than `size` candidates remain after dropping
    n_i=0 entries. Raises RuntimeError if not converged within
    `max_iterations`.
    """
    candidates = drop_empty(candidates)
    if len(candidates) < size:
        raise ValueError(
            f"candidate pool has {len(candidates)} non-empty tables, cannot select {size}"
        )

    lam = _seed_lambda(candidates, size, N_Q, n_Q, include_query)
    prev_selection = None

    for iteration in range(1, max_iterations + 1):
        selected = top_by_y(candidates, lam, size)
        selected_ids = tuple(sorted(c.table for c in selected))
        N_sel, n_sel = aggregate(selected, N_Q, n_Q, include_query)
        lam_next = N_sel / n_sel

        yield iteration, lam_next, selected

        if selected_ids == prev_selection or abs(lam_next - lam) <= tol:
            return
        lam = lam_next
        prev_selection = selected_ids

    raise RuntimeError(f"Dinkelbach did not converge within {max_iterations} iterations")


def solve(
    candidates: List[CandidateStats],
    size: int,
    N_Q: int = 0,
    n_Q: int = 0,
    include_query: bool = True,
    max_iterations: int = MAX_ITERATIONS,
    tol: float = CONVERGENCE_TOL,
) -> StageResult:
    """Exact solution to max_{|S|=size} F(T(S u Q)) via Dinkelbach's method."""
    n_iter = 0
    lam = None
    selected: List[CandidateStats] = []
    for n_iter, lam, selected in iterate(
        candidates, size, N_Q, n_Q, include_query, max_iterations, tol
    ):
        pass
    return StageResult(selected=selected, feasible=True, objective_value=lam, iterations=n_iter)


def certify_feasible(
    candidates: List[CandidateStats],
    k: int,
    tau: float,
    N_Q: int = 0,
    n_Q: int = 0,
    include_query: bool = True,
) -> Tuple[bool, float, int]:
    """C4: run Dinkelbach at cardinality k over ALL of D.

    Because Problem (1) fixes |R| = k with R subseteq D, F_max^k :=
    max_{|S|=k, S subseteq D} F(T(S u Q)) is exactly the best proportion any
    feasible solution could achieve -- sound directly from the cardinality
    constraint, independent of the (retired, and false under C1) claim that
    max_{|S|=k} F >= max_{|S|=alpha*k} F. See PLAN.md C4.

    Returns (F_max_k >= tau, F_max_k, dinkelbach_iterations). If |D| < k,
    returns (False, -inf, 0) -- infeasible on cardinality grounds alone.
    """
    non_empty = drop_empty(candidates)
    if len(non_empty) < k:
        return False, float("-inf"), 0
    result = solve(non_empty, k, N_Q, n_Q, include_query)
    return result.objective_value >= tau, result.objective_value, result.iterations
