"""Stage 2: unionability-aware selection via 0-1 ILP (PLAN.md §6.1; C1).

    R = argmax_{S subseteq P, |S| = k}  sum_{T_i in S} U_{V_D}(Q,T_i)
          s.t.  Delta(F*, F(T(S u Q))) <= delta

With tau := F* - delta fixed, the fractional constraint linearizes to a
single inequality: sum_i y_i(tau)*x_i >= -y_Q(tau), where y_i(tau) =
N_i - tau*n_i. C1: the RHS is -y_Q(tau), not 0 -- the paper's printed "0" is
the Q-free special case.

Three silent scipy.optimize.milp traps this module exists to avoid:
  1. `milp` MINIMIZES -- the objective passed here is -U, not U.
  2. `bounds` defaults to Bounds(0, inf); `integrality=1` alone means
     "non-negative integer", not binary. Omitting bounds=Bounds(0,1) lets the
     solver select the same table more than once while still satisfying both
     constraints -- verified on scipy 1.10.1: maximizing 10*x0+x1+x2 s.t.
     sum(x)=2 returns [2,0,0] (obj 20) without bounds, vs [1,0,1] (obj 11,
     correct) with Bounds(0,1).
  3. HiGHS's default presolve (`options={"presolve": True}`, scipy's default)
     can report success=True with a solution that VIOLATES the distribution
     constraint, on genuinely infeasible instances. Reproduced on scipy 1.10.1
     AND 1.13.1, with two independent constraint formulations, at ~1-in-3000
     on random small instances (see tests/test_stage2.py); confirmed as a
     presolve artifact because options={"presolve": False} on the exact same
     instance correctly reports infeasible, and a brute-force scan over all
     C(m,k) subsets confirms no feasible subset exists. Mitigation, applied
     below: (a) call milp with presolve disabled -- pool sizes here are
     bounded by alpha*k, independent of |T|, so presolve's speed benefit is
     not needed; (b) independently re-verify every returned solution against
     the actual constraints before trusting `success`, regardless of solver
     flags -- belt and suspenders once a solver has been caught lying once.

LP pre-check via linprog: LP-infeasible => ILP-infeasible (sound). For this
specific constraint shape -- one cardinality equality plus one linear
inequality, nothing else -- the converse also holds: an exchange argument
(shift mass t between any two fractional coordinates; Sigma x = k is
preserved exactly and y.x moves linearly in t, so one direction is
non-decreasing; walk it to a boundary) shows LP-feasible implies
ILP-feasible here (verified in tests/test_stage2.py, presolve disabled to
avoid trap 3 above). So the LP pre-check can never produce a false green
light for this formulation -- its only value is computational, skipping
branch-and-bound when the LP already proves infeasible.
"""
import time
from typing import List

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .types import CandidateStats, StageResult


def _y_tau(c: CandidateStats, tau: float) -> float:
    """y_i(tau) = N_i - tau * n_i, precomputed once per candidate (§6.1)."""
    return c.N - tau * c.n


def solve(
    pool: List[CandidateStats],
    k: int,
    tau: float,
    N_Q: int = 0,
    n_Q: int = 0,
    include_query: bool = True,
    lp_precheck: bool = True,
) -> StageResult:
    """Exact solution to max sum(U_i) s.t. |S|=k, F(T(S u Q)) >= tau, over
    `pool`. StageResult.selected is [] and .feasible is False if no size-k
    subset of `pool` satisfies the constraint (or |pool| < k)."""
    m = len(pool)
    info = {"lp_precheck_ran": False, "lp_feasible": None, "lp_time_s": None, "milp_time_s": None}

    if m < k:
        return StageResult(selected=[], feasible=False, objective_value=None,
                            iterations=0, info=info)

    y = np.array([_y_tau(c, tau) for c in pool], dtype=float)
    U = np.array([c.U for c in pool], dtype=float)
    y_Q = (N_Q - tau * n_Q) if include_query else 0.0
    rhs = -y_Q  # C1: RHS is -y_Q(tau); 0 is only the Q-free special case

    if lp_precheck:
        t0 = time.perf_counter()
        lp_result = linprog(
            c=-U,
            A_ub=np.array([-y]),
            b_ub=np.array([-rhs]),
            A_eq=np.array([np.ones(m)]),
            b_eq=np.array([float(k)]),
            bounds=[(0, 1)] * m,
            method="highs",
            options={"presolve": False},  # trap 3: presolve can misreport infeasible instances as feasible
        )
        info["lp_precheck_ran"] = True
        info["lp_feasible"] = bool(lp_result.success)
        info["lp_time_s"] = time.perf_counter() - t0
        if not lp_result.success:
            # LP-infeasible => ILP-infeasible (sound); skip the ILP entirely.
            return StageResult(selected=[], feasible=False, objective_value=None,
                                iterations=0, info=info)

    card = LinearConstraint(np.ones(m), lb=k, ub=k)
    dist = LinearConstraint(y, lb=rhs)  # ub defaults to +inf

    t0 = time.perf_counter()
    result = milp(
        c=-U,
        integrality=np.ones(m),
        bounds=Bounds(0, 1),  # trap 2: without this, integrality=1 alone is non-negative-integer
        constraints=[dist, card],
        options={"presolve": False},  # trap 3: see module docstring
    )
    info["milp_time_s"] = time.perf_counter() - t0

    if not result.success:
        return StageResult(selected=[], feasible=False, objective_value=None,
                            iterations=0, info=info)

    x = np.round(result.x).astype(int)
    assert set(np.unique(x)).issubset({0, 1}), "milp returned a non-binary solution"
    assert x.sum() == k, "milp returned a solution violating the cardinality constraint"
    assert float(y @ x) >= rhs - 1e-6, (
        "milp reported success on a solution violating the distribution constraint "
        "(trap 3 -- see module docstring); this should be unreachable with "
        "presolve disabled, so its appearance means the mitigation itself needs revisiting"
    )

    selected = [pool[i] for i in range(m) if x[i] == 1]
    total_U = sum(c.U for c in selected)
    return StageResult(selected=selected, feasible=True, objective_value=total_U,
                        iterations=0, info=info)
