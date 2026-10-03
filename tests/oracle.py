"""Brute-force exact solvers for Stage 1 and Stage 2 (PLAN.md §6).

The load-bearing test device: because both stages are pure functions of
List[CandidateStats] + (N_Q, n_Q), exhaustive enumeration over C(m, size) is
fast for m <= ~18, turning "Dinkelbach is exact" and "the ILP is exact" from
claims in the paper into properties verified against every subset.
"""
import itertools
from typing import List, Optional, Tuple

from duts.stats import F as F_ratio
from duts.stats import aggregate
from duts.types import CandidateStats


def brute_force_stage1(
    candidates: List[CandidateStats],
    size: int,
    N_Q: int = 0,
    n_Q: int = 0,
    include_query: bool = True,
) -> Tuple[List[CandidateStats], float]:
    """Exhaustive max_{|S|=size} F(T(S u Q)) over every C(len(candidates), size)
    subset. Ties broken arbitrarily (first-seen) -- callers should compare
    achieved values, not the identity of the selected subset."""
    best_F = None
    best_subset: Optional[Tuple[CandidateStats, ...]] = None
    for combo in itertools.combinations(candidates, size):
        N_S, n_S = aggregate(list(combo), N_Q, n_Q, include_query)
        f = F_ratio(N_S, n_S)
        if best_F is None or f > best_F:
            best_F, best_subset = f, combo
    return list(best_subset), best_F


def brute_force_stage2(
    candidates: List[CandidateStats],
    size: int,
    tau: float,
    N_Q: int = 0,
    n_Q: int = 0,
    include_query: bool = True,
) -> Tuple[Optional[List[CandidateStats]], Optional[float]]:
    """Exhaustive max sum(U_i) s.t. |S|=size and F(T(S u Q)) >= tau, over
    every C(len(candidates), size) subset. Returns (None, None) if no subset
    of the given size satisfies the constraint."""
    best_U = None
    best_subset: Optional[Tuple[CandidateStats, ...]] = None
    for combo in itertools.combinations(candidates, size):
        N_S, n_S = aggregate(list(combo), N_Q, n_Q, include_query)
        f = F_ratio(N_S, n_S)
        if f < tau - 1e-9:  # numeric guard against boundary float error
            continue
        u = sum(c.U for c in combo)
        if best_U is None or u > best_U:
            best_U, best_subset = u, combo
    if best_subset is None:
        return None, None
    return list(best_subset), best_U
