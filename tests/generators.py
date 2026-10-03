"""Synthetic instance generators for property + brute-force testing (PLAN.md §6).

Includes Theorem 3.2's E-kKP reduction as an instance generator: constructing
tables from a knapsack instance and solving with Stage 2 validates the
reduction itself as a side effect, cheap insurance on the paper's hardness
proof.
"""
import random
from typing import List, Sequence, Tuple

from duts.types import CandidateStats


def random_candidates(
    m: int,
    rng: random.Random,
    n_range: Tuple[int, int] = (1, 50),
    n_i_zero_prob: float = 0.0,
    force_ties: bool = False,
    force_boundary: bool = False,
) -> List[CandidateStats]:
    """m candidates with integer (N_i, n_i) and a float U_i in [0,1].

    n_i_zero_prob: fraction deliberately given n_i=0 (must be dropped upstream).
    force_ties: bias n_i toward a small set of repeated values, so y_i(lambda)
        ties are common -- exercises the deterministic tie-break rule (C5).
    force_boundary: sprinkle N_i=0 and N_i=n_i candidates (C5's degenerate cases).
    """
    out = []
    for i in range(m):
        if rng.random() < n_i_zero_prob:
            n_i = 0
        elif force_ties:
            n_i = rng.choice([10, 20, 20, 10, 30])
        else:
            n_i = rng.randint(*n_range)

        if n_i == 0:
            N_i = 0
        elif force_boundary and rng.random() < 0.2:
            N_i = rng.choice([0, n_i])
        else:
            N_i = rng.randint(0, n_i)

        out.append(CandidateStats(table=f"t{i}", N=N_i, n=n_i, U=rng.random()))
    return out


def knapsack_to_tables(
    profits: Sequence[float], weights: Sequence[int], c: int
) -> List[CandidateStats]:
    """Theorem 3.2's E-kKP reduction: for each item j with profit p_j and
    weight w_j, construct a table with n_j = c (capacity), N_j = c - w_j,
    U_j = p_j. Combine with tau_from_knapsack_capacity for the matching tau,
    and solve Q-free (include_query=False) -- the reduction's arithmetic
    (F_S = sum(N_j) / sum(n_j)) has no query term."""
    tables = []
    for j, (p, w) in enumerate(zip(profits, weights)):
        if w > c:
            raise ValueError(f"item weight {w} exceeds capacity {c}")
        tables.append(CandidateStats(table=f"item{j}", N=c - w, n=c, U=float(p)))
    return tables


def tau_from_knapsack_capacity(k: int, c: int, capacity: int) -> float:
    """F* - delta = 1 - capacity / (k * c).

    Derivation: with n_j = c for every table and |S| = k,
        F_S = 1 - sum_{j in S} w_j / (k*c),
    so F_S >= tau  <=>  sum(w_j) <= k*c*(1-tau). Matching a knapsack capacity
    C therefore requires tau = 1 - C/(k*c) (PLAN.md §6 -- corrects the
    original (k-1)/k fixture, which only encodes the special case C = c).
    """
    return 1.0 - capacity / (k * c)
