"""Distribution statistics over the value set M (PLAN.md §3; C1, C2).

C2: M is a value *set*, not a single value; N_i = sum_{c in M} count_i[c] is
computed in integer arithmetic throughout and never derived from float
proportions. C1: the query enters every aggregate as an explicit
``(N_Q, n_Q)`` offset rather than being folded into the candidate list.
"""
from typing import Dict, Iterable, List, Tuple

from .types import CandidateStats


def N_of(counts: Dict, M: Iterable) -> int:
    """Exact integer count of tuples matching M, from a table's raw
    ``{value: count}`` histogram synopsis (C2)."""
    return sum(counts.get(v, 0) for v in M)


def drop_empty(candidates: List[CandidateStats]) -> List[CandidateStats]:
    """Drop n_i=0 candidates (C5) -- such a table can never contribute to F."""
    return [c for c in candidates if c.n > 0]


def F(N: int, n: int) -> float:
    """Achieved proportion N/n. Undefined at n=0; n_i=0 candidates must be
    dropped upstream (drop_empty) before this is ever called on them."""
    if n == 0:
        raise ZeroDivisionError("F undefined for n=0; drop n_i=0 candidates upstream")
    return N / n


def aggregate(
    candidates: List[CandidateStats],
    N_Q: int = 0,
    n_Q: int = 0,
    include_query: bool = True,
) -> Tuple[int, int]:
    """Aggregate (N, n) over a candidate set, optionally offset by the query
    (C1). Returns exact integers -- never derived from float proportions."""
    N_S = sum(c.N for c in candidates)
    n_S = sum(c.n for c in candidates)
    if include_query:
        N_S += N_Q
        n_S += n_Q
    return N_S, n_S


def F_of_set(
    candidates: List[CandidateStats],
    N_Q: int = 0,
    n_Q: int = 0,
    include_query: bool = True,
) -> float:
    return F(*aggregate(candidates, N_Q, n_Q, include_query))


def delta(F_star: float, F_S: float) -> float:
    """Signed deviation F* - F(S); may be negative (overshoot is admissible)."""
    return F_star - F_S


def weighted_average_identity(candidates: List[CandidateStats]) -> float:
    """F_S as the n_i-weighted average of the per-table {F_i} (§5's identity).

    This identity is Q-FREE: it holds for aggregate(candidates,
    include_query=False). Under C1 (include_query=True) it does not hold,
    because Q is forced into the average at every cardinality -- see
    PLAN.md C4 for the consequence this has for infeasibility certification.
    """
    n_total = sum(c.n for c in candidates)
    if n_total == 0:
        raise ZeroDivisionError("empty or all n_i=0 candidate set")
    return sum(c.n * (c.N / c.n) for c in candidates) / n_total
