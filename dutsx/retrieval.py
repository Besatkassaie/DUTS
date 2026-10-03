"""§4.2/§7.3 candidate retrieval: ``D_pair = D_sem ∩ D_ovl``, then per-table
``argmax sim`` (PLAN-integration.md §2, Phase B).

The two retrieval signals -- semantic (§7.1, ``SemanticRetriever``) and
value-overlap (§7.2, ``OverlapFilter``) -- are retrieved **independently**
(PLAN-full.md §5's clarification: the intersection is parallel, not a
downstream-of-HNSW pipeline) and combined at the ``(table, attr_idx)`` **pair**
level, not at the table level. This is §7.3's explicit requirement and the
one detail that's easy to get backwards: a table that clears the semantic
filter through one attribute and the overlap filter through a *different*
attribute must NOT survive, because neither filter alone establishes that
*that attribute* is both semantically aligned to ``V_D`` and has the required
value overlap.

Only after the pair-level intersection does the per-table reduction happen:
if a table has more than one surviving pair, keep exactly one -- the
highest-``sim`` pair (§7.3's ``argmax sim`` tie-break) -- so each table
contributes at most one ``(attr, sim, overlap)`` triple downstream.
"""
from typing import Dict, List, NamedTuple, Set, Tuple

import numpy as np

from .ports import OverlapFilter, SemanticRetriever


class RetrievedCandidate(NamedTuple):
    table: str
    attr: int      # the ONE attribute aligned to V_D (§7.3 argmax)
    sim: float      # cosine sim to V_D (§7.3 tie-break; §6.2 pin weight)
    overlap: int    # |values(attr) ∩ M| -- reported, never scored (§7.3)


class RetrievalTelemetry(NamedTuple):
    n_sem: int    # distinct (table, attr) pairs surviving the semantic filter
    n_ovl: int    # distinct (table, attr) pairs surviving the overlap filter
    n_pair: int   # distinct (table, attr) pairs surviving BOTH, pre-argmax
    n_d: int      # distinct TABLES surviving after the per-table argmax


def retrieve_candidates(
    semantic: SemanticRetriever,
    overlap: OverlapFilter,
    query_vec: np.ndarray,
    M: Set[str],
    top_n: int,
) -> Tuple[List[RetrievedCandidate], RetrievalTelemetry]:
    """§7.3: ``D_sem`` (ANN over ``query_vec``) intersected with ``D_ovl``
    (inverted-index probe of ``M``) at the ``(table, attr_idx)`` pair level,
    then reduced to one pair per table by ``argmax sim``.

    Returns ``(D, telemetry)`` where ``D`` is the final table-level candidate
    list (``len(D) == telemetry.n_d``) and ``telemetry`` logs ``|D_sem|``,
    ``|D_ovl|``, ``|D_pair|``, ``|D|`` per query (Phase B accept criterion).
    """
    d_sem = semantic.query(query_vec, top_n)          # [(table, attr, sim)]
    d_ovl = overlap.query(M)                            # {(table, attr): overlap}

    sem_pairs: Dict[Tuple[str, int], float] = {
        (table, attr): sim for (table, attr, sim) in d_sem
    }
    pair_keys = set(sem_pairs) & set(d_ovl)

    best_per_table: Dict[str, RetrievedCandidate] = {}
    for (table, attr) in pair_keys:
        sim = sem_pairs[(table, attr)]
        ov = d_ovl[(table, attr)]
        current = best_per_table.get(table)
        if current is None or sim > current.sim:
            best_per_table[table] = RetrievedCandidate(table=table, attr=attr, sim=sim, overlap=ov)

    d = list(best_per_table.values())
    telemetry = RetrievalTelemetry(
        n_sem=len(sem_pairs),
        n_ovl=len(d_ovl),
        n_pair=len(pair_keys),
        n_d=len(d),
    )
    return d, telemetry
