"""Integration seams (PLAN-integration.md §2).

Four ``typing.Protocol`` definitions -- the entire swap surface between the
pure ``duts/`` core and real data. Replacing an implementation means writing
one class that satisfies the relevant Protocol, registering it in
``dutsx/registry.py``, and changing one config string; nothing that depends on
the port needs to change. (JOSIE was the original motivating example for
``OverlapFilter``; ``Fair_Table_Search_7.pdf`` dropped JOSIE from the paper
entirely in favor of posting lists -- see ``dutsx/adapters/overlap.py`` -- but
the swap-surface design this module provides doesn't depend on that example.)

``typing.Protocol`` is available on Python 3.8.5 with no ``typing_extensions``
dependency (verified -- PLAN-integration.md §6).

All four ports are implemented: ``SynopsisSource`` (Phase A,
``dutsx/adapters/synopsis.py``), ``SemanticRetriever`` and ``OverlapFilter``
(Phase B, ``dutsx/adapters/semantic.py`` / ``overlap.py``), and
``UnionabilityScorer`` (Phase C, ``dutsx/adapters/unionability.py``).
"""
from typing import Dict, List, Set, Tuple, Union

from typing import Protocol, runtime_checkable

import numpy as np

# An attribute is identified either by its column index (as
# ``protected_attributes_santos.csv``'s ``protected_attribute_id`` does) or
# by its column name (as callers reading a specific field may prefer).
# ``TableMetadata._resolve_column_identifier`` accepts either; the port does
# too, for the same reason.
AttrRef = Union[int, str]


@runtime_checkable
class SynopsisSource(Protocol):
    """Per-table row counts and per-attribute ``{value: count}`` histograms.

    This is the raw material for C2's ``N_i = sum_{c in M} count_i[c]``; the
    port never computes ``N_i`` itself -- that stays in ``duts/stats.py``,
    over a value *set* ``M`` supplied by the caller at query time.
    """

    def n_rows(self, table: str) -> int:
        """``n_i`` -- exact row count of ``table``."""
        ...

    def distribution(self, table: str, attr: AttrRef) -> Dict[str, int]:
        """Integer ``{value: count}`` histogram for one attribute of one
        table. Values are Python ``int`` (C2: never derived from float
        proportions). Returns ``{}`` if the table or attribute is unknown."""
        ...

    def categorical_attrs(self, table: str) -> List[AttrRef]:
        """Attributes of ``table`` that pass the categorical-domain-size
        filter (§7's ``theta_cat``). Column indices, ascending."""
        ...


@runtime_checkable
class SemanticRetriever(Protocol):
    """§7.1 -- ANN over attribute embeddings. No implementation until Phase B."""

    def query(self, vec: np.ndarray, top_n: int) -> List[Tuple[str, int, float]]:
        """-> ``[(table, attr_idx, similarity)]``, already threshold-filtered."""
        ...


@runtime_checkable
class OverlapFilter(Protocol):
    """§7.2 (v7: "Value-overlap filtering via posting lists") -- an exact
    Boolean membership test, |M ∩ V_c| >= 1, no ranking. The returned size is
    diagnostic only -- §7.3's tie-break uses similarity alone, never overlap.
    Implemented in ``dutsx/adapters/overlap.py``."""

    def query(self, M: Set[str]) -> Dict[Tuple[str, int], int]:
        ...


@runtime_checkable
class UnionabilityScorer(Protocol):
    """§6.2 -- ``U_{V_D}(Q, T_i)`` for ONE candidate, with the ``V_D`` pair
    pinned. No implementation until Phase C."""

    def score(self, q_table: str, c_table: str, pin: Tuple[int, int]) -> float:
        ...
