"""``OverlapFilter`` adapters (PLAN-integration.md §2, Phase B; §7.2).

**§7.2, "Value-overlap filtering via posting lists" (Fair_Table_Search_7.pdf,
superseding the JOSIE-based §7.2 in v6).** The paper's own retention rule is a
Boolean membership test, not a similarity score:

    |M ∩ V_c| >= 1

where ``V_c`` is categorical attribute ``c``'s value set -- evaluated exactly
via a classical inverted index (posting lists), explicitly framed as the
"database system" query paradigm (return every matching record) rather than
the ranked, fixed-size result sets of "search-engine-style retrieval" that
JOSIE represents. `[13]` no longer cites JOSIE; it cites Zobel & Moffat,
*Inverted files for text search engines* (2006), as the design this section
follows. **JOSIE is not part of the paper as of v7** -- no ``JosieOverlap``
adapter should be built; see PLAN-full.md I1 for the full history (the
question was resolved twice in one day, in opposite directions -- the paper
itself is what won).

``InvertedIndexOverlap`` below was already conforming before this section was
found, unmodified: the Boolean retention rule, `theta_cat` bounding only how
many posting lists one attribute appears in (never posting-list *length*),
and the §7.3 `argmax sim` tie-break never reading the overlap count (checked
at ``dutsx/retrieval.py``) all match the v7 spec exactly.

One implementation recommendation from v7 not yet taken: each posting list as
a **Roaring bitmap** (CRoaring / PyRoaring), for efficiency at scale. Plain
Python ``dict``/``set`` is used here instead -- adequate at santos scale
(550-931 tables), worth reconsidering only for a much larger benchmark.

* ``InvertedIndexOverlap`` -- exact ``value -> {(table, attr_idx)}``
  postings, built once from a ``SynopsisSource``'s categorical attributes;
  ``query(M)`` probes ``union_{c in M} postings[c]`` and returns, per
  surviving ``(table, attr_idx)`` pair, ``|{c in M : c in domain(attr)}|`` --
  a diagnostic overlap *size*, never used as a score (the paper's Boolean
  rule and §7.3's tie-break both ignore it beyond membership).
* ``NullOverlap`` -- returns every indexed ``(table, attr_idx)`` pair
  regardless of ``M``, for ablating the overlap filter's contribution to
  ``D_pair`` (PLAN-integration.md Phase B deliverable 2).

I4's constraint applies to both: ``M`` and the target values are query-time
parameters, so nothing here is keyed by ``M`` at build time -- only by
individual value, per attribute.
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from ..ports import SynopsisSource


def _build_pairs(
    synopsis: SynopsisSource, tables: List[str]
) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    for table in tables:
        for attr in synopsis.categorical_attrs(table):
            pairs.append((table, attr))
    return pairs


class InvertedIndexOverlap(object):
    """``OverlapFilter`` via an exact ``value -> {(table, attr_idx)}`` index.

    Satisfies ``ports.OverlapFilter``. Built once from every categorical
    attribute (``synopsis.categorical_attrs``) of every table in ``tables``;
    ``query(M)`` unions the postings lists for each value in ``M`` and
    tallies, per surviving pair, how many of ``M``'s values it actually
    matched.
    """

    def __init__(self, synopsis: SynopsisSource, tables: List[str]) -> None:
        self._postings: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)
        n_pairs = 0
        for table in tables:
            for attr in synopsis.categorical_attrs(table):
                n_pairs += 1
                dist = synopsis.distribution(table, attr)
                for value in dist:
                    self._postings[value].add((table, attr))
        self.n_pairs_indexed = n_pairs

    def query(self, M: Set[str]) -> Dict[Tuple[str, int], int]:
        counts: Dict[Tuple[str, int], int] = defaultdict(int)
        for value in M:
            for pair in self._postings.get(value, ()):
                counts[pair] += 1
        return dict(counts)


class NullOverlap(object):
    """``OverlapFilter`` that returns every indexed ``(table, attr_idx)``
    pair for any ``M``, with a constant overlap value.

    Used to ablate the overlap filter's contribution to ``D_pair`` (Phase B
    deliverable): with this in place of ``InvertedIndexOverlap``,
    ``D_ovl == D`` (the whole indexed universe) and ``D_pair`` collapses to
    exactly ``D_sem`` after the per-table argmax, isolating what the
    semantic filter alone contributes.
    """

    def __init__(
        self, synopsis: SynopsisSource, tables: List[str], overlap_value: int = 1
    ) -> None:
        self._pairs = _build_pairs(synopsis, tables)
        self._overlap_value = overlap_value

    def query(self, M: Set[str]) -> Dict[Tuple[str, int], int]:
        return {pair: self._overlap_value for pair in self._pairs}
