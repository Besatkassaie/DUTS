"""``UnionabilityScorer`` adapters (PLAN-integration.md §2, Phase C; PLAN-full.md M3, M4).

§6.2 computes ``U_{V_D}(Q, T)`` as a maximum-weight bipartite matching between
the two tables' attributes -- but with the distributional pair
``(V_D, V_{D_T})`` **already determined during candidate generation**, hence
"fixed and excluded from the optimization".

**M3 -- the correction this module exists for.** starmie_fair's
``bounds.py:verify_constrained(t1, t2, threshold, mandatory_idx)`` pins only
the *row*: it adds a large bonus ``B`` to every entry of row ``mandatory_idx``,
which forces Hungarian to match that row somewhere, but leaves the *column*
choice to the optimizer. §6.2 pins the *pair* -- the column is ``V_{D_T}``,
already chosen by §7.3's per-table ``argmax sim``, not something the matcher
gets to revisit. The two are different computations and can return different
scores on the same input; ``tests/test_dutsx_unionability.py``'s
``test_pinned_and_starmie_disagree_constructed`` is a deliberate witness.

``PinnedMatchScorer`` implements §6.2 as written: record ``s(V_D, V_{D_T})``,
**delete that row and that column**, match the remainder, return
``s(pin) + sum_{(x,y) in M*} s(x, y)``. This is also cheaper than the
big-bonus trick -- it solves an assignment problem one row and one column
smaller.

**M4 -- no normalization.** ``U`` is the raw sum of similarities, matching
§6.2's formula and Starmie's published numbers. Table 1's ``U in [0,1]`` is
what is wrong in the paper; the formula is not. Stage 2's argmax is invariant
to positive scaling, so this affects reporting only, never the selected set.

**Threshold comparison direction.** ``bounds.py`` uses strict ``sim >
threshold``; §6.2 says "at least the threshold sigma" (``>=``). This module
keeps ``>`` deliberately, for comparability with Starmie's published numbers.
The discrepancy is noted rather than silently resolved -- see PLAN-full.md M3.
"""
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_THRESHOLD = 0.6  # sigma, matching bounds.py / Starmie's evaluation


def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity, matching ``bounds.py:cosine_sim`` exactly."""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def similarity_matrix(
    q_vecs: np.ndarray, c_vecs: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> np.ndarray:
    """Pairwise cosine similarity, with entries **not** above ``threshold``
    zeroed -- the edge set of §6.2's weighted bipartite graph.

    Strict ``>``, matching ``bounds.py``; see the module docstring.
    """
    n_q, n_c = len(q_vecs), len(c_vecs)
    graph = np.zeros((n_q, n_c), dtype=float)
    for i in range(n_q):
        for j in range(n_c):
            sim = cosine_sim(q_vecs[i], c_vecs[j])
            if sim > threshold:
                graph[i, j] = sim
    return graph


def score_pinned(
    q_vecs: np.ndarray,
    c_vecs: np.ndarray,
    pin: Tuple[int, int],
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """§6.2 + M3: ``s(pin) + max-weight matching over the remaining attributes``.

    Returns ``0.0`` when the pinned pair itself fails the similarity
    threshold -- the two tables are then not unionable on ``V_D`` at all, so
    no amount of agreement among the *other* attributes should earn a score.
    This mirrors ``verify_constrained``'s early return on an all-zero
    mandatory row, and is the one place the two implementations must agree by
    construction.
    """
    qi, tj = pin
    if not (0 <= qi < len(q_vecs)):
        raise ValueError(f"pin row {qi} out of range for query with {len(q_vecs)} attributes")
    if not (0 <= tj < len(c_vecs)):
        raise ValueError(f"pin col {tj} out of range for candidate with {len(c_vecs)} attributes")

    s_pin = cosine_sim(q_vecs[qi], c_vecs[tj])
    if not s_pin > threshold:
        return 0.0

    graph = similarity_matrix(q_vecs, c_vecs, threshold)

    # M3: delete the pinned row AND the pinned column, then match the rest.
    rows = [r for r in range(graph.shape[0]) if r != qi]
    cols = [c for c in range(graph.shape[1]) if c != tj]

    total = s_pin
    if rows and cols:
        sub = graph[np.ix_(rows, cols)]
        r_idx, c_idx = linear_sum_assignment(sub, maximize=True)
        total += float(sub[r_idx, c_idx].sum())
    return float(total)


def load_starmie_vectors(pkl_path: str) -> Dict[str, np.ndarray]:
    """Load a starmie_fair column-vector pickle into ``{table: ndarray}``.

    On-disk format is ``List[(table_name, ndarray(n_cols, dim))]``, float32,
    **not** L2-normalized. Cosine similarity is scale-invariant, so
    normalization is irrelevant *here* -- it matters for Phase B's HNSW index,
    where Euclidean distance is used as a cosine proxy (I3).
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    return {name: np.asarray(vecs) for name, vecs in raw}


class PinnedMatchScorer(object):
    """§6.2 as specified, with the ``(V_D, V_{D_T})`` pair pinned (M3).

    Satisfies ``ports.UnionabilityScorer``. Vectors are supplied as
    ``{table_name: ndarray(n_cols, dim)}``; use ``from_starmie_pickles`` to
    build one from the benchmark's existing artifacts.
    """

    def __init__(
        self,
        query_vectors: Dict[str, np.ndarray],
        candidate_vectors: Dict[str, np.ndarray],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.query_vectors = query_vectors
        self.candidate_vectors = candidate_vectors
        self.threshold = threshold
        self.n_scored = 0  # counts real scoring calls -- see note below

    @classmethod
    def from_starmie_pickles(
        cls, query_pkl: str, datalake_pkl: str, threshold: float = DEFAULT_THRESHOLD
    ) -> "PinnedMatchScorer":
        return cls(
            load_starmie_vectors(query_pkl),
            load_starmie_vectors(datalake_pkl),
            threshold=threshold,
        )

    def score(self, q_table: str, c_table: str, pin: Tuple[int, int]) -> float:
        """``U_{V_D}(Q, T)`` for one candidate.

        ``n_scored`` is incremented on every call that actually runs a
        matching. PLAN-integration.md §3 makes "#unionability computations"
        the headline efficiency metric (it should equal ``|P| = alpha*k``,
        independent of ``|T|``), and counting it at the scorer is the only
        place that cannot be fooled by a caller that batches or caches.
        """
        q_vecs = self.query_vectors.get(q_table)
        c_vecs = self.candidate_vectors.get(c_table)
        if q_vecs is None or c_vecs is None:
            raise KeyError(
                "no vectors for {!r} / {!r}".format(q_table, c_table)
            )
        self.n_scored += 1
        return score_pinned(q_vecs, c_vecs, pin, self.threshold)


class StarmieVerifyScorer(object):
    """Thin wrapper over starmie_fair's ``bounds.py:verify_constrained``.

    **Not** the §6.2 spec -- it pins the row and lets Hungarian pick the
    column (M3). Kept for comparability with Starmie's published numbers and
    as the contrast case that makes M3 concrete. Prefer ``PinnedMatchScorer``
    for anything the paper's results depend on.

    ``verify_constrained`` returns a ``(score, indexes)`` tuple; this unwraps
    it to the ``float`` the port requires.
    """

    def __init__(
        self,
        query_vectors: Dict[str, np.ndarray],
        candidate_vectors: Dict[str, np.ndarray],
        threshold: float = DEFAULT_THRESHOLD,
        starmie_root: Optional[str] = None,
    ) -> None:
        self.query_vectors = query_vectors
        self.candidate_vectors = candidate_vectors
        self.threshold = threshold
        # default: the Starmie modules bundled in <repo>/starmie_fair (bounds.verify_constrained)
        self.starmie_root = starmie_root or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "starmie_fair")
        self.n_scored = 0

    def _verify_constrained(self):
        """Import lazily: starmie_fair is an external read-only repo that may
        not be present, and importing it must not break `import dutsx`."""
        import sys

        if self.starmie_root not in sys.path:
            sys.path.insert(0, self.starmie_root)
        from bounds import verify_constrained

        return verify_constrained

    def score(self, q_table: str, c_table: str, pin: Tuple[int, int]) -> float:
        q_vecs = self.query_vectors.get(q_table)
        c_vecs = self.candidate_vectors.get(c_table)
        if q_vecs is None or c_vecs is None:
            raise KeyError("no vectors for {!r} / {!r}".format(q_table, c_table))
        verify_constrained = self._verify_constrained()
        self.n_scored += 1
        result = verify_constrained(
            q_vecs, c_vecs, threshold=self.threshold, mandatory_idx=pin[0]
        )
        score = result[0] if isinstance(result, tuple) else result
        return float(score)


class ConstantScorer(object):
    """Test double: every candidate scores the same.

    Makes retrieval effects observable without alignment noise -- with all
    ``U`` equal, Stage 2's argmax is decided purely by the distributional
    constraint, so any variation in the result set is attributable to
    retrieval or to Stage 1's pool, never to scoring.
    """

    def __init__(self, value: float = 1.0) -> None:
        self.value = float(value)
        self.n_scored = 0

    def score(self, q_table: str, c_table: str, pin: Tuple[int, int]) -> float:
        self.n_scored += 1
        return self.value
