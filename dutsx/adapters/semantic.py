"""``SemanticRetriever`` adapters (PLAN-integration.md §2, Phase B; §7.1, I2, I3).

Two implementations:

* ``HnswRetriever`` -- hnswlib over the existing Starmie column vectors
  (``vectors/cl_{datalake,query}_drop_col_tfidf_entity_column_0.pkl``, I2:
  one embedding space shared with §6.2's unionability scoring). Built with
  ``space='cosine'``, matching starmie_fair's own canonical usage
  (``HNSWSearcher_Fair.py:152-156``) -- hnswlib's cosine space normalizes
  internally, so it returns cosine *distance* (``1 - cos_sim``) directly and
  needs no separate L2-normalization step or ``theta_dis`` conversion.
* ``ExactScanRetriever`` -- brute-force cosine over every column. The recall
  oracle for ``HnswRetriever``, not a retriever to ship: santos is small
  enough (6322 datalake columns) to make this cheap, and it is what makes
  "HNSW recall vs exact scan" (PLAN-integration.md Phase B accept criterion)
  a real measurement rather than an assumption.

**I3 -- Euclidean vs cosine.** §7.1 as printed filters on Euclidean distance
``< theta_dis``; §6.2 uses cosine similarity ``> sigma``. The two are only
the same ranking if every embedding is L2-normalized first
(``||a-b||^2 = 2 - 2*cos(a,b)``, so ``theta_dis = sqrt(2 - 2*sigma)``); the
stored Starmie vectors are **not** normalized (sampled norm approx 9.75).
Shipping ``space='cosine'`` sidesteps that conversion entirely (point 1 of
the coordinator's clarification: cosine space already returns
``1 - cos_sim``), but the *paper's specific claim* -- that thresholding
Euclidean distance on L2-normalized vectors at ``theta_dis = sqrt(2-2sigma)``
selects the same neighbours as thresholding cosine similarity at ``sigma``
-- is still worth checking empirically rather than assumed, since it is the
thing that lets §7.1's Euclidean-distance prose and §6.2's cosine-similarity
formula be read as describing the same filter. That check is done directly
against raw hnswlib ``space='l2'`` vs ``space='cosine'`` indexes in
``tests/test_dutsx_retrieval.py`` (not routed through this module, which only
ever builds ``space='cosine'``) -- see
``test_l2_theta_dis_equivalent_to_cosine_sigma``.
"""
from typing import Dict, List, Tuple

import hnswlib
import numpy as np

DEFAULT_SIGMA = 0.6  # sigma, matching unionability.py / bounds.py / experimental_config.json


def _flatten(vectors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """``{table: ndarray(n_cols, dim)}`` -> ``(data, labels)`` where
    ``labels[i] == (table, attr_idx)`` for row ``i`` of ``data``.

    Zero vectors are skipped -- cosine similarity/normalization is undefined
    for them, and none are expected in Starmie's TF-IDF-derived embeddings,
    but a defensive skip is cheap and keeps ``add_items``/norm math from
    producing ``nan``.
    """
    labels: List[Tuple[str, int]] = []
    rows: List[np.ndarray] = []
    for table, arr in vectors.items():
        arr = np.asarray(arr, dtype=np.float32)
        for j in range(arr.shape[0]):
            v = arr[j]
            if np.linalg.norm(v) == 0.0:
                continue
            rows.append(v)
            labels.append((table, j))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.vstack(rows).astype(np.float32), labels


class HnswRetriever(object):
    """``SemanticRetriever`` backed by hnswlib, ``space='cosine'`` (§7.1, I2/I3).

    Satisfies ``ports.SemanticRetriever``: ``query(vec, top_n)`` returns
    ``[(table, attr_idx, similarity)]``, nearest first, already filtered to
    ``similarity > sigma``.
    """

    def __init__(
        self,
        vectors: Dict[str, np.ndarray],
        sigma: float = DEFAULT_SIGMA,
        ef_construction: int = 200,
        M: int = 32,
        ef: int = 100,
        seed: int = 42,
    ) -> None:
        self.sigma = sigma
        data, self._labels = _flatten(vectors)
        self.n_indexed = len(self._labels)
        self._dim = data.shape[1] if data.size else 0
        self._ef = max(ef, 1)
        self.index = hnswlib.Index(space="cosine", dim=max(self._dim, 1))
        if len(self._labels) > 0:
            self.index.init_index(
                max_elements=len(self._labels),
                ef_construction=ef_construction,
                M=M,
                random_seed=seed,
            )
            self.index.add_items(data, np.arange(len(self._labels)))
            self.index.set_ef(self._ef)

    def query(self, vec: np.ndarray, top_n: int) -> List[Tuple[str, int, float]]:
        if not self._labels or top_n <= 0:
            return []
        v = np.asarray(vec, dtype=np.float32)
        if np.linalg.norm(v) == 0.0:
            return []
        k = min(top_n, len(self._labels))
        # hnswlib silently degrades recall (not an error) when ef < k, since its greedy search
        # only explores ef candidates regardless of k requested -- match ef to the requested
        # top_n every call so each setting reflects a properly-tuned search at that breadth, not
        # one inherited from a previous (possibly larger) top_n in the same run. A small margin
        # above k (not ef==k exactly) is required, not just good practice: ef==k occasionally
        # makes hnswlib raise "Cannot return the results in a contiguous 2D array" outright
        # (not a recall issue, a hard failure) -- reproduced empirically as a rare, non-
        # deterministic crash at top_n=5000 against a ~500K-vector index (2/2 and 1/2 across
        # trial reruns of the same query set), traced to exactly this zero-slack setting.
        target_ef = min(len(self._labels), max(k + 50, int(k * 1.2)))
        if target_ef != self._ef:
            self._ef = target_ef
            self.index.set_ef(self._ef)
        try:
            labels, dists = self.index.knn_query(v.reshape(1, -1), k=k)
        except RuntimeError:
            # Retry once with the maximum possible ef (the whole index). Does NOT always work --
            # verified empirically that some specific query vectors still fail even at
            # ef=len(index) against tier_100k's ~518K-vector index (7/301 tested columns): this
            # is a genuine limit of the graph's CONSTRUCTION-time parameters (ef_construction=200,
            # M=32 -- each node's own neighbor list was built exploring only ~200 candidates), not
            # something any query-time ef can fully recover from when k is a large multiple of
            # ef_construction (k=5000 is 25x). Raising ef_construction/M would need a slower,
            # costlier index rebuild -- out of scope for a query-time adapter fix.
            self.index.set_ef(len(self._labels))
            self._ef = len(self._labels)
            try:
                labels, dists = self.index.knn_query(v.reshape(1, -1), k=k)
            except RuntimeError:
                # Still fails: shrink k until hnswlib can actually fill it, rather than crash the
                # caller. Halving is a bounded, deterministic degrade (<=~19 attempts even from
                # k=500000) -- returns fewer than top_n candidates for this one query in this rare
                # case, not a correctness issue (every caller already treats top_n as a breadth
                # cap, never an exact count), just a smaller-than-requested breadth.
                lo_k = k
                while lo_k > 1:
                    lo_k //= 2
                    try:
                        labels, dists = self.index.knn_query(v.reshape(1, -1), k=lo_k)
                        break
                    except RuntimeError:
                        continue
                else:
                    return []
        out: List[Tuple[str, int, float]] = []
        for lbl, d in zip(labels[0], dists[0]):
            sim = 1.0 - float(d)  # cosine space: distance == 1 - cos_sim
            if sim > self.sigma:
                table, attr_idx = self._labels[int(lbl)]
                out.append((table, attr_idx, sim))
        return out


class ExactScanRetriever(object):
    """``SemanticRetriever`` via brute-force cosine similarity over every
    column. The recall oracle for ``HnswRetriever`` -- exact, not
    approximate, and only tractable because santos is small (6322 columns).
    Not intended to be shipped as a query-time retriever at benchmark scale.
    """

    def __init__(self, vectors: Dict[str, np.ndarray], sigma: float = DEFAULT_SIGMA) -> None:
        self.sigma = sigma
        data, self._labels = _flatten(vectors)
        self._matrix = data
        self._norms = np.linalg.norm(data, axis=1) if data.size else np.zeros(0)

    def query(self, vec: np.ndarray, top_n: int) -> List[Tuple[str, int, float]]:
        if not self._labels or top_n <= 0:
            return []
        v = np.asarray(vec, dtype=np.float32)
        vn = np.linalg.norm(v)
        if vn == 0.0:
            return []
        sims = (self._matrix @ v) / (self._norms * vn)
        # exact top_n nearest by similarity, descending, then threshold-filter
        order = np.argsort(-sims)[:top_n]
        out: List[Tuple[str, int, float]] = []
        for idx in order:
            s = float(sims[idx])
            if s > self.sigma:
                table, attr_idx = self._labels[int(idx)]
                out.append((table, attr_idx, s))
        return out
