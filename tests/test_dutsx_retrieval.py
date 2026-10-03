"""Phase B tests: ``SemanticRetriever``/``OverlapFilter`` adapters and §7.3
combination (PLAN-integration.md §4 Phase B).

Runs against real santos data at ``/u6/bkassaie/starmie_fair`` (read-only
upstream, CLAUDE.md), matching the Phase A/C convention -- skips cleanly if
that data isn't mounted at the expected path. ``ExactScanRetriever`` is used
throughout as the independent recall oracle for ``HnswRetriever``, the same
"independent oracle" pattern ``CsvSynopsis`` plays for
``MetadataStoreSynopsis`` in Phase A.

Load-bearing tests:

* ``test_hnsw_recall_vs_exact_scan_on_santos`` -- the actual recall number,
  not an assumption.
* ``test_l2_theta_dis_equivalent_to_cosine_sigma_exact_brute_force`` -- the
  I3 identity (``theta_dis = sqrt(2-2*sigma)`` selects the same neighbours as
  cosine ``> sigma``, after L2-normalizing) checked exactly, without ANN
  approximation in the way.
* ``test_l2_vs_cosine_hnsw_indexes_agree_closely`` -- the same check routed
  through two separately-built hnswlib indexes (``space='l2'`` over
  normalized vectors vs ``space='cosine'`` over raw vectors), which is a
  fair test of "does the identity survive real ANN search", not just the
  underlying math.
* ``test_table_entering_via_different_attributes_yields_no_pair`` -- §7.3's
  explicit, easy-to-get-backwards requirement: D_sem via A1 + D_ovl via A2
  (different attribute, same table) must NOT survive the pair-level
  intersection.
"""
import os
import random
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest

from dutsx.adapters.overlap import InvertedIndexOverlap, NullOverlap
from dutsx.adapters.semantic import (DEFAULT_SIGMA, ExactScanRetriever,
                                       HnswRetriever)
from dutsx.adapters.synopsis import CsvSynopsis
from dutsx.adapters.unionability import load_starmie_vectors
from dutsx.ports import OverlapFilter, SemanticRetriever
from dutsx.registry import build, build_from_config
from dutsx.retrieval import RetrievedCandidate, retrieve_candidates

STARMIE_FAIR_ROOT = "/u6/bkassaie/starmie_fair"
SANTOS_ROOT = os.path.join(STARMIE_FAIR_ROOT, "data", "santos")
DATALAKE_DIR = os.path.join(SANTOS_ROOT, "datalake")
VECTORS_DIR = os.path.join(SANTOS_ROOT, "vectors")
DATALAKE_VEC_PKL = os.path.join(VECTORS_DIR, "cl_datalake_drop_col_tfidf_entity_column_0.pkl")
QUERY_VEC_PKL = os.path.join(VECTORS_DIR, "cl_query_drop_col_tfidf_entity_column_0.pkl")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(DATALAKE_DIR) and os.path.isfile(DATALAKE_VEC_PKL)),
    reason="starmie_fair santos data not available at the expected read-only path",
)


@pytest.fixture(scope="module")
def datalake_vectors() -> Dict[str, np.ndarray]:
    return load_starmie_vectors(DATALAKE_VEC_PKL)


@pytest.fixture(scope="module")
def query_vectors() -> Dict[str, np.ndarray]:
    return load_starmie_vectors(QUERY_VEC_PKL)


@pytest.fixture(scope="module")
def hnsw_retriever(datalake_vectors):
    return HnswRetriever(datalake_vectors, sigma=DEFAULT_SIGMA)


@pytest.fixture(scope="module")
def exact_retriever(datalake_vectors):
    return ExactScanRetriever(datalake_vectors, sigma=DEFAULT_SIGMA)


@pytest.fixture(scope="module")
def datalake_files():
    return sorted(os.listdir(DATALAKE_DIR))


@pytest.fixture(scope="module")
def small_table_sample(datalake_files):
    """A fixed subset, not all 550 -- InvertedIndexOverlap has to read every
    CSV once at build time (pandas), and 550/550 already costs ~10s (Phase A
    territory); 80 tables is representative and keeps this module's own
    fixtures well under the "no test over ~30s" bar."""
    rng = random.Random(7)
    return sorted(rng.sample(datalake_files, 80))


@pytest.fixture(scope="module")
def csv_synopsis(small_table_sample):
    return CsvSynopsis([DATALAKE_DIR], theta_cat=50)


@pytest.fixture(scope="module")
def inverted_index(csv_synopsis, small_table_sample):
    return InvertedIndexOverlap(csv_synopsis, small_table_sample)


@pytest.fixture(scope="module")
def null_overlap(csv_synopsis, small_table_sample):
    return NullOverlap(csv_synopsis, small_table_sample)


# ---------------------------------------------------------------------------
# Registry: config-driven selection
# ---------------------------------------------------------------------------

def test_registry_builds_hnsw_adapter(datalake_vectors):
    adapter = build("semantic", "hnsw", vectors=datalake_vectors)
    assert isinstance(adapter, HnswRetriever)


def test_registry_builds_exact_scan_adapter(datalake_vectors):
    adapter = build("semantic", "exact_scan", vectors=datalake_vectors)
    assert isinstance(adapter, ExactScanRetriever)


def test_registry_builds_inverted_index_adapter(csv_synopsis, small_table_sample):
    adapter = build("overlap", "inverted_index", synopsis=csv_synopsis, tables=small_table_sample)
    assert isinstance(adapter, InvertedIndexOverlap)


def test_registry_builds_null_overlap_adapter(csv_synopsis, small_table_sample):
    adapter = build("overlap", "null", synopsis=csv_synopsis, tables=small_table_sample)
    assert isinstance(adapter, NullOverlap)


def test_registry_config_driven_semantic_selection_is_a_string_change(datalake_vectors):
    cfg = {"adapters": {"semantic": "hnsw"}}
    a = build_from_config(cfg, "semantic", vectors=datalake_vectors)
    cfg["adapters"]["semantic"] = "exact_scan"
    b = build_from_config(cfg, "semantic", vectors=datalake_vectors)
    assert isinstance(a, HnswRetriever)
    assert isinstance(b, ExactScanRetriever)


def test_registry_config_driven_overlap_selection_is_a_string_change(csv_synopsis, small_table_sample):
    cfg = {"adapters": {"overlap": "inverted_index"}}
    a = build_from_config(cfg, "overlap", synopsis=csv_synopsis, tables=small_table_sample)
    cfg["adapters"]["overlap"] = "null"
    b = build_from_config(cfg, "overlap", synopsis=csv_synopsis, tables=small_table_sample)
    assert isinstance(a, InvertedIndexOverlap)
    assert isinstance(b, NullOverlap)


def test_unionability_registry_untouched_by_this_phase():
    """Phase B is additive-only per the coordinator's instructions: Phase C's
    UNIONABILITY_REGISTRY must still have its three entries."""
    from dutsx.registry import UNIONABILITY_REGISTRY
    assert set(UNIONABILITY_REGISTRY) == {"pinned_match", "starmie_verify", "constant"}


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_hnsw_retriever_satisfies_semantic_retriever_protocol(hnsw_retriever):
    assert isinstance(hnsw_retriever, SemanticRetriever)


def test_exact_scan_retriever_satisfies_semantic_retriever_protocol(exact_retriever):
    assert isinstance(exact_retriever, SemanticRetriever)


def test_inverted_index_satisfies_overlap_filter_protocol(inverted_index):
    assert isinstance(inverted_index, OverlapFilter)


def test_null_overlap_satisfies_overlap_filter_protocol(null_overlap):
    assert isinstance(null_overlap, OverlapFilter)


# ---------------------------------------------------------------------------
# HNSW recall vs ExactScanRetriever (Phase B's headline accept criterion)
# ---------------------------------------------------------------------------

def test_santos_datalake_column_count_is_6322(datalake_vectors):
    total = sum(arr.shape[0] for arr in datalake_vectors.values())
    assert len(datalake_vectors) == 550
    assert total == 6322


def test_hnsw_recall_vs_exact_scan_on_santos(hnsw_retriever, exact_retriever, query_vectors):
    """Real recall numbers, over every santos query column (all 50 query
    tables) at top_n=100, sigma=0.6 -- not a sample, since santos is small
    enough that exact scan costs nothing. Recorded numbers (verified while
    writing this test): 236 query columns produce a non-empty exact-scan
    result; mean recall ~0.995, min recall ~0.94 over those. Pinned with a
    conservative floor so the test fails loudly on a real regression rather
    than on ANN nondeterminism."""
    recalls = []
    n_queries_with_results = 0
    for table, arr in query_vectors.items():
        for j in range(arr.shape[0]):
            vec = arr[j]
            exact_result = exact_retriever.query(vec, top_n=100)
            if not exact_result:
                continue
            n_queries_with_results += 1
            hnsw_result = hnsw_retriever.query(vec, top_n=100)
            exact_set = {(t, a) for t, a, s in exact_result}
            hnsw_set = {(t, a) for t, a, s in hnsw_result}
            recalls.append(len(hnsw_set & exact_set) / len(exact_set))

    assert n_queries_with_results > 200  # sanity: sigma=0.6 isn't starving the test
    mean_recall = sum(recalls) / len(recalls)
    min_recall = min(recalls)
    print(f"\nHNSW recall vs exact scan: n={len(recalls)} mean={mean_recall:.4f} min={min_recall:.4f}")
    assert mean_recall >= 0.95
    assert min_recall >= 0.85


def test_exact_scan_results_are_sorted_descending_by_similarity(exact_retriever, query_vectors):
    table, arr = next(iter(query_vectors.items()))
    result = exact_retriever.query(arr[0], top_n=50)
    sims = [s for _, _, s in result]
    assert sims == sorted(sims, reverse=True)


def test_hnsw_results_all_above_sigma(hnsw_retriever, query_vectors):
    for table, arr in list(query_vectors.items())[:10]:
        for j in range(arr.shape[0]):
            result = hnsw_retriever.query(arr[j], top_n=100)
            assert all(sim > hnsw_retriever.sigma for _, _, sim in result)


def test_exact_scan_results_all_above_sigma(exact_retriever, query_vectors):
    for table, arr in list(query_vectors.items())[:10]:
        for j in range(arr.shape[0]):
            result = exact_retriever.query(arr[j], top_n=100)
            assert all(sim > exact_retriever.sigma for _, _, sim in result)


def test_hnsw_retriever_empty_top_n_returns_empty(hnsw_retriever, query_vectors):
    vec = next(iter(query_vectors.values()))[0]
    assert hnsw_retriever.query(vec, top_n=0) == []


# ---------------------------------------------------------------------------
# I3: theta_dis = sqrt(2 - 2*sigma) selects the same neighbours as cosine
# > sigma, after L2-normalizing at index time. The stored vectors are NOT
# normalized (sampled norm ~9.75, verified below), so this is real work.
# ---------------------------------------------------------------------------

def test_stored_vectors_are_not_l2_normalized(datalake_vectors):
    """Confirms the premise the equivalence check below is actually testing:
    if the vectors already had unit norm, L2-vs-cosine equivalence would be
    a triviality, not a real reconciliation of §7.1 vs §6.2 (I3)."""
    sample = []
    for arr in list(datalake_vectors.values())[:20]:
        for row in arr:
            sample.append(np.linalg.norm(row))
    mean_norm = float(np.mean(sample))
    assert mean_norm > 2.0  # far from 1.0; sampled ~9.75 per PLAN-integration.md §6
    assert not np.allclose(sample, 1.0)


def test_l2_theta_dis_equivalent_to_cosine_sigma_exact_brute_force(
    datalake_vectors, query_vectors
):
    """The I3 identity, checked exactly (no ANN in the way): for every
    query column against every datalake column, L2-normalize both, then
    verify (a) the algebraic identity ||a-b||^2 == 2 - 2*cos(a,b) holds to
    float precision, and (b) thresholding squared-L2 at
    theta_dis^2 = 2 - 2*sigma selects EXACTLY the same set as thresholding
    cosine similarity at sigma. This isolates the mathematical claim from
    hnswlib's approximate search, which is checked separately below."""
    sigma = DEFAULT_SIGMA
    theta_dis_sq = 2.0 - 2.0 * sigma

    labels: List[Tuple[str, int]] = []
    rows: List[np.ndarray] = []
    for table, arr in datalake_vectors.items():
        for j in range(arr.shape[0]):
            v = arr[j]
            n = np.linalg.norm(v)
            if n == 0:
                continue
            rows.append(v / n)
            labels.append((table, j))
    normed = np.vstack(rows).astype(np.float64)

    rng = random.Random(2)
    query_tables = list(query_vectors.keys())
    sample_tables = rng.sample(query_tables, 15)

    total_checked = 0
    max_identity_err = 0.0
    mismatched_sets = 0
    for table in sample_tables:
        for j in range(query_vectors[table].shape[0]):
            vec = query_vectors[table][j].astype(np.float64)
            n = np.linalg.norm(vec)
            if n == 0:
                continue
            vecn = vec / n

            cos_sims = normed @ vecn
            sq_l2 = np.sum((normed - vecn) ** 2, axis=1)

            identity_err = np.max(np.abs(sq_l2 - (2.0 - 2.0 * cos_sims)))
            max_identity_err = max(max_identity_err, identity_err)

            set_cos = set(np.where(cos_sims > sigma)[0].tolist())
            set_l2 = set(np.where(sq_l2 < theta_dis_sq)[0].tolist())
            total_checked += 1
            if set_cos != set_l2:
                mismatched_sets += 1

    assert total_checked > 100
    print(f"\nI3 identity: {total_checked} queries, max |err|={max_identity_err:.2e}, "
          f"mismatched threshold sets={mismatched_sets}")
    assert max_identity_err < 1e-5
    assert mismatched_sets == 0


def test_l2_vs_cosine_hnsw_indexes_agree_closely(datalake_vectors, query_vectors):
    """The same I3 check routed through two REAL, separately-built hnswlib
    indexes: space='l2' over explicitly L2-normalized vectors (thresholded
    at theta_dis = sqrt(2-2*sigma)) vs space='cosine' over the raw,
    unnormalized vectors (thresholded at sigma) -- the adapter this repo
    ships. ANN search on two independently-constructed graphs introduces a
    small amount of divergence even though the underlying math is exactly
    equivalent (previous test); this test quantifies that divergence rather
    than assuming it away, and confirms it's small enough that shipping
    space='cosine' (simpler, no normalization step) is a safe substitute for
    the paper's literal Euclidean-distance description."""
    import hnswlib

    sigma = DEFAULT_SIGMA
    theta_dis_sq = 2.0 - 2.0 * sigma

    labels: List[Tuple[str, int]] = []
    rows: List[np.ndarray] = []
    for table, arr in datalake_vectors.items():
        for j in range(arr.shape[0]):
            v = arr[j]
            if np.linalg.norm(v) == 0:
                continue
            rows.append(v)
            labels.append((table, j))
    raw = np.vstack(rows).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normed = (raw / norms).astype(np.float32)
    dim = raw.shape[1]

    idx_cos = hnswlib.Index(space="cosine", dim=dim)
    idx_cos.init_index(max_elements=len(labels), ef_construction=200, M=32, random_seed=42)
    idx_cos.add_items(raw, np.arange(len(labels)))
    idx_cos.set_ef(150)

    idx_l2 = hnswlib.Index(space="l2", dim=dim)
    idx_l2.init_index(max_elements=len(labels), ef_construction=200, M=32, random_seed=42)
    idx_l2.add_items(normed, np.arange(len(labels)))
    idx_l2.set_ef(150)

    rng = random.Random(1)
    sample_tables = rng.sample(list(query_vectors.keys()), 20)

    jaccards = []
    for table in sample_tables:
        for j in range(query_vectors[table].shape[0]):
            vec = query_vectors[table][j].astype(np.float32)
            n = np.linalg.norm(vec)
            if n == 0:
                continue
            vec_n = (vec / n).astype(np.float32)

            lab_c, dist_c = idx_cos.knn_query(vec.reshape(1, -1), k=100)
            lab_l, dist_l = idx_l2.knn_query(vec_n.reshape(1, -1), k=100)

            set_cos = {int(l) for l, d in zip(lab_c[0], dist_c[0]) if (1.0 - float(d)) > sigma}
            set_l2 = {int(l) for l, d in zip(lab_l[0], dist_l[0]) if float(d) < theta_dis_sq}

            union = set_cos | set_l2
            if union:
                jaccards.append(len(set_cos & set_l2) / len(union))

    assert len(jaccards) > 100
    mean_j = sum(jaccards) / len(jaccards)
    print(f"\nL2-vs-cosine HNSW agreement: n={len(jaccards)} mean_jaccard={mean_j:.4f} "
          f"min_jaccard={min(jaccards):.4f}")
    assert mean_j >= 0.95


# ---------------------------------------------------------------------------
# InvertedIndexOverlap / NullOverlap correctness
# ---------------------------------------------------------------------------

def _brute_force_overlap(csv_synopsis, tables, M):
    out = {}
    for table in tables:
        for attr in csv_synopsis.categorical_attrs(table):
            dist = csv_synopsis.distribution(table, attr)
            ov = sum(1 for v in M if v in dist)
            if ov > 0:
                out[(table, attr)] = ov
    return out


def test_inverted_index_matches_brute_force(csv_synopsis, small_table_sample, inverted_index):
    # Build M from real values actually present, so the probe isn't trivially empty.
    rng = random.Random(9)
    table = rng.choice(small_table_sample)
    attrs = csv_synopsis.categorical_attrs(table)
    assert attrs, "fixture table unexpectedly has no categorical attrs"
    attr = rng.choice(attrs)
    dist = csv_synopsis.distribution(table, attr)
    M = set(list(dist.keys())[:3])

    got = inverted_index.query(M)
    expected = _brute_force_overlap(csv_synopsis, small_table_sample, M)
    assert got == expected


def test_inverted_index_empty_M_returns_empty(inverted_index):
    assert inverted_index.query(set()) == {}


def test_inverted_index_unknown_value_returns_empty(inverted_index):
    assert inverted_index.query({"__definitely_not_a_real_value__"}) == {}


def test_null_overlap_returns_every_indexed_pair_regardless_of_M(
    null_overlap, csv_synopsis, small_table_sample
):
    expected_pairs = {
        (t, a) for t in small_table_sample for a in csv_synopsis.categorical_attrs(t)
    }
    got_empty_M = set(null_overlap.query(set()).keys())
    got_nonempty_M = set(null_overlap.query({"some_value", "another"}).keys())
    assert got_empty_M == expected_pairs
    assert got_nonempty_M == expected_pairs


# ---------------------------------------------------------------------------
# §7.3 combination: dutsx/retrieval.py -- pair-level intersection, then
# per-table argmax sim. Synthetic stub retrievers, designed rather than
# searched (the repo's convention for load-bearing regression witnesses).
# ---------------------------------------------------------------------------

class _StubSemantic(object):
    def __init__(self, results: List[Tuple[str, int, float]]):
        self._results = results

    def query(self, vec, top_n):
        return list(self._results)[:top_n]


class _StubOverlap(object):
    def __init__(self, results: Dict[Tuple[str, int], int]):
        self._results = results

    def query(self, M: Set[str]):
        return dict(self._results)


def test_table_entering_via_different_attributes_yields_no_pair():
    """§7.3's explicit requirement: table t1 clears the semantic filter via
    attribute A1 (index 0) but the overlap filter only reports it via a
    DIFFERENT attribute A2 (index 1). No (table, attr) pair is common to
    both, so t1 must not survive at all -- even though, table-wise, both
    filters "found" it."""
    semantic = _StubSemantic([("t1", 0, 0.90)])
    overlap = _StubOverlap({("t1", 1): 3})

    d, telemetry = retrieve_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)

    assert d == []
    assert telemetry.n_sem == 1
    assert telemetry.n_ovl == 1
    assert telemetry.n_pair == 0
    assert telemetry.n_d == 0


def test_table_with_matching_pair_survives():
    """Sanity converse of the above: same attribute index in both -> survives."""
    semantic = _StubSemantic([("t1", 0, 0.90)])
    overlap = _StubOverlap({("t1", 0): 2})

    d, telemetry = retrieve_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)

    assert d == [RetrievedCandidate(table="t1", attr=0, sim=0.90, overlap=2)]
    assert telemetry == (1, 1, 1, 1)


def test_table_with_two_qualifying_attributes_keeps_only_the_higher_sim_one():
    """Table t1 has two attributes that both clear both filters. Exactly one
    survives the per-table reduction: the higher-sim pair."""
    semantic = _StubSemantic([("t1", 0, 0.70), ("t1", 1, 0.95)])
    overlap = _StubOverlap({("t1", 0): 5, ("t1", 1): 1})

    d, telemetry = retrieve_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)

    assert d == [RetrievedCandidate(table="t1", attr=1, sim=0.95, overlap=1)]
    assert telemetry.n_pair == 2  # both pairs cleared the intersection...
    assert telemetry.n_d == 1     # ...but only one table-level candidate survives


def test_multiple_tables_each_keep_their_own_argmax():
    semantic = _StubSemantic([
        ("t1", 0, 0.70), ("t1", 1, 0.95),
        ("t2", 0, 0.80),
    ])
    overlap = _StubOverlap({("t1", 0): 1, ("t1", 1): 1, ("t2", 0): 1})

    d, telemetry = retrieve_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)

    by_table = {c.table: c for c in d}
    assert set(by_table) == {"t1", "t2"}
    assert by_table["t1"].attr == 1
    assert by_table["t1"].sim == 0.95
    assert by_table["t2"].attr == 0
    assert telemetry.n_sem == 3
    assert telemetry.n_pair == 3
    assert telemetry.n_d == 2


def test_overlap_zero_size_pair_excluded():
    """A pair with overlap 0 should not even appear in D_ovl (both
    InvertedIndexOverlap and the port's semantics: only surviving pairs are
    returned), so it can never reach D_pair regardless of D_sem."""
    semantic = _StubSemantic([("t1", 0, 0.90)])
    overlap = _StubOverlap({})  # nothing survives the overlap filter at all

    d, telemetry = retrieve_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)

    assert d == []
    assert telemetry.n_ovl == 0
    assert telemetry.n_pair == 0


def test_telemetry_fields_are_logged_per_query_real_data(
    hnsw_retriever, inverted_index, csv_synopsis, small_table_sample, query_vectors,
    datalake_vectors,
):
    """Runs retrieve_candidates against REAL adapters (HnswRetriever +
    InvertedIndexOverlap) over real santos data, and checks all four
    telemetry counts are present and internally consistent -- the Phase B
    accept criterion 'log |D_sem|, |D_ovl|, |D_pair|, |D| per query'.

    An arbitrary query vector picked from a different query table (as in the
    other telemetry checks) is legitimate but frequently yields an empty
    D_pair by chance -- that's a correct, uninteresting answer, not a bug.
    To also demonstrate the non-empty, doing-real-work case with real data,
    this test additionally queries with a table's OWN column vector as the
    probe and M drawn from that SAME table/attribute's own values: cosine
    similarity to itself is 1.0 (>> sigma, so it survives D_sem), and M's
    values are drawn from that same column's domain (so overlap >= 1,
    surviving D_ovl) -- guaranteeing at least that one pair reaches D_pair
    and D, without hand-waving the numbers.
    """
    rng = random.Random(11)
    table = rng.choice(small_table_sample)
    attrs = csv_synopsis.categorical_attrs(table)
    assert attrs
    attr = rng.choice(attrs)
    dist = csv_synopsis.distribution(table, attr)
    M = set(list(dist.keys())[:2])

    q_table, q_arr = next(iter(query_vectors.items()))
    query_vec = q_arr[0]

    d, telemetry = retrieve_candidates(hnsw_retriever, inverted_index, query_vec, M, top_n=100)

    print(f"\nretrieval telemetry (arbitrary probe): |D_sem|={telemetry.n_sem} "
          f"|D_ovl|={telemetry.n_ovl} |D_pair|={telemetry.n_pair} |D|={telemetry.n_d}")

    assert telemetry.n_pair <= telemetry.n_sem
    assert telemetry.n_pair <= telemetry.n_ovl
    assert telemetry.n_d <= telemetry.n_pair
    assert telemetry.n_d == len(d)
    # each surviving candidate's table appears at most once
    assert len({c.table for c in d}) == len(d)

    # Self-referencing probe: guaranteed non-empty D_pair.
    self_vec = datalake_vectors[table][attr]

    d2, telemetry2 = retrieve_candidates(hnsw_retriever, inverted_index, self_vec, M, top_n=100)
    print(f"retrieval telemetry (self-referencing probe, table={table!r} attr={attr}): "
          f"|D_sem|={telemetry2.n_sem} |D_ovl|={telemetry2.n_ovl} "
          f"|D_pair|={telemetry2.n_pair} |D|={telemetry2.n_d}")

    assert telemetry2.n_pair >= 1
    assert telemetry2.n_d >= 1
    assert any(c.table == table and c.attr == attr for c in d2)


def test_retrieve_candidates_result_type_is_retrieved_candidate():
    semantic = _StubSemantic([("t1", 0, 0.90)])
    overlap = _StubOverlap({("t1", 0): 2})
    d, _ = retrieve_candidates(semantic, overlap, np.zeros(4), {"v"}, top_n=10)
    assert isinstance(d[0], RetrievedCandidate)
    assert d[0].table == "t1" and d[0].attr == 0 and d[0].overlap == 2
