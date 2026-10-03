"""Phase C tests: ``UnionabilityScorer`` adapters (PLAN-integration.md §4;
PLAN-full.md M3, M4).

The load-bearing test here is ``test_pinned_and_starmie_disagree_constructed``:
a deliberately built instance where §6.2-as-specified (pin the *pair*) and
starmie_fair's ``verify_constrained`` (pin the *row*) return different scores.
Without it, M3 would be an untested assertion that the two differ.

Tests needing starmie_fair skip cleanly when it isn't present, matching the
Phase A convention.
"""
import math
import os
import random

import numpy as np
import pytest

from dutsx.adapters.unionability import (DEFAULT_THRESHOLD, ConstantScorer,
                                          PinnedMatchScorer,
                                          StarmieVerifyScorer, cosine_sim,
                                          score_pinned, similarity_matrix)
from dutsx.ports import UnionabilityScorer
from dutsx.registry import build

STARMIE_FAIR_ROOT = "/u6/bkassaie/starmie_fair"
SANTOS_VECTORS = os.path.join(STARMIE_FAIR_ROOT, "data", "santos", "vectors")

needs_starmie = pytest.mark.skipif(
    not os.path.isdir(STARMIE_FAIR_ROOT),
    reason="starmie_fair not available at the expected read-only path",
)


def unit(angle_deg):
    """A 2-D unit vector at ``angle_deg``. Cosine similarity between two such
    vectors is exactly ``cos(a - b)``, which makes a similarity matrix
    designable by hand rather than found by search."""
    r = math.radians(angle_deg)
    return np.array([math.cos(r), math.sin(r)], dtype=float)


# --- ports / registry wiring -------------------------------------------------

def test_all_three_satisfy_the_port():
    assert isinstance(ConstantScorer(), UnionabilityScorer)
    assert isinstance(PinnedMatchScorer({}, {}), UnionabilityScorer)
    assert isinstance(StarmieVerifyScorer({}, {}), UnionabilityScorer)


def test_registry_builds_each_adapter():
    assert isinstance(build("unionability", "constant", value=0.5), ConstantScorer)
    assert isinstance(
        build("unionability", "pinned_match", query_vectors={}, candidate_vectors={}),
        PinnedMatchScorer,
    )
    assert isinstance(
        build("unionability", "starmie_verify", query_vectors={}, candidate_vectors={}),
        StarmieVerifyScorer,
    )


def test_registry_rejects_unknown_adapter():
    with pytest.raises(KeyError):
        build("unionability", "no_such_scorer")


# --- score_pinned semantics --------------------------------------------------

def test_returns_zero_when_pin_fails_threshold():
    """If the pinned (V_D, V_{D_T}) pair itself is below sigma, the tables are
    not unionable on V_D at all -- other attributes agreeing must not rescue
    the score. Mirrors verify_constrained's all-zero-mandatory-row return."""
    q = np.array([unit(0), unit(1)])
    c = np.array([unit(89), unit(1)])  # c[0] is ~orthogonal to q[0]
    assert cosine_sim(q[0], c[0]) < DEFAULT_THRESHOLD
    assert score_pinned(q, c, pin=(0, 0)) == 0.0


def test_pin_only_when_single_attribute_each():
    q = np.array([unit(0)])
    c = np.array([unit(10)])
    expected = cosine_sim(q[0], c[0])
    assert expected > DEFAULT_THRESHOLD
    assert score_pinned(q, c, pin=(0, 0)) == pytest.approx(expected)


def test_pinned_row_and_column_are_both_excluded_from_the_rematch():
    """The pinned column must not be reusable by another query attribute --
    that is the 'excluded from the optimization' half of §6.2 (M3)."""
    # q1 would love to match c0, but c0 is taken by the pin and is gone.
    q = np.array([unit(0), unit(46)])
    c = np.array([unit(45), unit(200)])
    s_pin = cosine_sim(q[0], c[0])
    assert s_pin > DEFAULT_THRESHOLD
    assert cosine_sim(q[1], c[0]) > DEFAULT_THRESHOLD   # tempting but unavailable
    assert cosine_sim(q[1], c[1]) < DEFAULT_THRESHOLD   # the only thing left is unusable
    assert score_pinned(q, c, pin=(0, 0)) == pytest.approx(s_pin)


def test_raises_on_out_of_range_pin():
    q = np.array([unit(0)])
    c = np.array([unit(0)])
    with pytest.raises(ValueError):
        score_pinned(q, c, pin=(5, 0))
    with pytest.raises(ValueError):
        score_pinned(q, c, pin=(0, 5))


def test_similarity_matrix_zeroes_at_or_below_threshold():
    """bounds.py uses strict '>', which this module deliberately matches --
    see PLAN-full.md M3's note on the paper saying '>=' instead."""
    q = np.array([unit(0)])
    c = np.array([unit(0)])
    g = similarity_matrix(q, c, threshold=1.0)   # sim == 1.0, not > 1.0
    assert g[0, 0] == 0.0


def test_unnormalized_raw_sum_not_clamped_to_unit_interval():
    """M4: U is the raw sum of similarities and may exceed 1.0. Table 1's
    'U in [0,1]' is what is wrong in the paper, not the §6.2 formula."""
    q = np.array([unit(0), unit(90)])
    c = np.array([unit(0), unit(90)])
    score = score_pinned(q, c, pin=(0, 0))
    assert score == pytest.approx(2.0)   # two perfect matches
    assert score > 1.0


# --- M3: the constructed disagreement ---------------------------------------

def _m3_witness():
    """Query/candidate vectors where pinning the pair and pinning the row
    genuinely differ.

    Designed, not searched for:
        s(q0,c0) = 0.70  <- the pin
        s(q0,c1) = 0.95  <- what an unconstrained matcher would rather use
        s(q1,c0) = 0.65
        s(q1,c1) = 0.23  <- below sigma, unusable

    §6.2/M3 (pin the pair):  0.70 + (q1 has only c1 left, unusable) = 0.70
    verify_constrained (pin the row): q0->c1 and q1->c0 = 0.95 + 0.65 = 1.60
    """
    q = np.array([unit(0.0), unit(95.03)])
    c = np.array([unit(45.57), unit(18.19)])
    return q, c


def test_m3_witness_has_the_intended_similarity_structure():
    """Pin down the construction itself, so a later edit to the angles can't
    silently turn the disagreement test into a tautology."""
    q, c = _m3_witness()
    assert cosine_sim(q[0], c[0]) == pytest.approx(0.70, abs=1e-3)
    assert cosine_sim(q[0], c[1]) == pytest.approx(0.95, abs=1e-3)
    assert cosine_sim(q[1], c[0]) == pytest.approx(0.65, abs=1e-3)
    assert cosine_sim(q[1], c[1]) < DEFAULT_THRESHOLD


def test_pinned_scores_the_pin_not_the_best_column():
    q, c = _m3_witness()
    assert score_pinned(q, c, pin=(0, 0)) == pytest.approx(0.70, abs=1e-3)


@needs_starmie
def test_pinned_and_starmie_disagree_constructed():
    """THE M3 witness: §6.2-as-specified and verify_constrained differ.

    If this ever starts passing trivially (both equal), M3 has been
    misimplemented -- most likely by letting the matcher re-choose the pinned
    column.
    """
    q, c = _m3_witness()
    vectors_q = {"Q": q}
    vectors_c = {"T": c}

    pinned = PinnedMatchScorer(vectors_q, vectors_c)
    starmie = StarmieVerifyScorer(vectors_q, vectors_c)

    pinned_score = pinned.score("Q", "T", pin=(0, 0))
    starmie_score = starmie.score("Q", "T", pin=(0, 0))

    assert pinned_score == pytest.approx(0.70, abs=1e-3)
    assert starmie_score == pytest.approx(1.60, abs=1e-3)
    assert pinned_score < starmie_score


@needs_starmie
def test_pinned_and_starmie_agree_when_pin_is_the_unconstrained_choice():
    """Converse of the witness: when the pinned column is the one an
    unconstrained matcher would pick anyway, the two must coincide."""
    q = np.array([unit(0.0), unit(90.0)])
    c = np.array([unit(2.0), unit(88.0)])
    vectors_q, vectors_c = {"Q": q}, {"T": c}

    pinned = PinnedMatchScorer(vectors_q, vectors_c).score("Q", "T", pin=(0, 0))
    starmie = StarmieVerifyScorer(vectors_q, vectors_c).score("Q", "T", pin=(0, 0))
    assert pinned == pytest.approx(starmie, abs=1e-9)


@needs_starmie
@pytest.mark.parametrize("seed", range(60))
def test_pinned_never_exceeds_starmie_property(seed):
    """Property: forcing q_i -> t_j optimizes over a strict SUBSET of the
    matchings verify_constrained considers (which only forces q_i to be
    matched *somewhere*), so the pinned score can never be larger.

    A violation would mean the pinned implementation is finding weight the
    unconstrained matcher cannot -- i.e. double-counting or reusing the
    pinned column."""
    rng = np.random.default_rng(seed)
    n_q = int(rng.integers(1, 5))
    n_c = int(rng.integers(1, 5))
    q = rng.normal(size=(n_q, 6))
    c = rng.normal(size=(n_c, 6))
    qi = int(rng.integers(0, n_q))
    tj = int(rng.integers(0, n_c))

    # low threshold so most edges survive and the comparison is non-trivial
    thr = 0.0
    pinned = score_pinned(q, c, (qi, tj), threshold=thr)
    starmie = StarmieVerifyScorer({"Q": q}, {"T": c}, threshold=thr).score(
        "Q", "T", pin=(qi, tj)
    )
    if pinned > 0.0 and starmie > 0.0:
        assert pinned <= starmie + 1e-9


# --- counters + test double --------------------------------------------------

def test_constant_scorer_is_constant_and_counts():
    s = ConstantScorer(value=0.25)
    assert s.score("a", "b", (0, 0)) == 0.25
    assert s.score("x", "y", (1, 2)) == 0.25
    assert s.n_scored == 2


def test_scorer_counts_calls_for_the_efficiency_metric():
    """PLAN-integration.md §3 makes '#unionability computations' the headline
    efficiency claim (should equal |P| = alpha*k, independent of |T|).
    Counting at the scorer is the one place a caller cannot fudge."""
    q = np.array([unit(0)])
    c = np.array([unit(1)])
    s = PinnedMatchScorer({"Q": q}, {"T": c})
    assert s.n_scored == 0
    for _ in range(4):
        s.score("Q", "T", (0, 0))
    assert s.n_scored == 4


def test_unknown_table_raises_rather_than_scoring_zero():
    """A missing table is a wiring bug; silently returning 0.0 would make it
    look like a legitimately non-unionable candidate."""
    s = PinnedMatchScorer({"Q": np.array([unit(0)])}, {})
    with pytest.raises(KeyError):
        s.score("Q", "missing", (0, 0))


# --- real data ---------------------------------------------------------------

@needs_starmie
def test_scores_real_santos_pair():
    q_pkl = os.path.join(SANTOS_VECTORS, "cl_query_drop_col_tfidf_entity_column_0.pkl")
    d_pkl = os.path.join(SANTOS_VECTORS, "cl_datalake_drop_col_tfidf_entity_column_0.pkl")
    if not (os.path.exists(q_pkl) and os.path.exists(d_pkl)):
        pytest.skip("santos vector pickles not available")

    scorer = PinnedMatchScorer.from_starmie_pickles(q_pkl, d_pkl)
    assert len(scorer.query_vectors) == 50
    assert len(scorer.candidate_vectors) == 550

    q_name = sorted(scorer.query_vectors)[0]
    c_name = sorted(scorer.candidate_vectors)[0]
    score = scorer.score(q_name, c_name, pin=(0, 0))
    assert isinstance(score, float)
    assert math.isfinite(score)
    assert score >= 0.0


@needs_starmie
def test_real_pair_scores_are_bounded_by_attribute_count():
    """Sanity bound on real data: U is a sum over a matching, so it cannot
    exceed min(n_q_attrs, n_c_attrs) since every similarity is <= 1."""
    q_pkl = os.path.join(SANTOS_VECTORS, "cl_query_drop_col_tfidf_entity_column_0.pkl")
    d_pkl = os.path.join(SANTOS_VECTORS, "cl_datalake_drop_col_tfidf_entity_column_0.pkl")
    if not (os.path.exists(q_pkl) and os.path.exists(d_pkl)):
        pytest.skip("santos vector pickles not available")

    scorer = PinnedMatchScorer.from_starmie_pickles(q_pkl, d_pkl)
    rng = random.Random(0)
    q_names = sorted(scorer.query_vectors)
    c_names = sorted(scorer.candidate_vectors)

    for _ in range(25):
        q_name = rng.choice(q_names)
        c_name = rng.choice(c_names)
        n_q = len(scorer.query_vectors[q_name])
        n_c = len(scorer.candidate_vectors[c_name])
        score = scorer.score(q_name, c_name, pin=(0, 0))
        assert 0.0 <= score <= min(n_q, n_c) + 1e-9
