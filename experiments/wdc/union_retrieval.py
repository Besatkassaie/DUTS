"""Union-mode candidate retrieval -- an experimental variant of ``dutsx.retrieval``
for the WDC scalability study only.

**Why this exists, and why it's not in ``dutsx/``.** ``dutsx.retrieval.retrieve_candidates``
implements Fair_Table_Search_7.pdf §7.3 exactly: ``D = D_sem ∩ D_ovl`` at the ``(table, attr)``
pair level. On WDC, an uncurated 50M-table corpus, that compound condition is rarely survivable --
``RESULTS-wdc.md`` §4 measured workable-query yield (``|D| >= 20``) at only 16.7%-20.0% across
``tier_10k``/``tier_100k`` even with frequency-biased ``M`` selection, because most (query,
literal-value) pairs simply don't have tables that are BOTH semantically similar AND contain the
value. That leaves most queries' Stage 1/Stage 2 either skipped (``|D| < k``) or running on pools too
small to show real timing behavior.

This module answers a narrower, purely-scalability question -- "if the pool were big enough, what
would Stage 1 / unionability scoring / Stage 2 actually cost at these two tier sizes?" -- by relaxing
the pair-level combination from intersection to **union**: ``D = D_sem ∪ D_ovl``. This is a
deliberate, explicit deviation from §7.3's paper-faithful semantics, kept entirely out of
``dutsx/retrieval.py`` (which stays exactly as specified) and out of ``dutsx/runner.py`` (whose
``retrieve_unscored_candidates`` remains the paper-faithful, intersection-based path used everywhere
else). Nothing outside ``experiments/wdc/`` imports this module.

A pair that clears only ONE signal gets a placeholder for the other: ``sim=0.0`` for an overlap-only
pair (never probed by HNSW, so no real cosine similarity exists to report), ``overlap=0`` for a
semantic-only pair (never matched a value in ``M``). Neither placeholder affects downstream
correctness -- ``sim`` is used only for the per-table ``argmax`` tie-break (a real semantic pair still
wins that tie-break over a sim=0.0 placeholder whenever both exist for the same table), and
``UnionabilityScorer.score`` never reads ``sim``/``overlap`` at all (``dutsx/runner.py::_score_pool`` /
``experiments/wdc/union_scalability_study.py::_score`` pass only ``(q_table, c.table, (q_attr_idx,
c.attr))``).
"""
import time
from typing import Dict, List, NamedTuple, Set, Tuple

import numpy as np

from dutsx.ports import OverlapFilter, SemanticRetriever
from dutsx.retrieval import RetrievalTelemetry, RetrievedCandidate
from dutsx.runner import QueryTask, RunnerContext, UnscoredCandidate
from duts.stats import N_of


def retrieve_union_candidates(
    semantic: SemanticRetriever,
    overlap: OverlapFilter,
    query_vec: np.ndarray,
    M: Set[str],
    top_n: int,
) -> Tuple[List[RetrievedCandidate], RetrievalTelemetry]:
    """``D = D_sem ∪ D_ovl`` at the ``(table, attr)`` pair level, then reduced
    to one pair per table by ``argmax sim`` (same reduction rule as the
    intersection path, applied over the larger union set). ``telemetry.n_pair``
    reports the UNION's pair count here, not an intersection count -- same
    field name as ``dutsx.retrieval.RetrievalTelemetry`` (reused, not
    duplicated), different meaning by construction of this function."""
    d_sem = semantic.query(query_vec, top_n)          # [(table, attr, sim)]
    d_ovl = overlap.query(M)                            # {(table, attr): overlap}

    sem_pairs: Dict[Tuple[str, int], float] = {
        (table, attr): sim for (table, attr, sim) in d_sem
    }
    pair_keys = set(sem_pairs) | set(d_ovl)

    best_per_table: Dict[str, RetrievedCandidate] = {}
    for (table, attr) in pair_keys:
        sim = sem_pairs.get((table, attr), 0.0)
        ov = d_ovl.get((table, attr), 0)
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


def retrieve_unscored_candidates_union(
    task: QueryTask, ctx: RunnerContext,
) -> "tuple[List[UnscoredCandidate], int, int, RetrievalTelemetry]":
    """Union-mode counterpart of ``dutsx.runner.retrieve_unscored_candidates``
    -- identical N_i/n_i lookup and ``n_i=0`` drop (C5), only the retrieval
    call differs (``retrieve_union_candidates`` instead of
    ``dutsx.retrieval.retrieve_candidates``)."""
    q_vecs = ctx.query_vectors.get(task.q_table)
    if q_vecs is None:
        raise KeyError(f"no query vectors for {task.q_table!r}")
    q_attr_idx = task.attr if isinstance(task.attr, int) else int(task.attr)
    if not (0 <= q_attr_idx < len(q_vecs)):
        raise IndexError(
            f"attr {task.attr!r} out of range for query {task.q_table!r} "
            f"({len(q_vecs)} columns)"
        )
    query_vec = q_vecs[q_attr_idx]

    retrieved, retrieval_telemetry = retrieve_union_candidates(
        ctx.semantic, ctx.overlap, query_vec, task.M, task.top_n
    )

    n_Q = ctx.synopsis.n_rows(task.q_table)
    N_Q = N_of(ctx.synopsis.distribution(task.q_table, task.attr), task.M)

    D: List[UnscoredCandidate] = []
    for rc in retrieved:
        n_i = ctx.synopsis.n_rows(rc.table)
        if n_i == 0:
            continue  # C5: drop n_i=0 candidates
        N_i = N_of(ctx.synopsis.distribution(rc.table, rc.attr), task.M)
        D.append(UnscoredCandidate(table=rc.table, N=N_i, n=n_i, attr=rc.attr))

    return D, N_Q, n_Q, retrieval_telemetry


def _reduce_to_unscored(retrieved, synopsis, M) -> List[UnscoredCandidate]:
    D: List[UnscoredCandidate] = []
    for rc in retrieved:
        n_i = synopsis.n_rows(rc.table)
        if n_i == 0:
            continue  # C5: drop n_i=0 candidates
        N_i = N_of(synopsis.distribution(rc.table, rc.attr), M)
        D.append(UnscoredCandidate(table=rc.table, N=N_i, n=n_i, attr=rc.attr))
    return D


class CombinedRetrievalTiming(NamedTuple):
    """Per-phase breakdown -- what the user asked for explicitly: the
    intersection/union COMBINE step timed separately from the shared probe
    and from the (variant-specific) N_i/n_i lookup, instead of one lumped
    number. ``probe_time_s`` is paid once regardless of combine rule (it's
    identical work either way -- one HNSW query, one overlap query);
    ``*_combine_time_s`` is the set-combine + per-table ``argmax`` reduction;
    ``*_lookup_time_s`` is the synopsis ``N_i``/``n_i`` lookup over that
    variant's OWN reduced candidate set (not reusable between variants -- see
    the function docstring on why the two can pick different attrs)."""
    probe_time_s: float
    union_combine_time_s: float
    intersection_combine_time_s: float
    union_lookup_time_s: float
    intersection_lookup_time_s: float


def retrieve_unscored_candidates_union_and_intersection(
    task: QueryTask, ctx: RunnerContext, dedupe_by_table: bool = True,
) -> "tuple[List[UnscoredCandidate], List[UnscoredCandidate], int, int, RetrievalTelemetry, RetrievalTelemetry, CombinedRetrievalTiming]":
    """Probes ``semantic``/``overlap`` exactly ONCE, then derives BOTH the
    union-reduced and intersection-reduced candidate sets from that single
    probe -- the fix for a real measurement bug found by inspection: an
    earlier version of ``union_scalability_study.py`` called
    ``retrieve_unscored_candidates`` (intersection) AND
    ``retrieve_unscored_candidates_union`` back-to-back inside the same timed
    block, which re-ran the expensive HNSW probe and the overlap posting-list
    scan TWICE and reported their combined cost as "the union path's
    retrieval time" -- wrong, since the caller only needs one of the two
    reduced sets for the actual experiment; the other was purely a
    diagnostic comparison stat. Retrieval (semantic + overlap probing)
    genuinely happens once; intersection vs. union is a cheap combine step
    over the SAME two probed sets, not a second probe.

    ``dedupe_by_table`` (default ``True``, the paper-faithful §7.3 behavior):
    when ``True``, each physical table contributes at most one candidate --
    all ``(table, attr)`` pairs surviving the combine rule are reduced to a
    single ``argmax sim`` pair per table. When ``False`` -- a deliberate,
    WDC-scalability-**stress-test-only** relaxation requested explicitly for
    this experiment, not a general capability -- that per-table reduction is
    skipped entirely: every surviving ``(table, attr)`` pair becomes its own
    candidate, so a table with 5 matching columns contributes 5 rows to
    ``D`` instead of 1. This exists purely to see how Stage 1/scoring/Stage 2
    behave against pools an order of magnitude larger than the per-table-
    deduped ones; it is not a more "correct" retrieval mode and must never
    become the default anywhere outside this stress-test call site.

    Returns ``(D_union, D_intersection, N_Q, n_Q, telemetry_union,
    telemetry_intersection, timing)`` -- ``timing`` is a
    ``CombinedRetrievalTiming`` breaking the "which operator costs what"
    question down explicitly (probe / combine / lookup, per variant) rather
    than one lumped ``retrieval_time_s``. ``D_intersection``'s own lookup is
    NOT free of cost (the two combine rules can pick a DIFFERENT winning
    attribute for the same table, since the per-table ``argmax sim``
    reduction runs over a different candidate pair set in each case) -- only
    the probe itself is shared/paid-once.
    """
    q_vecs = ctx.query_vectors.get(task.q_table)
    if q_vecs is None:
        raise KeyError(f"no query vectors for {task.q_table!r}")
    q_attr_idx = task.attr if isinstance(task.attr, int) else int(task.attr)
    if not (0 <= q_attr_idx < len(q_vecs)):
        raise IndexError(
            f"attr {task.attr!r} out of range for query {task.q_table!r} "
            f"({len(q_vecs)} columns)"
        )
    query_vec = q_vecs[q_attr_idx]

    t0 = time.perf_counter()
    d_sem = ctx.semantic.query(query_vec, task.top_n)          # ONE semantic probe
    d_ovl = ctx.overlap.query(task.M)                            # ONE overlap probe
    probe_t = time.perf_counter() - t0

    sem_pairs: Dict[Tuple[str, int], float] = {
        (table, attr): sim for (table, attr, sim) in d_sem
    }

    def _reduce(pair_keys):
        if not dedupe_by_table:
            # Stress-test mode: keep every surviving (table, attr) pair as
            # its own candidate -- no per-table collapse.
            return [
                RetrievedCandidate(
                    table=table, attr=attr,
                    sim=sem_pairs.get((table, attr), 0.0),
                    overlap=d_ovl.get((table, attr), 0),
                )
                for (table, attr) in pair_keys
            ]
        best: Dict[str, RetrievedCandidate] = {}
        for (table, attr) in pair_keys:
            sim = sem_pairs.get((table, attr), 0.0)
            ov = d_ovl.get((table, attr), 0)
            current = best.get(table)
            if current is None or sim > current.sim:
                best[table] = RetrievedCandidate(table=table, attr=attr, sim=sim, overlap=ov)
        return list(best.values())

    t0 = time.perf_counter()
    union_keys = set(sem_pairs) | set(d_ovl)
    retrieved_union = _reduce(union_keys)
    union_combine_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    inter_keys = set(sem_pairs) & set(d_ovl)
    retrieved_inter = _reduce(inter_keys)
    inter_combine_t = time.perf_counter() - t0

    n_Q = ctx.synopsis.n_rows(task.q_table)
    N_Q = N_of(ctx.synopsis.distribution(task.q_table, task.attr), task.M)

    t0 = time.perf_counter()
    D_union = _reduce_to_unscored(retrieved_union, ctx.synopsis, task.M)
    union_lookup_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    D_inter = _reduce_to_unscored(retrieved_inter, ctx.synopsis, task.M)
    inter_lookup_t = time.perf_counter() - t0

    telem_union = RetrievalTelemetry(
        n_sem=len(sem_pairs), n_ovl=len(d_ovl), n_pair=len(union_keys), n_d=len(retrieved_union),
    )
    telem_inter = RetrievalTelemetry(
        n_sem=len(sem_pairs), n_ovl=len(d_ovl), n_pair=len(inter_keys), n_d=len(retrieved_inter),
    )
    timing = CombinedRetrievalTiming(
        probe_time_s=probe_t,
        union_combine_time_s=union_combine_t,
        intersection_combine_time_s=inter_combine_t,
        union_lookup_time_s=union_lookup_t,
        intersection_lookup_time_s=inter_lookup_t,
    )
    return D_union, D_inter, N_Q, n_Q, telem_union, telem_inter, timing
