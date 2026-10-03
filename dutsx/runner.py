"""End-to-end runner (PLAN-integration.md §3, §4 Phase D).

Composes the verified core (``duts.stage1_dinkelbach``, ``duts.stage2_ilp``)
directly with the Phase A/B/C adapters, WITHOUT calling ``duts.pipeline.run``.

**Why not `duts.pipeline.run`.** It takes ``D: List[CandidateStats]`` with
``U`` already populated for every candidate -- correct for the core's own
tests (where ``D`` is a fixture or synthetic generator and ``U`` is free),
but wrong here: populating ``U`` for every retrieved candidate would run
``|D|`` bipartite matchings, where §4.1's efficiency claim is that only
``alpha*k`` (the pool ``P``, post-Stage-1) ever needs scoring. Composing the
stages by hand instead makes "how many unionability computations happened"
a directly countable, structurally-guaranteed quantity rather than a hope.

**The structural guarantee.** ``retrieve_unscored_candidates`` builds ``D``
as ``List[UnscoredCandidate]`` -- a NamedTuple with NO ``U`` field at all.
Stage 1 (``duts.stage1_dinkelbach``) duck-types over ``(table, N, n)`` only --
never ``.U`` -- so it runs unmodified against ``UnscoredCandidate`` instances.
Stage 2 (``duts.stage2_ilp``) reads ``c.U`` directly, so it CANNOT run on an
``UnscoredCandidate`` -- it fails with ``AttributeError`` immediately, not a
silently-wrong answer. ``_score_pool`` is the ONLY function in this module
that constructs a real ``CandidateStats`` (with a real ``U``), and it is
called on Stage 1's *output* (``P_unscored``, size <= alpha*k), never on
``D``. A reader can see this guarantee by grep: ``_score_pool`` has exactly
one call site, and it is not ``D``.

::

    retrieve            -> D   (UnscoredCandidate: table, N, n, attr; no U)
    Stage 1             -> P_unscored   (duts.stage1_dinkelbach.solve; N_i, n_i only)
    score U on P ONLY   -> P: List[CandidateStats]   <- the efficiency claim, measurable
    Stage 2             -> R   (duts.stage2_ilp.solve)
"""
import csv
import os
import time
from typing import Dict, List, NamedTuple, Optional, Set

import numpy as np

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import F as F_ratio
from duts.stats import N_of, aggregate, delta as delta_of, drop_empty
from duts.types import CandidateStats

from .ports import AttrRef, OverlapFilter, SemanticRetriever, SynopsisSource, UnionabilityScorer
from .retrieval import RetrievalTelemetry, retrieve_candidates


class UnscoredCandidate(NamedTuple):
    """``D`` and ``P_unscored`` -- deliberately has NO ``U`` field (see module
    docstring). Duck-types as a ``CandidateStats`` for Stage 1 (which reads
    only ``.table``/``.N``/``.n``); cannot be passed to Stage 2, which reads
    ``.U``."""
    table: str
    N: int      # exact integer count over M (C2), via duts.stats.N_of
    n: int      # n_rows(table)
    attr: int   # the ONE attribute aligned to V_D (§7.3 argmax) -- the pin's column


class QueryTask(NamedTuple):
    """One query: a query table, its protected attribute ``V_D``, a value
    set ``M``, and the two-stage optimization's parameters (PLAN.md
    ``QuerySpec``, extended with retrieval knobs)."""
    q_table: str
    attr: AttrRef             # V_D -- column index (protected_attributes_santos.csv) or name
    M: Set[str]                 # C2/M2: a genuine value SET; scalar (len(M)==1) is the common case
    F_star: float
    delta: float
    k: int
    alpha: float = 1.0
    include_query: bool = True
    top_n: int = 100            # §7.1 ANN probe width


class RunTelemetry(NamedTuple):
    """Per-query telemetry -- superset of ``duts.types.Telemetry``, plus
    retrieval counts and the headline efficiency metric."""
    retrieval: RetrievalTelemetry
    n_D: int
    n_P: int
    n_R: int
    n_unionability_computations: int   # == len(P); the paper's central efficiency claim, made countable
    F_P: Optional[float]
    F_R: Optional[float]
    delta_R: Optional[float]            # signed F_star - F_R. NOT bounded by query.delta: only the
                                         # floor F_R >= tau is enforced, and C3's unbounded argmax F
                                         # routinely overshoots F_star. Measured on the 2026-08-10
                                         # alpha sweep: 60 of 115 feasible rows have |delta_R| > delta
                                         # (min -0.700, i.e. F_R = 1.0 against F_star = 0.3).
    sum_U: Optional[float]
    dinkelbach_iterations_stage1: int
    retrieval_time_s: Optional[float]   # index probe (semantic + overlap) + N_i/n_i synopsis lookup
    stage1_time_s: Optional[float]      # Dinkelbach, or the C5 clamp when alpha*k >= |D|
    scoring_time_s: Optional[float]     # wall time spent in _score_pool (unionability matchings)
    ilp_time_s: Optional[float]
    lp_precheck_ran: bool
    lp_feasible: Optional[bool]
    lp_time_s: Optional[float]
    effective_alpha: float
    feasible: bool
    infeasibility_cause: Optional[str]  # None | "insufficient_candidates" | "stage2_infeasible"


class RunResult(NamedTuple):
    selected: List[CandidateStats]        # R; [] if infeasible
    pool: List[CandidateStats]             # P, scored; R subseteq P always
    candidates: List[UnscoredCandidate]     # D, unscored; P subseteq D always (as tables)
    telemetry: RunTelemetry


class RunnerContext(NamedTuple):
    """Bundles the four adapters plus the query-side embeddings needed to
    issue a semantic probe (§7.1: probe vector is the query table's OWN
    ``V_D`` column embedding). Built once per benchmark run, reused across
    all ``QueryTask``s."""
    synopsis: SynopsisSource
    semantic: SemanticRetriever
    overlap: OverlapFilter
    unionability: UnionabilityScorer
    query_vectors: Dict[str, np.ndarray]


def _default_telemetry(retrieval: RetrievalTelemetry, **overrides) -> RunTelemetry:
    defaults: Dict = dict(
        retrieval=retrieval, n_D=retrieval.n_d, n_P=0, n_R=0,
        n_unionability_computations=0, F_P=None, F_R=None, delta_R=None, sum_U=None,
        dinkelbach_iterations_stage1=0, retrieval_time_s=None, stage1_time_s=None,
        scoring_time_s=None, ilp_time_s=None, lp_precheck_ran=False, lp_feasible=None,
        lp_time_s=None, effective_alpha=None, feasible=False, infeasibility_cause=None,
    )
    defaults.update(overrides)
    return RunTelemetry(**defaults)


def retrieve_unscored_candidates(
    task: QueryTask, ctx: RunnerContext,
) -> "tuple[List[UnscoredCandidate], int, int, RetrievalTelemetry]":
    """§4.2/§7.3 retrieval, then N_i/n_i lookup per candidate (C2) -- builds
    ``D`` with NO ``U``. Returns ``(D, N_Q, n_Q, retrieval_telemetry)``."""
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

    retrieved, retrieval_telemetry = retrieve_candidates(
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


def _score_pool(
    pool_unscored: List[UnscoredCandidate],
    ctx: RunnerContext,
    q_table: str,
    q_attr_idx: int,
) -> List[CandidateStats]:
    """The ONLY call site of ``UnionabilityScorer.score`` in this module.
    Runs exactly ``len(pool_unscored)`` matchings -- the efficiency claim,
    made measurable via each adapter's own ``n_scored`` counter."""
    scored: List[CandidateStats] = []
    for c in pool_unscored:
        u = ctx.unionability.score(q_table, c.table, (q_attr_idx, c.attr))
        scored.append(CandidateStats(table=c.table, N=c.N, n=c.n, U=u))
    return scored


def run_query(task: QueryTask, ctx: RunnerContext) -> RunResult:
    """retrieve -> Stage 1 -> score(P) -> Stage 2, per the module docstring's
    composition, with an extra scoring step over Stage 1's output only.

    **No C4 feasibility certificate.** It was removed from this path entirely
    on 2026-08-10 at the user's instruction ("I do not want the certificate C4
    to be checked at all"), superseding the 2026-08-09 note that merely
    defaulted it off. There is no flag to turn it back on -- an infeasible
    instance that clears ``|D| >= k`` is reported as ``stage2_infeasible``,
    with no attempt to attribute it to an intrinsic-vs-alpha cause. PLAN.md
    C5's degenerate-cardinality handling is unchanged."""
    t_retr0 = time.perf_counter()
    D, N_Q, n_Q, retrieval_telemetry = retrieve_unscored_candidates(task, ctx)
    retrieval_time_s = time.perf_counter() - t_retr0
    D = drop_empty(D)  # belt and suspenders; retrieve_unscored_candidates already filters n_i=0
    tau = task.F_star - task.delta
    k, alpha, include_query = task.k, task.alpha, task.include_query
    q_attr_idx = task.attr if isinstance(task.attr, int) else int(task.attr)

    if len(D) < k:
        return RunResult(selected=[], pool=[], candidates=D, telemetry=_default_telemetry(
            retrieval_telemetry, n_D=len(D), effective_alpha=alpha,
            retrieval_time_s=retrieval_time_s,
            infeasibility_cause="insufficient_candidates",
        ))

    pool_size = int(alpha * k)
    effective_alpha = alpha
    n_iter_stage1 = 0
    t_s1 = time.perf_counter()
    if pool_size >= len(D):
        # C5: |D| < alpha*k -- clamp P to D, never pad.
        P_unscored = D
        effective_alpha = len(D) / k
        F_P = F_ratio(*aggregate(P_unscored, N_Q, n_Q, include_query))
    else:
        stage1_result = stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, include_query)
        P_unscored = stage1_result.selected
        F_P = stage1_result.objective_value
        n_iter_stage1 = stage1_result.iterations
    stage1_time_s = time.perf_counter() - t_s1

    # ---- the efficiency claim: score U for exactly the tables in P_unscored ----
    t_score0 = time.perf_counter()
    P = _score_pool(P_unscored, ctx, task.q_table, q_attr_idx)
    scoring_time_s = time.perf_counter() - t_score0
    n_unionability_computations = len(P_unscored)

    t1 = time.perf_counter()
    stage2_result = stage2_ilp.solve(P, k, tau, N_Q, n_Q, include_query)
    ilp_time_s = time.perf_counter() - t1
    info = stage2_result.info or {}

    if not stage2_result.feasible:
        return RunResult(selected=[], pool=P, candidates=D, telemetry=_default_telemetry(
            retrieval_telemetry, n_D=len(D), n_P=len(P),
            n_unionability_computations=n_unionability_computations,
            F_P=F_P,
            dinkelbach_iterations_stage1=n_iter_stage1,
            retrieval_time_s=retrieval_time_s, stage1_time_s=stage1_time_s,
            scoring_time_s=scoring_time_s,
            ilp_time_s=ilp_time_s, lp_precheck_ran=info.get("lp_precheck_ran", False),
            lp_feasible=info.get("lp_feasible"), lp_time_s=info.get("lp_time_s"),
            effective_alpha=effective_alpha, infeasibility_cause="stage2_infeasible",
        ))

    R = stage2_result.selected
    F_R = F_ratio(*aggregate(R, N_Q, n_Q, include_query))
    telemetry = _default_telemetry(
        retrieval_telemetry, n_D=len(D), n_P=len(P), n_R=len(R),
        n_unionability_computations=n_unionability_computations,
        F_P=F_P, F_R=F_R, delta_R=delta_of(task.F_star, F_R), sum_U=stage2_result.objective_value,
        dinkelbach_iterations_stage1=n_iter_stage1,
        retrieval_time_s=retrieval_time_s, stage1_time_s=stage1_time_s,
        scoring_time_s=scoring_time_s,
        ilp_time_s=ilp_time_s, lp_precheck_ran=info.get("lp_precheck_ran", False),
        lp_feasible=info.get("lp_feasible"), lp_time_s=info.get("lp_time_s"),
        effective_alpha=effective_alpha, feasible=True, infeasibility_cause=None,
    )
    return RunResult(selected=R, pool=P, candidates=D, telemetry=telemetry)


def load_queries_from_csv(
    csv_path: str,
    query_dir: str,
    k: int,
    alpha: float,
    F_star: float,
    delta: float,
    include_query: bool = True,
    top_n: int = 100,
) -> "tuple[List[QueryTask], List[str]]":
    """``protected_attributes_santos.csv`` -> ``List[QueryTask]``.

    Format (confirmed, PLAN-integration.md §6): ``q_name,
    protected_attribute_id, protected_value``. ``protected_attribute_id`` is
    a **column index** into ``q_name``; ``protected_value`` is a **scalar**,
    wrapped as the singleton set ``M = {protected_value}`` (C2/M2: ``M`` is a
    genuine value set in the API -- scalar is just the common case here, not
    a limitation of ``QueryTask``).

    **Not every row maps to a usable santos query.** Found while building
    this loader (see NOTES.md "Phase D"): the CSV has 96 rows, but only 48
    have a ``q_name`` that is an actual file in ``query_dir``. The other 48
    all end in ``_fair.csv`` and belong to a DIFFERENT benchmark directory
    (``santos3/query/``, not ``santos/query/``) that is never on this path.
    Conversely, 2 of the 50 real ``santos/query/`` files
    (``albums_b.csv``, ``film_locations_in_san_francisco_a.csv``) have no
    row in this CSV at all -- not a parsing artifact, genuinely absent.
    Net: 48 of 50 santos query tables get a ``QueryTask`` here, not 50.

    Rows that don't resolve to a real file are silently skipped (not an
    error -- this is expected, not a data problem) and returned as the
    second element (skipped ``q_name``s) so a caller can report the count
    without treating it as a failure.
    """
    query_files = set(os.listdir(query_dir)) if os.path.isdir(query_dir) else set()
    tasks: List[QueryTask] = []
    skipped: List[str] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_name = row["q_name"]
            if q_name not in query_files:
                skipped.append(q_name)
                continue
            attr = int(row["protected_attribute_id"])
            value = row["protected_value"]
            tasks.append(QueryTask(
                q_table=q_name, attr=attr, M={value},
                F_star=F_star, delta=delta, k=k, alpha=alpha,
                include_query=include_query, top_n=top_n,
            ))
    return tasks, skipped
