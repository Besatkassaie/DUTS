"""``top_n`` x ``alpha`` x ``k`` scalability sweep at widened ``sigma`` (user-requested ad hoc
experiment, 2026-08-13 -- not part of ``PLAN-integration.md``'s phase sequence).

Answers: at a much wider semantic-similarity gate (``sigma=0.30`` vs. the paper-matching default
0.6) and much wider HNSW probe breadths (``top_n`` up to 5000 vs. the earlier tier studies' fixed
200), how do retrieval yield and Stage 1/scoring/Stage 2 cost trend as ``top_n``, ``alpha`` (Stage 1
pool = ``alpha*k``), and ``k`` all move -- for ``tier_10k`` and ``tier_100k``.

Modeled directly on ``union_scalability_study.py::run_tier_study_union`` (same union-mode retrieval,
same per-phase timing breakdown via ``union_retrieval.retrieve_unscored_candidates_union_and_intersection``,
same ``_score``/write_rows shape) with three deliberate structural changes:

1. ``ctx`` is built ONCE per tier, not once per ``top_n``. ``sigma`` only changes ``HnswRetriever``'s
   post-search similarity filter, not the underlying HNSW index structure (``dutsx/adapters/
   semantic.py::HnswRetriever.query`` filters ``sim > self.sigma`` AFTER ``knn_query`` returns) --
   so sweeping ``top_n`` needs no index rebuild, only a different ``query(vec, top_n)`` call per
   query. This avoids paying the (one-time, per-tier) HNSW/inverted-index build cost 6x.
2. ``top_n`` is swept (500..5000); ``select_queries_with_comparison`` is re-run at each ``top_n``
   (its own eligibility/min-posting-size threshold depends on ``top_n``), so each ``top_n`` gets its
   own 30-query selection at the same seed -- not the same 30 queries reused across settings.
3. ``ks``/``alphas`` are both swept (vs. the original two-tuples), producing one row per
   ``(top_n, k, alpha, query)`` cell -- the grid the user asked to see trends over.

**``ef`` correctness note (the reason ``dutsx/adapters/semantic.py::HnswRetriever.query`` was
touched for this experiment):** hnswlib's ``ef`` search-width parameter must be >= the requested
``k`` (here, ``top_n``) for correct/complete results; a stale, too-small ``ef`` doesn't error, it
silently degrades recall (confirmed empirically: ``ef=100`` against ``top_n=200`` in the ORIGINAL
tier studies was already under-provisioned, though those runs never surfaced it because 200 is only
2x over-budget, not the 5-50x this sweep asks for). ``HnswRetriever.query`` now retunes ``ef`` to
match ``top_n`` on every call, so each ``top_n`` setting in this sweep reflects a properly-tuned
search at that breadth rather than an inherited-too-narrow one.

**No embedding inference anywhere in this loop.** ``build_wdc_context`` loads each tier's
precomputed ``<tier>_roberta.pkl`` once (``query_vectors is candidate_vectors``, per
``context.py``'s docstring -- WDC has no query/datalake split); ``ctx.semantic.query()`` is a pure
hnswlib ANN lookup over those already-loaded vectors, so nothing downstream of context-build ever
runs the embedding model. ``embedding_extraction_s`` (the one real inference cost, a separate
offline GPU step) lives only in ``build_times``, reported once per tier, and deliberately never
folded into any per-``(top_n, alpha, k)`` timing field below.
"""
import csv
import json
import os
import time
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import F as F_ratio
from duts.stats import aggregate, drop_empty
from duts.types import CandidateStats

from .context import build_wdc_context, paths_for_tier
from .query_selection import select_queries_with_comparison
from .union_retrieval import retrieve_unscored_candidates_union_and_intersection

DEFAULT_TOP_NS = (500, 1000, 2000, 3000, 4000, 5000)
DEFAULT_KS = (10, 20, 30, 40, 50)
DEFAULT_ALPHAS = (5.0, 10.0, 15.0, 20.0)
DEFAULT_SIGMA = 0.30


class SweepRow(NamedTuple):
    tier: str
    top_n: int
    q_table: str
    attr: int
    k: int
    alpha: float
    tau: float
    n_sem: int
    n_ovl: int
    n_D_union: int
    pool_size_requested: int
    clamped: bool
    feasible: bool
    tau_reached: bool
    n_P: int
    n_R: int
    n_unionability_computations: int
    F_Q: Optional[float]
    F_P: Optional[float]
    F_R: Optional[float]
    sum_U: Optional[float]
    dinkelbach_ran: bool
    dinkelbach_iterations: int
    retrieval_time_s: float           # = probe_time_s + union_combine_time_s + union_lookup_time_s
    probe_time_s: float                # semantic.query + overlap.query -- shared once per (top_n, query)
    union_combine_time_s: float        # set union + per-table argmax reduction
    union_lookup_time_s: float         # N_i/n_i synopsis lookup over D_union
    stage1_time_s: float
    scoring_time_s: float
    stage2_time_s: float
    end_to_end_s: float


def _score(cands, ctx, q_table, q_attr_idx, dedupe_by_table: bool = True) -> List[CandidateStats]:
    """Duplicated from ``dutsx.runner._score_pool`` on purpose -- same precedent as
    ``scalability_study.py``/``union_scalability_study.py``'s own ``_score``.

    ``dedupe_by_table=False`` mirrors ``union_scalability_study.py::_score``'s stress-test id
    fix: several rows in ``cands`` can share the same physical table (different columns) once
    the per-table reduction is skipped, so each gets a synthetic ``"<table>::<attr>"`` id to keep
    Stage 1's table-keyed identity/tie-break well-defined. ``ctx.unionability.score`` is still
    called with the REAL ``c.table``."""
    out = []
    for c in cands:
        u = ctx.unionability.score(q_table, c.table, (q_attr_idx, c.attr))
        cand_id = c.table if dedupe_by_table else "{}::{}".format(c.table, c.attr)
        out.append(CandidateStats(table=cand_id, N=c.N, n=c.n, U=u))
    return out


def run_topn_sweep(
    tier: str,
    top_ns: Sequence[int] = DEFAULT_TOP_NS,
    n_queries: int = 30,
    ks: Sequence[int] = DEFAULT_KS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    F_star: float = 0.3,
    delta: float = 0.15,
    sigma: float = DEFAULT_SIGMA,
    seed: int = 42,
    verbose: bool = True,
    dedupe_by_table: bool = True,
    ctx: object = None,
    build_times: object = None,
) -> Tuple[List[SweepRow], object, Dict[int, dict]]:
    if ctx is None:
        ctx, build_times = build_wdc_context(tier, sigma=sigma)
    if verbose and build_times is not None:
        print(
            "[{}] index build: {:.2f}s over {} tables ({}/{} categorical, ratio {:.1%}), "
            "sigma={}".format(
                tier, build_times.total_index_s, build_times.n_tables,
                build_times.n_columns_categorical, build_times.n_columns_total,
                build_times.categorical_coverage_ratio, sigma,
            ), flush=True,
        )

    tau = F_star - delta
    paths = paths_for_tier(tier)
    tables = sorted(f for f in os.listdir(paths.csv_dir) if f.endswith(".csv"))
    max_k = max(ks)

    rows: List[SweepRow] = []
    selection_reports: Dict[int, dict] = {}

    for top_n in top_ns:
        t_sel = time.perf_counter()
        tasks, selection_report = select_queries_with_comparison(
            ctx, tables, n_queries=n_queries, seed=seed, k=ks[0], alpha=alphas[0],
            F_star=F_star, delta=delta, top_n=top_n,
        )
        selection_reports[top_n] = selection_report
        if verbose:
            print(
                "[{} top_n={}] {} queries selected ({:.1f}s to select)".format(
                    tier, top_n, len(tasks), time.perf_counter() - t_sel,
                ), flush=True,
            )

        n_done, n_skipped = 0, 0
        t_tier_topn = time.perf_counter()
        for base_task in tasks:
            D_union_full, _D_inter, N_Q, n_Q, telem_u, _telem_i, rtiming = \
                retrieve_unscored_candidates_union_and_intersection(
                    base_task, ctx, dedupe_by_table=dedupe_by_table,
                )
            retr_t = rtiming.probe_time_s + rtiming.union_combine_time_s + rtiming.union_lookup_time_s
            D = drop_empty(D_union_full)
            F_Q = (N_Q / n_Q) if n_Q else None

            if len(D) < max_k:
                n_skipped += 1
                continue

            for k in ks:
                for alpha in alphas:
                    task = base_task._replace(k=k, alpha=alpha)
                    t_e2e = time.perf_counter()
                    pool_size = int(alpha * k)
                    clamped = pool_size >= len(D)

                    t0 = time.perf_counter()
                    if clamped:
                        P_un, iters, dink_ran = D, 0, False
                        F_P = F_ratio(*aggregate(D, N_Q, n_Q, True))
                    else:
                        s1 = stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, True)
                        P_un, iters, dink_ran = s1.selected, s1.iterations, True
                        F_P = s1.objective_value
                    stage1_t = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    P = _score(P_un, ctx, task.q_table, int(task.attr), dedupe_by_table)
                    scoring_t = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    r2 = stage2_ilp.solve(P, k, tau, N_Q, n_Q, True)
                    stage2_t = time.perf_counter() - t0
                    e2e = retr_t + (time.perf_counter() - t_e2e)
                    F_R = F_ratio(*aggregate(r2.selected, N_Q, n_Q, True)) if r2.feasible else None

                    rows.append(SweepRow(
                        tier=tier, top_n=top_n, q_table=task.q_table, attr=int(task.attr),
                        k=k, alpha=alpha, tau=tau, n_sem=telem_u.n_sem, n_ovl=telem_u.n_ovl,
                        n_D_union=len(D), pool_size_requested=pool_size, clamped=clamped,
                        feasible=r2.feasible,
                        tau_reached=bool(r2.feasible and F_R is not None and F_R >= tau - 1e-12),
                        n_P=len(P), n_R=len(r2.selected), n_unionability_computations=len(P_un),
                        F_Q=F_Q, F_P=F_P, F_R=F_R,
                        sum_U=r2.objective_value if r2.feasible else None,
                        dinkelbach_ran=dink_ran, dinkelbach_iterations=iters,
                        retrieval_time_s=retr_t, probe_time_s=rtiming.probe_time_s,
                        union_combine_time_s=rtiming.union_combine_time_s,
                        union_lookup_time_s=rtiming.union_lookup_time_s,
                        stage1_time_s=stage1_t, scoring_time_s=scoring_t, stage2_time_s=stage2_t,
                        end_to_end_s=e2e,
                    ))
            n_done += 1

        if verbose:
            print(
                "[{} top_n={}] done: {} queries used, {} skipped (|D_union|<{}) -- {:.1f}s".format(
                    tier, top_n, n_done, n_skipped, max_k, time.perf_counter() - t_tier_topn,
                ), flush=True,
            )

    return rows, build_times, selection_reports


def write_rows(rows: List[SweepRow], name: str, output_dir: str = "experiments/results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SweepRow._fields)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    return path


def write_selection_reports(reports: Dict[int, dict], name: str, output_dir: str = "experiments/results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name + "_selection_reports.json")
    with open(path, "w") as f:
        json.dump({str(top_n): r for top_n, r in reports.items()}, f, indent=2)
    return path
