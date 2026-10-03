"""WDC union-pool scalability experiment.

``scalability_study.py::run_tier_study`` uses the paper-faithful intersection retrieval
(``dutsx.runner.retrieve_unscored_candidates``), whose pools are usually too small to exercise
Stage 1/scoring/Stage 2 meaningfully -- ``RESULTS-wdc.md`` §4/§5 measured only 5/30 and 6/30
"workable" queries (``|D| >= 20``) at ``tier_10k``/``tier_100k``. This module answers a narrower
question instead: **given the SAME auto-selected queries, what does Stage 1 / unionability scoring /
Stage 2 actually cost once the pool is large enough to run them for real** -- by retrieving with
``experiments.wdc.union_retrieval.retrieve_unscored_candidates_union_and_intersection``
(``D = D_sem ∪ D_ovl``, see that module's docstring for why this is an explicit,
WDC-scalability-only deviation from §7.3).

For each query this also records the intersection pool size (``n_D_intersection``, a comparison
stat only) alongside the union pool size (``n_D_union``) actually used for the timing loop, plus the
raw semantic/overlap pair counts (``n_sem``, ``n_ovl``) -- the source data for "how many attributes
were indexed vs. retrieved" reporting.

**Correction found by inspection (not by the original author of this module):** an earlier version
called ``dutsx.runner.retrieve_unscored_candidates`` (intersection) AND
``union_retrieval.retrieve_unscored_candidates_union`` back-to-back inside the SAME timed block, and
reported their combined wall time as ``retrieval_time_s``. That's wrong -- it silently doubled the
actual probe cost (the HNSW query and the overlap posting-list scan both ran twice), and everywhere
this module reported "retrieval time," it actually meant "cost of computing both variants for
comparison," not "cost of the union path alone." Fixed by probing once
(``union_retrieval.retrieve_unscored_candidates_union_and_intersection``) and deriving both reduced
candidate sets from that single probe, with an explicit per-phase timing breakdown
(``union_retrieval.CombinedRetrievalTiming``): the shared probe, the union combine step, the
intersection combine step, and each variant's own N_i/n_i lookup, each timed separately rather than
lumped into one number. ``retrieval_time_s`` below is ``probe + union_combine + union_lookup`` (the
actual union-path cost); ``intersection_time_s`` is the analogous ``probe + intersection_combine +
intersection_lookup`` for the comparison-only intersection set, reusing the same probe.

**Stress-test mode (``dedupe_by_table=False``, requested explicitly for this experiment only):**
by default (``dedupe_by_table=True``), §7.3's per-table reduction still applies here -- each
physical table contributes at most one row to ``D`` (the ``argmax sim`` column), so ``n_D_union``
counts distinct TABLES. Passing ``dedupe_by_table=False`` skips that reduction: every surviving
``(table, attr)`` pair becomes its own row, so a table matched on 5 columns contributes 5 rows.
This inflates ``D`` (and therefore Stage 1's input size and the N_i/n_i lookup cost) without
changing Stage 2's output size (``P``/``R`` stay capped at ``alpha*k``/``k``), which is exactly the
point -- it stresses Stage 1 and the lookup step against much larger, more realistic-scale pools.
It is NOT a more correct retrieval mode (a table can end up "selected" multiple times under
different synthetic ids -- see ``_score``'s note) and is not used anywhere outside this explicit
stress-test call.

**A second, independent pool-growth lever: ``sigma`` (the HNSW similarity gate).** ``dedupe_by_table``
grows ``D`` by keeping every retrieved column instead of collapsing to one-per-table; ``sigma`` grows
``D`` a different way -- ``dutsx/adapters/semantic.py::HnswRetriever.query()`` fetches ``top_n``
nearest neighbors then filters to ``sim > sigma``, so lowering ``sigma`` below the paper-matching
default (``context.SIGMA = 0.6``) lets more of the already-fetched ``top_n`` neighbors survive the
filter -- which can grow the number of *distinct tables* reachable via the semantic side even with
``dedupe_by_table=True`` (the paper-faithful one-to-one reduction) left on. The two levers are
independent and can be combined, but the default here still applies the per-table reduction.
"""
import csv
import os
import time
from typing import List, NamedTuple, Optional

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import F as F_ratio
from duts.stats import aggregate, drop_empty
from duts.types import CandidateStats

from .context import SIGMA, build_wdc_context, paths_for_tier
from .query_selection import select_queries_with_comparison
from .union_retrieval import retrieve_unscored_candidates_union_and_intersection


class WdcUnionRow(NamedTuple):
    tier: str
    q_table: str
    attr: int
    k: int
    alpha: float
    tau: float
    n_sem: int
    n_ovl: int
    n_D_intersection: int
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
    retrieval_time_s: float          # = probe_time_s + union_combine_time_s + union_lookup_time_s
    intersection_time_s: float        # = probe_time_s + intersection_combine_time_s + intersection_lookup_time_s (comparison only, not on the critical path)
    probe_time_s: float                # semantic.query + overlap.query -- shared, paid once regardless of combine rule
    union_combine_time_s: float        # set union + per-table argmax reduction
    intersection_combine_time_s: float  # set intersection + per-table argmax reduction
    union_lookup_time_s: float          # N_i/n_i synopsis lookup over D_union
    intersection_lookup_time_s: float   # N_i/n_i synopsis lookup over D_intersection
    stage1_time_s: float
    scoring_time_s: float
    stage2_time_s: float
    end_to_end_s: float


def _score(cands, ctx, q_table, q_attr_idx, dedupe_by_table: bool = True) -> List[CandidateStats]:
    """Duplicated from ``dutsx.runner._score_pool`` on purpose -- same
    precedent as ``scalability_study.py::_score``.

    ``CandidateStats.table`` doubles as Stage 1's candidate identity (its
    convergence test and tie-break both key off it -- see
    ``duts/stage1_dinkelbach.py``). Under the ``dedupe_by_table=False``
    stress test, ``cands`` can contain several rows sharing the same
    physical table (different columns), so each gets a synthetic
    ``"<table>::<attr>"`` id here to keep Stage 1's per-candidate identity
    well-defined; ``ctx.unionability.score`` is still called with the REAL
    ``c.table`` (unaffected -- unionability is a real per-table lookup)."""
    out = []
    for c in cands:
        u = ctx.unionability.score(q_table, c.table, (q_attr_idx, c.attr))
        cand_id = c.table if dedupe_by_table else "{}::{}".format(c.table, c.attr)
        out.append(CandidateStats(table=cand_id, N=c.N, n=c.n, U=u))
    return out


def run_tier_study_union(
    tier: str,
    n_queries: int = 30,
    ks=(10, 20),
    alphas=(2.0, 3.0),
    F_star: float = 0.3,
    delta: float = 0.15,
    top_n: int = 200,
    seed: int = 42,
    verbose: bool = True,
    dedupe_by_table: bool = True,
    sigma: float = SIGMA,
):
    ctx, build_times = build_wdc_context(tier, sigma=sigma)
    if verbose:
        print(
            "[{}] index build: {:.2f}s over {} tables ({}/{} categorical, ratio {:.1%})".format(
                tier, build_times.total_index_s, build_times.n_tables,
                build_times.n_columns_categorical, build_times.n_columns_total,
                build_times.categorical_coverage_ratio,
            ), flush=True,
        )

    tau = F_star - delta
    paths = paths_for_tier(tier)
    tables = sorted(f for f in os.listdir(paths.csv_dir) if f.endswith(".csv"))

    # Same seed/params as scalability_study.py::run_tier_study -- identical
    # (q_table, attr, M) selections, so intersection- and union-mode results
    # are directly comparable per query, not just per tier.
    tasks, selection_report = select_queries_with_comparison(
        ctx, tables, n_queries=n_queries, seed=seed, k=ks[0], alpha=alphas[0],
        F_star=F_star, delta=delta, top_n=top_n,
    )
    if verbose:
        print("[{}] {} queries selected for union-pool sweep".format(tier, len(tasks)), flush=True)

    rows: List[WdcUnionRow] = []
    for base_task in tasks:
        # One probe (semantic + overlap) derives BOTH reduced sets, each
        # phase timed separately -- see
        # retrieve_unscored_candidates_union_and_intersection's docstring.
        D_union_full, D_inter, N_Q, n_Q, telem_u, _telem_i, rtiming = \
            retrieve_unscored_candidates_union_and_intersection(
                base_task, ctx, dedupe_by_table=dedupe_by_table,
            )
        retr_t = rtiming.probe_time_s + rtiming.union_combine_time_s + rtiming.union_lookup_time_s
        inter_t = rtiming.probe_time_s + rtiming.intersection_combine_time_s + rtiming.intersection_lookup_time_s

        D = drop_empty(D_union_full)
        F_Q = (N_Q / n_Q) if n_Q else None
        n_D_intersection = len(drop_empty(D_inter))

        if len(D) < max(ks):
            if verbose:
                print("  skip {} (union |D|={} < {})".format(
                    base_task.q_table, len(D), max(ks)), flush=True)
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

                rows.append(WdcUnionRow(
                    tier=tier, q_table=task.q_table, attr=int(task.attr), k=k, alpha=alpha,
                    tau=tau, n_sem=telem_u.n_sem, n_ovl=telem_u.n_ovl,
                    n_D_intersection=n_D_intersection, n_D_union=len(D),
                    pool_size_requested=pool_size, clamped=clamped,
                    feasible=r2.feasible,
                    tau_reached=bool(r2.feasible and F_R is not None and F_R >= tau - 1e-12),
                    n_P=len(P), n_R=len(r2.selected), n_unionability_computations=len(P_un),
                    F_Q=F_Q, F_P=F_P, F_R=F_R,
                    sum_U=r2.objective_value if r2.feasible else None,
                    dinkelbach_ran=dink_ran, dinkelbach_iterations=iters,
                    retrieval_time_s=retr_t, intersection_time_s=inter_t,
                    probe_time_s=rtiming.probe_time_s,
                    union_combine_time_s=rtiming.union_combine_time_s,
                    intersection_combine_time_s=rtiming.intersection_combine_time_s,
                    union_lookup_time_s=rtiming.union_lookup_time_s,
                    intersection_lookup_time_s=rtiming.intersection_lookup_time_s,
                    stage1_time_s=stage1_t,
                    scoring_time_s=scoring_t, stage2_time_s=stage2_t, end_to_end_s=e2e,
                ))
        if verbose:
            print("  done {} (union |D|={}, intersection |D|={})".format(
                task.q_table, len(D), n_D_intersection), flush=True)

    return rows, build_times, selection_report


def write_rows(rows: List[WdcUnionRow], name: str, output_dir: str = "experiments/results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(WdcUnionRow._fields)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    return path
