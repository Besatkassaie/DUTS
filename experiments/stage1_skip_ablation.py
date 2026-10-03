"""Experiment: Stage-1-skip ablation (user request, 2026-08-09) -- "how long
does it take to skip the fractional optimization step and go directly to
Stage 2, vs. the combined two-stage pipeline?"

Two conditions, same query, same retrieved ``D``, same unionability scorer:

    two_stage    retrieve -> Stage 1 pool (size alpha*k) -> score U on P only -> Stage 2 ILP over P
    skip_stage1  retrieve -> score U on ALL of D -> Stage 2 ILP over D directly, cardinality k

Both conditions solve the identical Stage 2 problem (``|R|=k, F_R >= tau``,
``duts.stage2_ilp.solve`` unmodified) -- they differ only in which candidate
set it runs over. So ``skip_stage1`` is not an approximation of ``two_stage``;
it is what "no distribution-first pre-filtering" costs, holding everything
else fixed. This isolates two effects that ``run_query``'s combined
``wall_time_s`` conflates:

- **scoring cost**: two_stage scores exactly ``len(P) == alpha*k`` candidates
  (the paper's central efficiency claim, dutsx/runner.py); skip_stage1 scores
  every candidate in ``D``, which is >= that and, on a larger/noisier
  datalake than santos's ~10-candidate pools, could be far larger.
- **ILP cost**: Stage 2's branch-and-bound runs over a bigger candidate set
  in skip_stage1 (``len(D)`` vs. ``len(P)``) -- on santos, both are small
  enough that this is unlikely to dominate (see NOTES.md's LP pre-check
  finding: HiGHS is fast at this scale regardless), but the sweep reports it
  either way rather than assuming it.

**Quality is also reported, not just runtime.** Stage 2's objective is
``max ΣU`` subject to ``F_R >= tau`` -- run over a *superset* of candidates
(D ⊇ P), so ``skip_stage1``'s optimum can only be >= ``two_stage``'s (more
choices, same constraint). Any runtime savings from ``two_stage`` therefore
trade off against this: Stage 1's pool is a distribution-aware pre-filter
that may exclude a higher-``U`` table Stage 2 would otherwise have picked.
Reporting ``sum_U``/``F_R`` alongside runtime is what makes that trade-off
legible instead of assumed.

Deliberately does NOT reuse ``dutsx.runner._score_pool`` -- that function's
docstring in ``dutsx/runner.py`` makes a grep-able guarantee ("exactly one
call site, and it is not D") that this ablation would falsify by construction
if it called into it with ``D``. ``_score_all`` below duplicates its five
lines rather than weaken that guarantee.
"""
import time
from typing import List, Optional, Tuple

from duts import stage2_ilp
from duts.stats import F as F_ratio
from duts.stats import aggregate, delta as delta_of, drop_empty
from duts.types import CandidateStats
from dutsx.runner import (
    QueryTask,
    RunnerContext,
    RunResult,
    retrieve_unscored_candidates,
)
from dutsx.runner import _default_telemetry  # noqa: SLF001 -- see module docstring

from . import context as ctx_mod
from .schema import ResultRow, row_from_result, timed_run_query
from dutsx.runner import run_query


def _score_all(D, ctx: RunnerContext, q_table: str, q_attr_idx: int) -> List[CandidateStats]:
    """Score every candidate in ``D`` -- the thing ``two_stage`` deliberately
    avoids doing (see module docstring). Duplicates ``_score_pool``'s body on
    purpose rather than importing it."""
    scored: List[CandidateStats] = []
    for c in D:
        u = ctx.unionability.score(q_table, c.table, (q_attr_idx, c.attr))
        scored.append(CandidateStats(table=c.table, N=c.N, n=c.n, U=u))
    return scored


def run_query_skip_stage1(task: QueryTask, ctx: RunnerContext) -> Tuple[RunResult, float]:
    """Same retrieval as ``dutsx.runner.run_query``, but Stage 1 is skipped
    entirely: every candidate in ``D`` is scored, then handed straight to
    Stage 2's ILP at cardinality ``task.k``. Returns ``(RunResult,
    scoring_time_s)`` -- the scoring time is the ablation's main measurement,
    reported separately from ``RunResult.telemetry.ilp_time_s`` rather than
    folded into it, so runtime attribution is explicit.
    """
    D, N_Q, n_Q, retrieval_telemetry = retrieve_unscored_candidates(task, ctx)
    D = drop_empty(D)
    tau = task.F_star - task.delta
    k, include_query = task.k, task.include_query
    q_attr_idx = task.attr if isinstance(task.attr, int) else int(task.attr)

    if len(D) < k:
        telemetry = _default_telemetry(
            retrieval_telemetry, n_D=len(D), effective_alpha=task.alpha,
            infeasibility_cause="insufficient_candidates",
        )
        return RunResult(selected=[], pool=[], candidates=D, telemetry=telemetry), 0.0

    t0 = time.perf_counter()
    P = _score_all(D, ctx, task.q_table, q_attr_idx)  # ALL of D, not alpha*k -- the point being measured
    scoring_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    stage2_result = stage2_ilp.solve(P, k, tau, N_Q, n_Q, include_query)
    ilp_time_s = time.perf_counter() - t1
    info = stage2_result.info or {}
    F_P = F_ratio(*aggregate(P, N_Q, n_Q, include_query))
    effective_alpha = len(D) / k

    if not stage2_result.feasible:
        telemetry = _default_telemetry(
            retrieval_telemetry, n_D=len(D), n_P=len(P),
            n_unionability_computations=len(P), F_P=F_P,
            scoring_time_s=scoring_time_s, ilp_time_s=ilp_time_s,
            lp_precheck_ran=info.get("lp_precheck_ran", False),
            lp_feasible=info.get("lp_feasible"), lp_time_s=info.get("lp_time_s"),
            effective_alpha=effective_alpha, infeasibility_cause="stage2_infeasible",
        )
        return RunResult(selected=[], pool=P, candidates=D, telemetry=telemetry), scoring_time_s

    R = stage2_result.selected
    F_R = F_ratio(*aggregate(R, N_Q, n_Q, include_query))
    telemetry = _default_telemetry(
        retrieval_telemetry, n_D=len(D), n_P=len(P), n_R=len(R),
        n_unionability_computations=len(P), F_P=F_P, F_R=F_R,
        delta_R=delta_of(task.F_star, F_R), sum_U=stage2_result.objective_value,
        scoring_time_s=scoring_time_s, ilp_time_s=ilp_time_s,
        lp_precheck_ran=info.get("lp_precheck_ran", False),
        lp_feasible=info.get("lp_feasible"), lp_time_s=info.get("lp_time_s"),
        effective_alpha=effective_alpha, feasible=True, infeasibility_cause=None,
    )
    return RunResult(selected=R, pool=P, candidates=D, telemetry=telemetry), scoring_time_s


def run_stage1_skip_ablation(
    k: int, alpha: float, F_star: float, delta: float,
    include_query: bool = True, top_n: int = 100, seed: int = 42,
    unionability: str = "pinned_match",
    benchmark: str = "santos",
    experiment: str = "stage1_skip_ablation",
    verbose: bool = True,
    task_limit: Optional[int] = None,
) -> List[ResultRow]:
    """Run every usable query for ``benchmark`` under both conditions
    (``two_stage`` via ``dutsx.runner.run_query`` unmodified, ``skip_stage1``
    via ``run_query_skip_stage1`` above), sharing one ``RunnerContext`` --
    the two conditions differ only in which function composes the stages,
    never in retrieval or scoring logic itself.
    """
    paths = ctx_mod.paths_for(benchmark)
    ctx, adapters = ctx_mod.build_santos_context(unionability=unionability, paths=paths)
    tasks, skipped = ctx_mod.load_base_tasks(
        k=k, alpha=alpha, F_star=F_star, delta=delta,
        include_query=include_query, top_n=top_n, paths=paths,
    )
    if task_limit is not None:
        tasks = tasks[:task_limit]
    if verbose:
        print("{} usable {} queries ({} skipped)".format(len(tasks), benchmark, len(skipped)))

    config_id = "k{}_a{}".format(k, alpha)
    rows: List[ResultRow] = []
    for task in tasks:
        result_two, wall_two, err_two = timed_run_query(run_query, task, ctx)
        rows.append(row_from_result(
            experiment, "two_stage__" + config_id, seed, task, result_two, wall_two,
            adapters, error=err_two,
        ))

        t0 = time.perf_counter()
        try:
            result_skip, _scoring_time_s = run_query_skip_stage1(task, ctx)
            err_skip = None
        except Exception as exc:  # noqa: BLE001 -- one query's failure must not abort the sweep
            result_skip, err_skip = None, "{}: {}".format(type(exc).__name__, exc)
        wall_skip = time.perf_counter() - t0
        rows.append(row_from_result(
            experiment, "skip_stage1__" + config_id, seed, task, result_skip, wall_skip,
            adapters, error=err_skip,
        ))
    return rows
