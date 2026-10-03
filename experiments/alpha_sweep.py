"""Experiment 1 -- the alpha sweep (PLAN-integration.md §4 Phase E, headline
experiment). ``alpha in {1,2,3,5,10} x k in {5,10,20}`` over every usable
santos query. The C4 feasibility certificate is NOT run (removed
2026-08-10, user instruction) -- infeasibility is therefore reported only as
``insufficient_candidates`` (|D| < k) or ``stage2_infeasible``, with no
intrinsic-vs-alpha attribution.

Quality (``sum_U``, ``F_R``) vs. cost (``n_unionability_computations``, wall
time) vs. infeasibility rate split by cause: all three come straight out of
the tidy ``ResultRow``s this driver produces, no separate aggregation logic
needed at collection time -- see ``experiments/analysis.py`` for the derived
summaries.

Also harvests ``Stage2Instance``s: every ``(task, N_Q, n_Q, P)`` where Stage 1
actually produced a pool (``n_P > 0``) -- i.e. every condition that reached
Stage 2, feasible or not. ``experiments/lp_precheck.py`` replays these
directly through ``duts.stage2_ilp.solve`` with both ``lp_precheck`` settings,
without repeating retrieval/Stage 1/scoring (the expensive part) for a second
experiment that only cares about Stage 2's own cost.
"""
import time
from typing import List, NamedTuple, Tuple

from duts.stats import N_of
from dutsx.runner import QueryTask, RunnerContext, run_query

from .schema import ResultRow, row_from_result, timed_run_query


class Stage2Instance(NamedTuple):
    config_id: str
    task: QueryTask
    N_Q: int
    n_Q: int
    P: list   # List[CandidateStats], already scored -- reuse as-is, never re-score


def run_alpha_sweep(
    base_tasks: List[QueryTask],
    ctx: RunnerContext,
    adapters: dict,
    alphas: Tuple[float, ...] = (1, 2, 3, 5, 10),
    ks: Tuple[int, ...] = (5, 10, 20),
    seed: int = 42,
    experiment: str = "alpha_sweep",
    verbose: bool = True,
) -> Tuple[List[ResultRow], List[Stage2Instance]]:
    """Run every ``(k, alpha)`` condition over every task in ``base_tasks``.

    ``base_tasks`` must already carry the shared
    ``F_star``/``delta``/``top_n``/``include_query`` -- only ``k`` and
    ``alpha`` vary per condition, via ``task._replace``.
    """
    rows: List[ResultRow] = []
    instances: List[Stage2Instance] = []
    n_conditions = len(ks) * len(alphas)
    cond_i = 0
    for k in ks:
        for alpha in alphas:
            cond_i += 1
            config_id = "k={}_alpha={}".format(k, alpha)
            t_cond0 = time.perf_counter()
            for base in base_tasks:
                task = base._replace(k=k, alpha=alpha)
                result, wall, error = timed_run_query(run_query, task, ctx)

                rows.append(row_from_result(
                    experiment, config_id, seed, task, result, wall, adapters,
                    lp_precheck=True, error=error,
                ))

                if result is not None and result.telemetry.n_P > 0:
                    n_Q = ctx.synopsis.n_rows(task.q_table)
                    N_Q = N_of(ctx.synopsis.distribution(task.q_table, task.attr), task.M)
                    instances.append(Stage2Instance(
                        config_id=config_id, task=task, N_Q=N_Q, n_Q=n_Q, P=result.pool,
                    ))
            if verbose:
                print("  [{}/{}] {} ({} queries, {:.1f}s)".format(
                    cond_i, n_conditions, config_id, len(base_tasks),
                    time.perf_counter() - t_cond0,
                ))
    return rows, instances
