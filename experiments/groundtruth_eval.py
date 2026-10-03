"""Precision/recall against santos-style groundtruth, at each pipeline stage
(D, P, R), plus runtime and feasibility rate.

Answers, per query: how many candidates survive retrieval (``D``), how many
survive Stage 1's distribution-optimal pool (``P``), how many are in the
final Stage 2 result (``R``); of each, how many are also considered
genuinely unionable by the benchmark's own groundtruth (precision/recall);
how long the query took; and whether it was feasible at all.

See ``experiments/groundtruth.py`` for what precision/recall means here and
what it deliberately does not test (column identity is ignored).
"""
from typing import List, Optional

from dutsx.runner import run_query

from . import context as ctx_mod
from .groundtruth import load_groundtruth, precision_recall
from .schema import ResultRow, row_from_result, timed_run_query


def run_groundtruth_eval(
    paths: ctx_mod.BenchmarkPaths,
    k: int,
    alpha: float,
    F_star: float,
    delta: float,
    include_query: bool = True,
    top_n: int = 100,
    unionability: str = "pinned_match",
    seed: int = 42,
    task_limit: Optional[int] = None,
) -> List[ResultRow]:
    """Run every usable query for ``paths.name`` end to end, scoring
    precision/recall of ``D``/``P``/``R`` against ``paths.groundtruth_csv``."""
    truth = load_groundtruth(paths.groundtruth_csv)

    synopsis, dl_files, dl_vecs, q_vecs = ctx_mod.load_shared_resources(paths=paths)
    ctx, adapters = ctx_mod.build_santos_context(
        unionability=unionability, synopsis=synopsis, datalake_files=dl_files,
        datalake_vectors=dl_vecs, query_vectors=q_vecs, paths=paths,
    )
    tasks, _skipped = ctx_mod.load_base_tasks(
        k=k, alpha=alpha, F_star=F_star, delta=delta, include_query=include_query,
        top_n=top_n, paths=paths,
    )
    if task_limit is not None:
        tasks = tasks[:task_limit]

    rows: List[ResultRow] = []
    for task in tasks:
        result, wall_time_s, error = timed_run_query(run_query, task, ctx)
        gt = truth.get(task.q_table)

        d_tables = [c.table for c in result.candidates] if result else []
        p_tables = [c.table for c in result.pool] if result else []
        r_tables = [c.table for c in result.selected] if result else []

        if gt is not None:
            precision_d, recall_d, d_hits = precision_recall(d_tables, gt)
            precision_p, recall_p, p_hits = precision_recall(p_tables, gt)
            precision_r, recall_r, r_hits = precision_recall(r_tables, gt)
            gt_size = len(gt)
        else:
            precision_d = recall_d = precision_p = recall_p = precision_r = recall_r = None
            d_hits = p_hits = r_hits = None
            gt_size = None

        rows.append(row_from_result(
            experiment="groundtruth_eval",
            config_id="k{}_a{}".format(k, alpha),
            seed=seed, task=task, result=result, wall_time_s=wall_time_s,
            adapters=adapters, error=error,
            gt_size=gt_size, d_hits=d_hits, p_hits=p_hits, r_hits=r_hits,
            precision_d=precision_d, recall_d=recall_d,
            precision_p=precision_p, recall_p=recall_p,
            precision_r=precision_r, recall_r=recall_r,
        ))
    return rows
