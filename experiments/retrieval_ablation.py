"""Experiment 3 -- retrieval ablation (PLAN-integration.md §4 Phase E).

``hnsw`` vs. ``exact_scan`` (semantic retriever), and ``inverted_index`` vs.
``null`` (overlap filter) -- four adapter combinations at one fixed
``(k, alpha, F_star, delta)`` point, over every usable santos query.
Quantifies what each filter contributes to ``|D|`` (via the retrieval
telemetry columns ``n_sem``/``n_ovl``/``n_pair``/``n_D``) and, downstream, to
feasibility and quality.

Each combination needs its own ``RunnerContext`` (semantic/overlap are baked
into the index at construction time), built via ``experiments.context.build_santos_context``.
"""
import time
from typing import List, Optional

from dutsx import registry
from dutsx.runner import RunnerContext, run_query

from . import context as ctx_mod
from .schema import ResultRow, row_from_result, timed_run_query

SEMANTIC_VARIANTS = ("hnsw", "exact_scan")
OVERLAP_VARIANTS = ("inverted_index", "null")


def run_retrieval_ablation(
    k: int, alpha: float, F_star: float, delta: float,
    include_query: bool = True, top_n: int = 100, seed: int = 42,
    unionability: str = "pinned_match",
    experiment: str = "retrieval_ablation",
    verbose: bool = True,
    task_limit: Optional[int] = None,
    paths=None,
) -> List[ResultRow]:
    """Build and run the 2x2 semantic x overlap grid. Loads the base task
    list once (retrieval adapters don't affect which CSV rows resolve to a
    file), builds one ``RunnerContext`` per combination, and reuses the task
    list (``_replace``d for the fixed ``k``/``alpha``/etc.) across all four.

    ``task_limit`` truncates the query list (first N by CSV row order) --
    a smoke-test aid for tests/CLI ``--limit-queries``; leave ``None`` for a
    real sweep over every usable query.
    """
    base_tasks, skipped = ctx_mod.load_base_tasks(
        k=k, alpha=alpha, F_star=F_star, delta=delta,
        include_query=include_query, top_n=top_n, paths=paths,
    )
    if task_limit is not None:
        base_tasks = base_tasks[:task_limit]

    # Loaded once, shared across all 4 combinations below -- semantic/overlap
    # choice doesn't change the CSVs or vector pickles, and InvertedIndexOverlap's
    # construction alone reads every one of 550 tables' categoricals through
    # this synopsis, so a fresh CsvSynopsis per combination would re-read the
    # CSVs from disk 4x for no reason (see load_shared_resources docstring).
    synopsis, datalake_files, datalake_vectors, query_vectors = ctx_mod.load_shared_resources(
        paths=paths)

    # Each of the 4 (semantic, overlap) combinations shares its semantic
    # adapter with one other combination and its overlap adapter with
    # another (2x2 grid) -- build each adapter exactly ONCE (2 semantic + 2
    # overlap builds, not 4 of each) rather than rebuilding e.g. a fresh
    # hnswlib index twice for the two overlap variants that pair with it.
    # Unionability doesn't vary in this experiment, so it's built once too.
    semantic_vectors = ctx_mod.categorical_only_vectors(datalake_vectors, synopsis, datalake_files)
    semantic_objs = {
        name: registry.build("semantic", name, vectors=semantic_vectors, sigma=ctx_mod.SIGMA)
        for name in SEMANTIC_VARIANTS
    }
    overlap_objs = {
        name: registry.build("overlap", name, synopsis=synopsis, tables=datalake_files)
        for name in OVERLAP_VARIANTS
    }
    unionability_obj = registry.build(
        "unionability", unionability, query_vectors=query_vectors,
        candidate_vectors=datalake_vectors, threshold=ctx_mod.SIGMA,
    )

    rows: List[ResultRow] = []
    for semantic in SEMANTIC_VARIANTS:
        for overlap in OVERLAP_VARIANTS:
            config_id = "semantic={}_overlap={}".format(semantic, overlap)
            adapters = {"synopsis": "csv", "semantic": semantic, "overlap": overlap,
                        "unionability": unionability}
            ctx = RunnerContext(
                synopsis=synopsis, semantic=semantic_objs[semantic], overlap=overlap_objs[overlap],
                unionability=unionability_obj, query_vectors=query_vectors,
            )
            t_cond0 = time.perf_counter()
            for task in base_tasks:
                result, wall, error = timed_run_query(run_query, task, ctx)
                rows.append(row_from_result(
                    experiment, config_id, seed, task, result, wall, adapters,
                    lp_precheck=True, error=error,
                ))
            if verbose:
                print("  {} ({} queries, {:.1f}s)".format(
                    config_id, len(base_tasks), time.perf_counter() - t_cond0))
    return rows
