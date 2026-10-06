"""Run one system over every query of a dataset and return one ``QueryOutcome`` per query.

* ``duts`` composes ``dutsx.runner.run_query`` (retrieval -> Stage 1 -> unionability on the pool
  -> Stage 2) over a context built from the configured paths, with the HNSW index loaded from
  the index directory (never rebuilt here -- see ``prereqs``).
* the Starmie baselines reuse ``experiments/baselines/run_starmie_baselines.py``'s ``_init`` /
  ``_one`` unchanged, with a ``Settings`` built from the configured paths.

Both use the same metric code (``experiments.groundtruth.precision_recall``) and the same value
set ``M`` (the protected value plus its numeric spelling variants, ``eval_fair3.expand_M``).
Per-query runtime covers that query's work only; one-off setup (loading embeddings and the
synopsis, building the in-memory indexes) is timed separately.
"""
import os
import shutil
import tempfile
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from .prereqs import (StaleIndexError, datalake_vec_pkl, hnsw_key, indexed_tables, load_synopsis,
                      query_dir, query_vec_pkl, semantic_vectors)


class QueryOutcome(NamedTuple):
    q_table: str
    feasible: bool
    n_returned: int
    precision: Optional[float]       # of what was returned; None if nothing was returned
    recall: Optional[float]
    ideal_recall: float               # min(k, |gt|) / |gt|
    gt_size: int
    F_R: Optional[float]
    sum_U: Optional[float]
    runtime_s: float
    cause: Optional[str]              # why it is infeasible, when known
    tables: str                       # ';'-joined result tables


class RunOutput(NamedTuple):
    outcomes: List[QueryOutcome]
    n_skipped: int                    # query-list rows with no query table / no embedding
    setup_s: float
    details: Dict[str, Any]


def run(v: Dict[str, Any], hnsw_m: int, log: Callable[[str], None] = print) -> RunOutput:
    if v["system"] == "duts":
        return run_duts(v, hnsw_m, log)
    return run_baseline(v, log)


def _ideal(k: int, gt_size: int) -> float:
    return (min(k, gt_size) / gt_size) if gt_size else 0.0


def run_duts(v: Dict[str, Any], hnsw_m: int, log=print) -> RunOutput:
    from dutsx import registry
    from dutsx.adapters.unionability import load_starmie_vectors
    from dutsx.runner import RunnerContext, load_queries_from_csv, run_query
    from experiments.fair3.eval_fair3 import expand_M
    from experiments.groundtruth import load_groundtruth, precision_recall
    from experiments.semantic_cache import load_cached

    t0 = time.perf_counter()
    log("loading synopsis, embeddings and index ...")
    synopsis = load_synopsis(v)
    dl_vecs = load_starmie_vectors(datalake_vec_pkl(v))
    q_vecs = load_starmie_vectors(query_vec_pkl(v))
    sem = semantic_vectors(v, synopsis, dl_vecs)
    semantic = load_cached(hnsw_key(v), sem, v["sigma"], v["theta_cat"], m=hnsw_m, cache_dir=v["index_path"])
    if semantic is None:
        raise StaleIndexError(
            "the DUTS HNSW index in {} was built from different embeddings or synopsis than the "
            "current ones".format(v["index_path"]))
    files = indexed_tables(v, synopsis, dl_vecs)
    overlap = registry.build("overlap", "inverted_index", synopsis=synopsis, tables=files)
    scorer = registry.build("unionability", "pinned_match", query_vectors=q_vecs,
                            candidate_vectors=dl_vecs, threshold=v["sigma"])
    ctx = RunnerContext(synopsis=synopsis, semantic=semantic, overlap=overlap,
                        unionability=scorer, query_vectors=q_vecs)
    truth = load_groundtruth(v["groundtruth_csv"])
    tasks, skipped = load_queries_from_csv(
        v["protected_csv"], query_dir(v), k=v["k"], alpha=v["alpha"], F_star=v["f_star"],
        delta=v["delta"], include_query=True, top_n=v["top_n"])
    no_vec = [t for t in tasks if t.q_table not in q_vecs or not synopsis.has_table(t.q_table)]
    tasks = [t for t in tasks if t not in no_vec]
    if v.get("limit_queries"):
        tasks = tasks[:v["limit_queries"]]
    setup_s = time.perf_counter() - t0
    log("setup {:.1f}s: {} datalake tables indexed, {} queries ({} skipped)".format(
        setup_s, len(files), len(tasks), len(skipped) + len(no_vec)))

    outcomes: List[QueryOutcome] = []
    for i, base in enumerate(tasks, 1):
        (value,) = base.M
        task = base._replace(M=expand_M(value)[0])
        gt = truth.get(task.q_table, frozenset())
        t = time.perf_counter()
        try:
            res = run_query(task, ctx)
            err = None
        except (KeyError, IndexError) as e:     # a malformed query row; reported, not fatal
            res, err = None, "{}: {}".format(type(e).__name__, e)
        runtime = time.perf_counter() - t
        if res is None:
            outcomes.append(QueryOutcome(task.q_table, False, 0, None, None, _ideal(v["k"], len(gt)),
                                         len(gt), None, None, runtime, err, ""))
        else:
            tel = res.telemetry
            names = [c.table for c in res.selected]
            prec, rec, _ = precision_recall(names, gt)
            outcomes.append(QueryOutcome(
                task.q_table, tel.feasible, len(names), prec, rec, _ideal(v["k"], len(gt)), len(gt),
                tel.F_R, tel.sum_U, runtime, tel.infeasibility_cause, ";".join(names)))
        if i % 10 == 0 or i == len(tasks):
            log("  {}/{} queries".format(i, len(tasks)))
    return RunOutput(outcomes, len(skipped) + len(no_vec), setup_s,
                     {"n_indexed_tables": len(files), "hnsw_m": hnsw_m})


def run_baseline(v: Dict[str, Any], log=print) -> RunOutput:
    import csv
    import multiprocessing as mp
    import pickle

    from experiments.baselines import run_starmie_baselines as rsb
    from .config import BUNDLED_STARMIE, METADATA_FILE, SYSTEMS

    approach = SYSTEMS[v["system"]].starmie_approach
    scratch = None
    metadata_dir = os.path.dirname(v["metadata_path"])
    if os.path.basename(v["metadata_path"]) != METADATA_FILE:
        # HNSWSearcher_Fair only looks for <metadata_dir>/metadata_combined.pkl (a symlink, no copy)
        scratch = tempfile.mkdtemp(prefix="duts_baseline_")
        metadata_dir = os.path.join(scratch, "metadata")
        os.makedirs(metadata_dir)
        os.symlink(v["metadata_path"], os.path.join(metadata_dir, METADATA_FILE))
    # scratch_dir=None: Starmie's HNSW index is kept in memory only, never written to disk.
    st = rsb.Settings(
        bench=v["dataset"], approach=approach, starmie_root=BUNDLED_STARMIE,
        datalake_vec_pkl=datalake_vec_pkl(v), query_vec_pkl=query_vec_pkl(v),
        query_dir=query_dir(v), datalake_dir=os.path.join(v["dataset_path"], "datalake"),
        metadata_dir=metadata_dir, groundtruth_csv=v["groundtruth_csv"], scratch_dir=None,
        ks=(v["k"],), n=v["n_columns"], sigma=v["sigma"], f_star=v["f_star"], delta=v["delta"])

    qs, n_rows = [], 0
    with open(v["protected_csv"], newline="") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            qs.append((r["q_name"].strip(), int(r["protected_attribute_id"]), r["protected_value"].strip()))
    with open(query_vec_pkl(v), "rb") as f:
        have = {q[0] for q in pickle.load(f)}
    qs = [q for q in qs if q[0] in have]
    n_skipped = n_rows - len(qs)
    if v.get("limit_queries"):
        qs = qs[:v["limit_queries"]]

    cwd = os.getcwd()
    t0 = time.perf_counter()
    rows: List[Tuple[dict, float]] = []
    try:
        # As in the paper's runs: each worker builds its own Starmie searcher (HNSW index). The index
        # is kept in memory only (scratch_dir=None), so nothing is written to disk per worker.
        t_q = t0
        if v["workers"] == 1:
            log("building Starmie's HNSW index in memory ...")
            _init_quiet(st)
            setup_s = time.perf_counter() - t0
            log("setup {:.1f}s; {} queries ({} skipped)".format(setup_s, len(qs), n_skipped))
            t_q = time.perf_counter()
            for i, q in enumerate(qs, 1):
                rows.append(_timed_one(q))
                if i % 10 == 0 or i == len(qs):
                    log("  {}/{} queries".format(i, len(qs)))
        else:
            log("starting {} workers (each builds Starmie's HNSW index in memory) ...".format(v["workers"]))
            setup_s = 0.0   # per-worker setup overlaps with queries; it is inside total runtime
            with mp.get_context("fork").Pool(v["workers"], initializer=_init_quiet, initargs=(st,)) as pool:
                for i, r in enumerate(pool.imap(_timed_one, qs), 1):
                    rows.append(r)
                    if i % 10 == 0 or i == len(qs):
                        log("  {}/{} queries".format(i, len(qs)))
        query_wall_s = time.perf_counter() - t_q
    finally:
        os.chdir(cwd)     # _init chdirs into the Starmie module directory
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)

    outcomes = []
    for (row, runtime), q in zip(rows, qs):
        names = [t for t in row["tables"].split(";") if t]
        outcomes.append(QueryOutcome(
            row["q_table"], bool(row["feasible"]), len(names), row["prec"], row["rec"], row["ideal_recall"],
            row["gt_size"], row["F"] if row["feasible"] else None, row["duts_sum_U"] if names else None,
            runtime, None if row["feasible"] else _baseline_cause(row, v["k"]), row["tables"]))
    return RunOutput(outcomes, n_skipped, setup_s, {"approach": approach, "workers": v["workers"],
                                                    "query_wall_s": query_wall_s})


def _init_quiet(st) -> None:
    from experiments.baselines import run_starmie_baselines as rsb
    rsb._init(st)


def _timed_one(q) -> Tuple[dict, float]:
    from experiments.baselines import run_starmie_baselines as rsb
    t = time.perf_counter()
    (row,) = rsb._one(q)          # one row per k; exactly one k is configured
    return row, time.perf_counter() - t


def _baseline_cause(row: dict, k: int) -> str:
    if row["n_returned"] < k:
        return "fewer_than_k_returned"
    return "below_tau"
