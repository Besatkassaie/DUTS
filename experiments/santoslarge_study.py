"""santosLarge at scale: the first configuration where ``alpha`` actually bites.

**Why this benchmark and this cohort.** On santos/santos3 the retrieved ``|D|``
is 6-10 tables, so ``alpha*k >= |D|`` fires PLAN.md's C5 clamp and ``P = D`` --
Stage 1 never selects anything and ``alpha`` is inert. santosLarge (11086
datalake tables) does not fix that on its own: measured, its *median* ``|D|`` is
9 even at ``top_n = 11000`` (an exhaustive scan), because most queries' protected
value occurs in very few tables. What it does have is a long tail -- at
``top_n = 5000``, 19 of 78 queries reach ``|D| > 50`` and 9 reach ``|D| > 100``
(max 269). This module keeps exactly that tail (``min_n_D``), which is the only
population on which an ``alpha`` sweep can show anything.

**What cannot be reported here.** santosLarge ships no groundtruth CSV, so there
is no precision/recall -- those stay on santos3. And the brute-force
reachability check (``experiments.fraction_reachability``) is not computable at
this scale: ``C(418, 10)`` is about 1e19 subsets. So "did we reach tau" here
means *the achieved* ``F_R >= tau``, not "was tau attainable in principle" --
the latter is a max-F-at-cardinality-k computation, which is the removed C4
certificate.

**Self-retrieval is kept, by decision.** All but 4 of santosLarge's 80 query
tables are also in its datalake, so a query can retrieve and select *itself* --
a trivially perfect union candidate (every column matches itself at cosine 1.0)
whose distribution is then counted twice under ``include_query=True``. The user
ruled on 2026-08-10 that this is acceptable, so ``D`` is left as retrieved and
matches the behaviour of every earlier report. ``self_in_D``/``self_in_R`` are
still recorded per row so the effect stays measurable rather than invisible.
"""
import csv
import os
import time
from typing import List, NamedTuple, Optional

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import F as F_ratio
from duts.stats import aggregate, drop_empty
from duts.types import CandidateStats
from dutsx import registry
from dutsx.runner import RunnerContext, retrieve_unscored_candidates

from . import context as ctx_mod

BENCHMARK = "santosLarge"


class BuildTimes(NamedTuple):
    """One-off, query-independent index construction -- amortized over all
    queries, so reported separately and never folded into per-query means."""
    n_datalake_tables: int
    load_resources_s: float
    hnsw_build_s: float
    inverted_index_build_s: float
    total_index_s: float
    parse_recovered: int      # tables that needed the tolerant CSV reader


class LargeRow(NamedTuple):
    q_table: str
    attr: int
    arm: str                 # always "include_self" -- self-retrieval kept by decision
    k: int
    alpha: float
    tau: float
    n_D: int
    pool_size_requested: int   # int(alpha*k)
    clamped: bool              # C5 fired: alpha*k >= |D|, so P = D and Stage 1 is a no-op
    # ---- outcome ----
    feasible: bool
    tau_reached: bool          # F_R >= tau on a non-empty result
    n_P: int
    n_R: int
    n_unionability_computations: int
    F_Q: Optional[float]
    F_P: Optional[float]
    F_R: Optional[float]
    abs_gap_R: Optional[float]
    sum_U: Optional[float]
    self_in_D: bool
    self_in_R: bool
    # ---- Stage 1 ----
    dinkelbach_ran: bool
    dinkelbach_iterations: int
    # ---- LP pre-check ----
    lp_ran: bool
    lp_feasible: Optional[bool]
    lp_decisive: bool
    lp_time_s: Optional[float]
    milp_time_s: Optional[float]
    # ---- per-stage time ----
    retrieval_time_s: float
    stage1_time_s: float
    scoring_time_s: float
    stage2_time_s: float
    end_to_end_s: float
    # ---- skip-Stage-1 arm ----
    skip_n_scored: int
    skip_scoring_time_s: float
    skip_stage2_time_s: float
    skip_end_to_end_s: float
    skip_feasible: bool
    skip_F_R: Optional[float]
    skip_abs_gap_R: Optional[float]
    skip_sum_U: Optional[float]


def _score(cands, ctx, q_table, q_attr_idx) -> List[CandidateStats]:
    """Duplicated from ``dutsx.runner._score_pool`` on purpose -- that function
    documents exactly one call site that is not ``D``, which the skip-Stage-1
    arm would falsify."""
    return [
        CandidateStats(table=c.table, N=c.N, n=c.n,
                       U=ctx.unionability.score(q_table, c.table, (q_attr_idx, c.attr)))
        for c in cands
    ]


def build_context(top_n_unused: int = 0, unionability: str = "pinned_match"):
    """Build the santosLarge context once, timing each index separately."""
    paths = ctx_mod.paths_for(BENCHMARK)
    t0 = time.perf_counter()
    synopsis, dl_files, dl_vecs, q_vecs = ctx_mod.load_shared_resources(paths=paths)
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    semantic_vectors = ctx_mod.categorical_only_vectors(dl_vecs, synopsis, dl_files)
    # Cached graph -> reproducible |D| across runs (experiments/semantic_cache.py)
    from experiments.semantic_cache import get_or_build
    semantic = get_or_build("santosLarge", semantic_vectors, ctx_mod.SIGMA, ctx_mod.THETA_CAT)
    t_hnsw = time.perf_counter() - t1

    t2 = time.perf_counter()
    overlap = registry.build("overlap", "inverted_index", synopsis=synopsis, tables=dl_files)
    t_inv = time.perf_counter() - t2

    scorer = registry.build(
        "unionability", unionability, query_vectors=q_vecs,
        candidate_vectors=dl_vecs, threshold=ctx_mod.SIGMA,
    )
    ctx = RunnerContext(synopsis=synopsis, semantic=semantic, overlap=overlap,
                        unionability=scorer, query_vectors=q_vecs)
    times = BuildTimes(
        n_datalake_tables=len(dl_files), load_resources_s=t_load, hnsw_build_s=t_hnsw,
        inverted_index_build_s=t_inv, total_index_s=t_load + t_hnsw + t_inv,
        parse_recovered=len(getattr(synopsis, "parse_recovered", ())),
    )
    return ctx, paths, times


def run_study(
    top_n: int = 5000,
    min_n_D: int = 50,
    ks=(10, 20),
    alphas=(2.0, 3.0),
    F_star: float = 0.3,
    delta: float = 0.15,
    verbose: bool = True,
):
    ctx, paths, build_times = build_context()
    if verbose:
        print("index build: load {:.1f}s + hnsw {:.1f}s + inverted {:.1f}s = {:.1f}s "
              "over {} tables ({} needed the tolerant CSV reader)".format(
                  build_times.load_resources_s, build_times.hnsw_build_s,
                  build_times.inverted_index_build_s, build_times.total_index_s,
                  build_times.n_datalake_tables, build_times.parse_recovered), flush=True)

    tau = F_star - delta
    base, skipped = ctx_mod.load_base_tasks(
        k=ks[0], alpha=alphas[0], F_star=F_star, delta=delta, top_n=top_n, paths=paths)
    if verbose:
        print("{} queries loaded ({} skipped)".format(len(base), len(skipped)), flush=True)

    # ---- retrieve ONCE per query; D depends only on top_n, not on k or alpha ----
    cohort = []
    t_retr_total = 0.0
    for task in base:
        try:
            t0 = time.perf_counter()
            D, N_Q, n_Q, _ = retrieve_unscored_candidates(task, ctx)
            rt = time.perf_counter() - t0
        except Exception:
            continue
        D = drop_empty(D)
        t_retr_total += rt
        if len(D) >= min_n_D:
            cohort.append((task, D, N_Q, n_Q, rt))
    if verbose:
        print("cohort: {} of {} queries have |D| >= {} at top_n={} "
              "(mean retrieval {:.1f} ms over all queries)".format(
                  len(cohort), len(base), min_n_D, top_n,
                  1000 * t_retr_total / max(len(base), 1)), flush=True)

    rows: List[LargeRow] = []
    for task, D_full, N_Q, n_Q, retr_t in cohort:
        F_Q = (N_Q / n_Q) if n_Q else None
        # Self-retrieval kept by decision (see module docstring) -- D as retrieved.
        D = D_full
        self_in_D = any(d.table == task.q_table for d in D_full)
        if len(D) >= max(ks):
            for k in ks:
                for alpha in alphas:
                    t_e2e = time.perf_counter()
                    pool_size = int(alpha * k)
                    clamped = pool_size >= len(D)

                    t0 = time.perf_counter()
                    if clamped:
                        P_un, iters, dink_ran = D, 0, False
                        F_P = F_ratio(*aggregate(P_un, N_Q, n_Q, True))
                    else:
                        s1 = stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, True)
                        P_un, iters, dink_ran = s1.selected, s1.iterations, True
                        F_P = s1.objective_value
                    stage1_t = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    P = _score(P_un, ctx, task.q_table, int(task.attr))
                    scoring_t = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    r2 = stage2_ilp.solve(P, k, tau, N_Q, n_Q, True)
                    stage2_t = time.perf_counter() - t0
                    e2e = retr_t + (time.perf_counter() - t_e2e)
                    info = r2.info or {}
                    F_R = F_ratio(*aggregate(r2.selected, N_Q, n_Q, True)) if r2.feasible else None

                    # ---- skip Stage 1: all of D into Stage 2 ----
                    t_sk = time.perf_counter()
                    t0 = time.perf_counter()
                    D_scored = _score(D, ctx, task.q_table, int(task.attr))
                    skip_score_t = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    r2s = stage2_ilp.solve(D_scored, k, tau, N_Q, n_Q, True)
                    skip_s2_t = time.perf_counter() - t0
                    skip_e2e = retr_t + (time.perf_counter() - t_sk)
                    skip_F_R = (F_ratio(*aggregate(r2s.selected, N_Q, n_Q, True))
                                if r2s.feasible else None)

                    rows.append(LargeRow(
                        q_table=task.q_table, attr=int(task.attr), arm="include_self", k=k, alpha=alpha,
                        tau=tau, n_D=len(D), pool_size_requested=pool_size, clamped=clamped,
                        feasible=r2.feasible,
                        tau_reached=bool(r2.feasible and F_R is not None and F_R >= tau - 1e-12),
                        n_P=len(P), n_R=len(r2.selected), n_unionability_computations=len(P_un),
                        F_Q=F_Q, F_P=F_P, F_R=F_R,
                        abs_gap_R=abs(F_star - F_R) if F_R is not None else None,
                        sum_U=r2.objective_value if r2.feasible else None,
                        self_in_D=self_in_D,
                        self_in_R=any(c.table == task.q_table for c in r2.selected),
                        dinkelbach_ran=dink_ran, dinkelbach_iterations=iters,
                        lp_ran=bool(info.get("lp_precheck_ran", False)),
                        lp_feasible=info.get("lp_feasible"),
                        lp_decisive=bool(info.get("lp_precheck_ran") and
                                         info.get("lp_feasible") is False),
                        lp_time_s=info.get("lp_time_s"),
                        milp_time_s=info.get("milp_time_s"),
                        retrieval_time_s=retr_t, stage1_time_s=stage1_t,
                        scoring_time_s=scoring_t, stage2_time_s=stage2_t, end_to_end_s=e2e,
                        skip_n_scored=len(D), skip_scoring_time_s=skip_score_t,
                        skip_stage2_time_s=skip_s2_t, skip_end_to_end_s=skip_e2e,
                        skip_feasible=r2s.feasible, skip_F_R=skip_F_R,
                        skip_abs_gap_R=abs(F_star - skip_F_R) if skip_F_R is not None else None,
                        skip_sum_U=r2s.objective_value if r2s.feasible else None,
                    ))
        if verbose:
            print("  done {}".format(task.q_table), flush=True)
    return rows, build_times


def write_rows(rows: List[LargeRow], name: str, output_dir: str = "experiments/results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LargeRow._fields)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    return path
