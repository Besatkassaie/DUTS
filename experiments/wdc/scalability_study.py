"""WDC at scale: the tier-sweep scalability study (WDC scalability study Phase 6).

Modeled directly on ``experiments/santoslarge_study.py``'s per-(query,k,alpha) timing loop --
reuses ``duts.stage1_dinkelbach``, ``duts.stage2_ilp``, and ``dutsx.runner.
retrieve_unscored_candidates`` exactly as-is (already benchmark-agnostic), wrapped in a tier loop.
The only genuinely new code here is: (a) the tier loop itself, (b) sourcing
``embedding_extraction_s`` from Phase 3's own timing sidecar rather than re-timing inline (that GPU
step is a separate offline process by design), and (c) the retrieval-yield / column-coverage
reporting Phase 4/2 add on top of the santosLarge-style per-stage timing.

**Self-retrieval.** WDC has no query/datalake split at all -- every query is a member of its own
tier's table universe, so it is trivially self-retrievable by construction, more so than
santosLarge's "all but 4". Kept, and recorded (``self_in_D``/``self_in_R``), same precedent.
"""
import csv
import os
import time
from typing import List, NamedTuple, Optional

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import F as F_ratio
from duts.stats import aggregate, drop_empty
from duts.types import CandidateStats
from dutsx.runner import retrieve_unscored_candidates

from .context import build_wdc_context, paths_for_tier
from .query_selection import select_queries_with_comparison


class WdcRow(NamedTuple):
    tier: str
    q_table: str
    attr: int
    k: int
    alpha: float
    tau: float
    n_D: int
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
    abs_gap_R: Optional[float]
    sum_U: Optional[float]
    self_in_D: bool
    self_in_R: bool
    dinkelbach_ran: bool
    dinkelbach_iterations: int
    lp_ran: bool
    lp_feasible: Optional[bool]
    lp_decisive: bool
    lp_time_s: Optional[float]
    retrieval_time_s: float
    stage1_time_s: float
    scoring_time_s: float
    stage2_time_s: float
    end_to_end_s: float
    skip_n_scored: int
    skip_scoring_time_s: float
    skip_stage2_time_s: float
    skip_end_to_end_s: float
    skip_feasible: bool
    skip_F_R: Optional[float]
    skip_abs_gap_R: Optional[float]
    skip_sum_U: Optional[float]


def _score(cands, ctx, q_table, q_attr_idx) -> List[CandidateStats]:
    """Duplicated from ``dutsx.runner._score_pool`` on purpose, matching
    ``santoslarge_study.py``'s own precedent -- the skip-Stage-1 arm needs
    to score a DIFFERENT set (all of D) than the one real call site scores."""
    return [
        CandidateStats(table=c.table, N=c.N, n=c.n,
                       U=ctx.unionability.score(q_table, c.table, (q_attr_idx, c.attr)))
        for c in cands
    ]


def run_tier_study(
    tier: str,
    n_queries: int = 30,
    ks=(10, 20),
    alphas=(2.0, 3.0),
    F_star: float = 0.3,
    delta: float = 0.15,
    top_n: int = 200,
    seed: int = 42,
    verbose: bool = True,
):
    ctx, build_times = build_wdc_context(tier)
    if verbose:
        print(
            "[{}] index build: csv {} + embed {} + load {:.2f}s + hnsw {:.2f}s + inverted {:.2f}s "
            "= {:.2f}s over {} tables ({}/{} categorical, ratio {:.1%})".format(
                tier, build_times.csv_conversion_s, build_times.embedding_extraction_s,
                build_times.load_resources_s, build_times.hnsw_build_s,
                build_times.inverted_index_build_s, build_times.total_index_s,
                build_times.n_tables, build_times.n_columns_categorical,
                build_times.n_columns_total, build_times.categorical_coverage_ratio,
            ), flush=True,
        )

    tau = F_star - delta
    paths = paths_for_tier(tier)
    tables = sorted(f for f in os.listdir(paths.csv_dir) if f.endswith(".csv"))

    tasks, selection_report = select_queries_with_comparison(
        ctx, tables, n_queries=n_queries, seed=seed, k=ks[0], alpha=alphas[0],
        F_star=F_star, delta=delta, top_n=top_n,
    )
    if verbose:
        y = selection_report["yield"]
        uy = selection_report["unbiased_side_sample"]["yield"]
        print(
            "[{}] {} queries selected (eligibility {:.1%}); realized |D| biased median={} "
            "vs unbiased median={} (paired mean diff {:+.2f})".format(
                tier, len(tasks), selection_report["eligibility_rate"], y["median"],
                uy["median"], selection_report["paired_biased_minus_unbiased_mean"],
            ), flush=True,
        )

    rows: List[WdcRow] = []
    for base_task in tasks:
        t0 = time.perf_counter()
        D_full, N_Q, n_Q, _telemetry = retrieve_unscored_candidates(base_task, ctx)
        retr_t = time.perf_counter() - t0
        D = drop_empty(D_full)
        F_Q = (N_Q / n_Q) if n_Q else None
        self_in_D = any(d.table == base_task.q_table for d in D)

        if len(D) < max(ks):
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
                P = _score(P_un, ctx, task.q_table, int(task.attr))
                scoring_t = time.perf_counter() - t0

                t0 = time.perf_counter()
                r2 = stage2_ilp.solve(P, k, tau, N_Q, n_Q, True)
                stage2_t = time.perf_counter() - t0
                e2e = retr_t + (time.perf_counter() - t_e2e)
                info = r2.info or {}
                F_R = F_ratio(*aggregate(r2.selected, N_Q, n_Q, True)) if r2.feasible else None

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

                rows.append(WdcRow(
                    tier=tier, q_table=task.q_table, attr=int(task.attr), k=k, alpha=alpha,
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

    return rows, build_times, selection_report


def write_rows(rows: List[WdcRow], name: str, output_dir: str = "experiments/results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(WdcRow._fields)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    return path
