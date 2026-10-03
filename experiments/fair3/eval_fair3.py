"""Precision / recall / ideal recall / feasibility / unionability vs k on the three fairified
benchmarks (santos3, tusSmall3, tusLarge3), with explicit accounting of BYPASSED stages.

Composition = ``dutsx.runner.run_query`` (retrieve -> Stage 1 -> score P -> Stage 2), except that
retrieval is done once per (query, top_n) and reused across the (alpha, k) cells (retrieval does not
depend on them). F*=0.4, delta=0.1 (tau=0.3). Per cell it records:
  * Stage 1: 'ran' (+iterations) | 'clamped' (pool >= |D| so P=D, Stage 1 BYPASSED) | 'na' (|D|<k)
  * Stage 2: 'milp' (LP pre-check feasible, MILP ran) | 'lp_decisive' (LP proved infeasible, MILP
             BYPASSED) | 'na' (|D|<k, both stages bypassed)
  * precision/recall of D, P, R against the fair ground truth; ideal recall = min(k,|gt|)/|gt|
  * sum_U, F_P, F_R
  * a NO-FAIRNESS reference: plain top-k by unionability U over all of D (same scorer)
M is the value set {v} plus its numeric spelling variants ("4" <-> "4.0"): the protected value is
taken from the query table, but datalake columns with missing values store integers as "4.0".
Usage: python -m experiments.fair3.eval_fair3 <benchmark> [top_n ...]
"""
import csv
import os
import sys
import time
from typing import List, NamedTuple, Optional

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import F as F_ratio
from duts.stats import aggregate, drop_empty
from dutsx import registry
from dutsx.adapters.unionability import load_starmie_vectors
from dutsx.runner import RunnerContext, _score_pool, load_queries_from_csv, retrieve_unscored_candidates

from ..context import SIGMA, THETA_CAT, categorical_only_vectors, paths_for
from ..groundtruth import load_groundtruth, precision_recall

RESULTS = "/u6/bkassaie/DUTS/experiments/results"
KS = (5, 10, 20)
ALPHAS = (5.0, 10.0)
F_STAR, DELTA = 0.40, 0.10
TAU = F_STAR - DELTA


def expand_M(value: str):
    M = {value}
    try:
        f = float(value)
    except ValueError:
        return M, False
    if f != f or f in (float("inf"), float("-inf")):
        return M, False
    if f.is_integer():
        M |= {str(int(f)), str(float(f))}
    return M, len(M) > 1


class Row(NamedTuple):
    bench: str
    top_n: int
    q_table: str
    gt_size: int
    M_expanded: bool
    n_D: int
    n_sem: int
    n_ovl: int
    k: int
    alpha: float
    pool: int
    stage1: str
    stage1_iters: int
    stage2: str
    lp_ran: bool
    lp_feasible: Optional[bool]
    milp_ran: bool
    feasible: bool
    n_P: int
    n_R: int
    F_P: Optional[float]
    F_R: Optional[float]
    sum_U: Optional[float]
    prec_D: Optional[float]
    rec_D: Optional[float]
    prec_P: Optional[float]
    rec_P: Optional[float]
    prec_R: Optional[float]
    rec_R: Optional[float]
    ideal_recall: float
    unf_prec: Optional[float]
    unf_rec: Optional[float]
    unf_sum_U: Optional[float]
    unf_F: Optional[float]
    retrieval_s: float
    stage1_s: float
    scoring_s: float
    stage2_s: float


def build_ctx(bench: str):
    p = paths_for(bench)
    root = os.path.dirname(p.query_dir)
    synopsis = registry.build("synopsis", "metadata_store",
                              pkl_path=os.path.join(root, "indexes", "metadata_combined.pkl"),
                              theta_cat=THETA_CAT)
    dl_vecs = load_starmie_vectors(p.datalake_vec_pkl)
    q_vecs = load_starmie_vectors(p.query_vec_pkl)
    files = [f for f in sorted(os.listdir(p.datalake_dir)) if f in dl_vecs and synopsis.has_table(f)]
    sem_vecs = categorical_only_vectors(dl_vecs, synopsis, files)
    semantic = registry.build("semantic", "hnsw", vectors=sem_vecs, sigma=SIGMA)
    overlap = registry.build("overlap", "inverted_index", synopsis=synopsis, tables=files)
    scorer = registry.build("unionability", "pinned_match", query_vectors=q_vecs,
                            candidate_vectors=dl_vecs, threshold=SIGMA)
    return RunnerContext(synopsis=synopsis, semantic=semantic, overlap=overlap,
                         unionability=scorer, query_vectors=q_vecs), p


def gt_path(bench: str, p) -> str:
    alt = os.path.join(os.path.dirname(p.query_dir), "{}_benchmark_groundtruth.csv".format(bench))
    return alt if os.path.isfile(alt) else p.groundtruth_csv


def run(bench: str, top_ns: List[int]) -> None:
    t0 = time.time()
    ctx, p = build_ctx(bench)
    truth = load_groundtruth(gt_path(bench, p))
    tasks, skipped = load_queries_from_csv(p.protected_csv, p.query_dir, k=KS[0], alpha=ALPHAS[0],
                                           F_star=F_STAR, delta=DELTA, include_query=True, top_n=100)
    print("[{}] ctx ready {:.0f}s; {} queries ({} skipped)".format(bench, time.time() - t0, len(tasks), len(skipped)),
          flush=True)
    for top_n in top_ns:
        rows: List[Row] = []
        for qi, base in enumerate(tasks):
            (value,) = base.M
            M, expanded = expand_M(value)
            task = base._replace(M=M, top_n=top_n)
            gt = truth.get(task.q_table, frozenset())
            t = time.perf_counter()
            try:
                D, N_Q, n_Q, tel = retrieve_unscored_candidates(task, ctx)
            except Exception as e:
                print("  retrieval error", task.q_table, e, flush=True)
                continue
            retrieval_s = time.perf_counter() - t
            D = drop_empty(D)
            d_tables = [c.table for c in D]
            q_idx = int(task.attr)
            # no-fairness reference: score all of D once
            scored_all = _score_pool(D, ctx, task.q_table, q_idx) if D else []
            ranked = sorted(scored_all, key=lambda c: (-c.U, c.table))
            for alpha in ALPHAS:
                for k in KS:
                    pool = int(alpha * k)
                    ideal = (min(k, len(gt)) / len(gt)) if gt else 0.0
                    pd_, rd_, _ = precision_recall(d_tables, gt)
                    base_kw = dict(bench=bench, top_n=top_n, q_table=task.q_table, gt_size=len(gt),
                                   M_expanded=expanded, n_D=len(D), n_sem=tel.n_sem, n_ovl=tel.n_ovl,
                                   k=k, alpha=alpha, pool=pool, ideal_recall=ideal,
                                   prec_D=pd_, rec_D=rd_, retrieval_s=retrieval_s)
                    if len(D) < k:
                        rows.append(Row(stage1="na", stage1_iters=0, stage2="na", lp_ran=False, lp_feasible=None,
                                        milp_ran=False, feasible=False, n_P=0, n_R=0, F_P=None, F_R=None,
                                        sum_U=None, prec_P=None, rec_P=None, prec_R=None, rec_R=None,
                                        unf_prec=None, unf_rec=None, unf_sum_U=None, unf_F=None,
                                        stage1_s=0.0, scoring_s=0.0, stage2_s=0.0, **base_kw))
                        continue
                    clamped = pool >= len(D)
                    t = time.perf_counter()
                    if clamped:
                        P_un, iters = D, 0
                    else:
                        r1 = stage1_dinkelbach.solve(D, pool, N_Q, n_Q, True)
                        P_un, iters = r1.selected, r1.iterations
                    stage1_s = time.perf_counter() - t
                    t = time.perf_counter()
                    P = _score_pool(P_un, ctx, task.q_table, q_idx)
                    scoring_s = time.perf_counter() - t
                    t = time.perf_counter()
                    r2 = stage2_ilp.solve(P, k, TAU, N_Q, n_Q, True)
                    stage2_s = time.perf_counter() - t
                    info = r2.info or {}
                    lp_ran = bool(info.get("lp_precheck_ran"))
                    lp_feas = info.get("lp_feasible")
                    decisive = lp_ran and lp_feas is False
                    milp_ran = (info.get("milp_time_s") or 0) > 0
                    R = r2.selected if r2.feasible else []
                    p_t, r_t = [c.table for c in P_un], [c.table for c in R]
                    pp, rp, _ = precision_recall(p_t, gt)
                    pr, rr, _ = precision_recall(r_t, gt)
                    top = ranked[:k]
                    up, ur, _ = precision_recall([c.table for c in top], gt)
                    rows.append(Row(
                        stage1="clamped" if clamped else "ran", stage1_iters=iters,
                        stage2="lp_decisive" if decisive else "milp", lp_ran=lp_ran, lp_feasible=lp_feas,
                        milp_ran=milp_ran, feasible=bool(r2.feasible), n_P=len(P_un), n_R=len(R),
                        F_P=F_ratio(*aggregate(P_un, N_Q, n_Q, True)),
                        F_R=F_ratio(*aggregate(R, N_Q, n_Q, True)) if R else None,
                        sum_U=sum(c.U for c in R) if R else None,
                        prec_P=pp, rec_P=rp, prec_R=pr if R else None, rec_R=rr if R else None,
                        unf_prec=up, unf_rec=ur, unf_sum_U=sum(c.U for c in top),
                        unf_F=F_ratio(*aggregate(top, N_Q, n_Q, True)),
                        stage1_s=stage1_s, scoring_s=scoring_s, stage2_s=stage2_s, **base_kw))
            if (qi + 1) % 10 == 0:
                print("  [{} top_n={}] {}/{} queries".format(bench, top_n, qi + 1, len(tasks)), flush=True)
        out = os.path.join(RESULTS, "fair3_eval_{}_n{}.csv".format(bench, top_n))
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(Row._fields)
            for r in rows:
                w.writerow(r)
        print("[{}] wrote {} ({} rows), total {:.0f}s".format(bench, out, len(rows), time.time() - t0), flush=True)


if __name__ == "__main__":
    bench = sys.argv[1]
    run(bench, [int(x) for x in sys.argv[2:]] or [1000])
