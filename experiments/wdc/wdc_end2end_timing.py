"""Per-stage and end-to-end execution time on WDC with REAL unionability, all three tiers.

Grid: top_n=5000, alpha in {10,5}, k in {5,10,20}; F*=0.20, delta=0.10 (tau=0.10); the 67-query
cohort. Composition follows ``dutsx.runner.run_query`` exactly, but retrieval is timed ONCE per
query and reused across the 6 cells (it does not depend on k/alpha):

    retrieval (HNSW probe + overlap filter) -> Stage 1 (Dinkelbach, or C5 clamp) ->
    unionability scoring of P only (real pinned-match, |P| = alpha*k matchings) ->
    Stage 2 (LP pre-check + MILP)

``end2end = retrieval + stage1 + scoring + stage2``. Real U (``PinnedMatchScorer``, sigma=0.6),
so this supersedes wdc_stage2_timing.py's synthetic-U Stage 2 numbers.
"""
import csv
import os
import resource
import time
from typing import List, NamedTuple

from duts import stage1_dinkelbach, stage2_ilp
from duts.stats import drop_empty
from dutsx import registry
from dutsx.adapters.unionability import PinnedMatchScorer, load_starmie_vectors
from dutsx.runner import QueryTask, RunnerContext, _score_pool, retrieve_unscored_candidates

from .context import SIGMA, THETA_CAT, build_wdc_context, paths_for_tier
from .wdc_topn_pool_sweep import _cache_paths, _load_cached_semantic_index, _load_cohort

REPO = "/u6/bkassaie/DUTS"
TOP_N = 5000
KS = (5, 10, 20)
ALPHAS = (10.0, 5.0)
F_STAR, DELTA = 0.20, 0.10
TAU = F_STAR - DELTA


class Row(NamedTuple):
    tier: str
    q_table: str
    k: int
    alpha: float
    pool: int
    n_D: int
    clamped: bool
    feasible: bool
    cause: str
    n_scored: int
    retrieval_s: float
    stage1_s: float
    scoring_s: float
    lp_s: float
    milp_s: float
    stage2_s: float
    end2end_s: float


def _rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def build_tier_1m_ctx() -> RunnerContext:
    """Cached lean HNSW index + full vectors dict for real scoring (no rebuild)."""
    paths = paths_for_tier("tier_1m")
    index_path, meta_path, _ = _cache_paths("tier_1m", 16)
    semantic = _load_cached_semantic_index(index_path, meta_path, SIGMA)
    print("[tier_1m] index loaded, rss {:.1f}GB".format(_rss_gb()), flush=True)
    tables = sorted(f for f in os.listdir(paths.csv_dir) if f.endswith(".csv"))
    synopsis = registry.build("synopsis", "metadata_store", pkl_path=paths.metadata_pkl,
                              theta_cat=THETA_CAT)
    overlap = registry.build("overlap", "inverted_index", synopsis=synopsis, tables=tables)
    print("[tier_1m] overlap built, rss {:.1f}GB".format(_rss_gb()), flush=True)
    vectors = load_starmie_vectors(paths.vectors_pkl)
    print("[tier_1m] vectors loaded, rss {:.1f}GB".format(_rss_gb()), flush=True)
    scorer = PinnedMatchScorer(vectors, vectors, threshold=SIGMA)
    return RunnerContext(synopsis=synopsis, semantic=semantic, overlap=overlap,
                         unionability=scorer, query_vectors=vectors)


def run_tier(tier: str) -> List[Row]:
    cohort = _load_cohort()
    t0 = time.perf_counter()
    ctx = build_tier_1m_ctx() if tier == "tier_1m" else build_wdc_context(tier)[0]
    print("[{}] context ready in {:.1f}s, rss {:.1f}GB".format(
        tier, time.perf_counter() - t0, _rss_gb()), flush=True)

    rows: List[Row] = []
    for qi, (q_table, attr, value) in enumerate(cohort):
        task = QueryTask(q_table=q_table, attr=attr, M={value}, F_star=F_STAR, delta=DELTA,
                         k=KS[0], alpha=ALPHAS[0], include_query=True, top_n=TOP_N)
        try:
            t = time.perf_counter()
            D, N_Q, n_Q, _ = retrieve_unscored_candidates(task, ctx)
            retrieval_s = time.perf_counter() - t
        except Exception:
            continue
        D = drop_empty(D)
        if not D:
            continue
        for alpha in ALPHAS:
            for k in KS:
                pool_size = int(alpha * k)
                if len(D) < k:
                    rows.append(Row(tier, q_table, k, alpha, pool_size, len(D), True, False,
                                    "insufficient_candidates", 0, retrieval_s, 0, 0, 0, 0, 0,
                                    retrieval_s))
                    continue
                clamped = pool_size >= len(D)
                t = time.perf_counter()
                P_un = D if clamped else stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, True).selected
                stage1_s = time.perf_counter() - t
                t = time.perf_counter()
                P = _score_pool(P_un, ctx, q_table, attr)
                scoring_s = time.perf_counter() - t
                t = time.perf_counter()
                r = stage2_ilp.solve(P, k, TAU, N_Q, n_Q, True)
                stage2_s = time.perf_counter() - t
                info = r.info or {}
                rows.append(Row(
                    tier, q_table, k, alpha, pool_size, len(D), clamped, r.feasible,
                    "" if r.feasible else "stage2_infeasible", len(P_un), retrieval_s, stage1_s,
                    scoring_s, info.get("lp_time_s") or 0.0, info.get("milp_time_s") or 0.0,
                    stage2_s, retrieval_s + stage1_s + scoring_s + stage2_s,
                ))
        if (qi + 1) % 10 == 0:
            print("  [{}] {}/{} queries, rss {:.1f}GB".format(
                tier, qi + 1, len(cohort), _rss_gb()), flush=True)
    return rows


if __name__ == "__main__":
    import sys
    tier = sys.argv[1]
    rows = run_tier(tier)
    out = os.path.join(REPO, "experiments/results/wdc_end2end_timing_{}.csv".format(tier))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(Row._fields)
        for r in rows:
            w.writerow(r)
    print("wrote", out, "({} rows)".format(len(rows)),
          "peak rss {:.1f}GB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6),
          flush=True)
