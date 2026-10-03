"""Split WDC retrieval time into its parts, per tier, on the 67-query cohort (top_n=5000).

Parts (same calls as dutsx.retrieval.retrieve_candidates / runner.retrieve_unscored_candidates):
  sem   = HNSW knn_query (top_n=5000)          -- run twice: first call and immediate repeat (warm)
  ovl   = overlap.query(M) posting-list probe  -- run twice
  join  = build sem dict + set intersection + best-per-table
  look  = N_i / n_i synopsis lookup for |D|
Also logs |D_sem|, |D_ovl|, |D_pair|, |D|.
"""
import csv
import os
import resource
import sys
import time

from dutsx.retrieval import retrieve_candidates
from dutsx.runner import N_of
from .context import build_wdc_context
from .wdc_end2end_timing import REPO, TOP_N, _rss_gb, build_tier_1m_ctx
from .wdc_topn_pool_sweep import _load_cohort

FIELDS = ["tier", "q_table", "sem_first_s", "sem_repeat_s", "ovl_first_s", "ovl_repeat_s", "join_s",
          "lookup_s", "total_s", "n_sem", "n_ovl", "n_pair", "n_D"]


def run_tier(tier):
    cohort = _load_cohort()
    ctx = build_tier_1m_ctx() if tier == "tier_1m" else build_wdc_context(tier)[0]
    print("[{}] ctx ready rss {:.1f}GB".format(tier, _rss_gb()), flush=True)
    rows = []
    for qi, (q_table, attr, value) in enumerate(cohort):
        M = {value}
        qv = ctx.query_vectors.get(q_table)
        if qv is None:
            continue
        v = qv[int(attr)]
        try:
            t = time.perf_counter(); d_sem = ctx.semantic.query(v, TOP_N); s1 = time.perf_counter() - t
            t = time.perf_counter(); ctx.semantic.query(v, TOP_N); s2 = time.perf_counter() - t
            t = time.perf_counter(); d_ovl = ctx.overlap.query(M); o1 = time.perf_counter() - t
            t = time.perf_counter(); ctx.overlap.query(M); o2 = time.perf_counter() - t
        except Exception as e:
            print("skip", q_table, e, flush=True)
            continue
        # join: replicate retrieve_candidates' post-processing on already-fetched lists
        t = time.perf_counter()
        sem_pairs = {(tb, a): s for (tb, a, s) in d_sem}
        pk = set(sem_pairs) & set(d_ovl)
        best = {}
        for (tb, a) in pk:
            sim = sem_pairs[(tb, a)]
            cur = best.get((tb))
            if cur is None or sim > cur[1]:
                best[tb] = (a, sim)
        join_s = time.perf_counter() - t
        t = time.perf_counter()
        for tb, (a, _) in best.items():
            if ctx.synopsis.n_rows(tb):
                N_of(ctx.synopsis.distribution(tb, a), M)
        look_s = time.perf_counter() - t
        rows.append(dict(tier=tier, q_table=q_table, sem_first_s=s1, sem_repeat_s=s2, ovl_first_s=o1,
                         ovl_repeat_s=o2, join_s=join_s, lookup_s=look_s,
                         total_s=s1 + o1 + join_s + look_s, n_sem=len(sem_pairs), n_ovl=len(d_ovl),
                         n_pair=len(pk), n_D=len(best)))
        if (qi + 1) % 10 == 0:
            print("  [{}] {}/{} rss {:.1f}GB".format(tier, qi + 1, len(cohort), _rss_gb()), flush=True)
    return rows


if __name__ == "__main__":
    tier = sys.argv[1]
    rows = run_tier(tier)
    out = os.path.join(REPO, "experiments/results/wdc_retrieval_breakdown_{}.csv".format(tier))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    print("wrote", out, len(rows), "peak rss {:.1f}GB".format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), flush=True)
