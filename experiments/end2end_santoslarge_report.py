"""Per-step runtime vs. candidate pool size (`alpha*k`), santosLarge.

Reuses `experiments/results/santoslarge_maxd_topn5000_D60.csv` (top_n=5000,
F*=0.122, delta=0.07, the 46-query maxD cohort) -- no rerun needed, since
every metric here is already a column on that CSV.

**Five metrics**, each reported as mean/median/max/min (ms) across the 46
queries, per candidate pool size:

  - index_probe_total   <- retrieval_time_s   (HNSW semantic probe + posting-
                           list overlap probe + their intersection/combine +
                           per-candidate N_i/n_i synopsis lookup -- i.e.
                           everything that happens before Stage 1 runs)
  - stage1_total         <- stage1_time_s      (Dinkelbach, or the C5 clamp)
  - stage2_total         <- stage2_time_s      (LP pre-check + milp solve +
                           constraint setup/post-solve verification, in full)
  - unionability_total   <- scoring_time_s     (computing U for each of the
                           |P| candidates Stage 1 selected -- NOT part of
                           either stage: it runs strictly between them)
  - end_to_end_total     <- end_to_end_s       (sum of all of the above, per
                           query, for the two-stage arm)

**Pool size, not (k,alpha), is the x-axis.** The grid k in {10,15} x alpha in
{2,3,4} produces pool=alpha*k in {20,30,40,30,45,60} -- pool=30 is reached by
TWO different (k,alpha) pairs (k=10,alpha=3 and k=15,alpha=2). Per the user's
instruction, when two (k,alpha) combos share a pool size, each reported
statistic (mean/median/max/min) at that pool size is the LARGER of the two
combos' values for that statistic -- an elementwise max, not an average or a
whole-combo pick.

Outputs:
  - experiments/results/end2end_santoslarge_raw.csv       (long format, one
    row per (query, k, alpha, metric) -- for figure generation)
  - experiments/results/end2end_santoslarge_by_poolsize.csv (one row per
    (pool_size, metric) with mean/median/max/min -- the resolved table)
"""
import csv
import os
import statistics
from typing import Dict, List, NamedTuple

REPO = "/u6/bkassaie/DUTS"
SRC_CSV = os.path.join(REPO, "experiments/results/santoslarge_maxd_topn5000_D60.csv")
OUT_DIR = os.path.join(REPO, "experiments/results")

METRICS = {
    "index_probe_total": "retrieval_time_s",
    "stage1_total": "stage1_time_s",
    "stage2_total": "stage2_time_s",
    "unionability_total": "scoring_time_s",
    "end_to_end_total": "end_to_end_s",
}


class Stat(NamedTuple):
    mean: float
    median: float
    max: float
    min: float


def _stats_ms(vals_s: List[float]) -> Stat:
    vals_ms = [v * 1000 for v in vals_s]
    return Stat(mean=statistics.mean(vals_ms), median=statistics.median(vals_ms),
                max=max(vals_ms), min=min(vals_ms))


def run():
    with open(SRC_CSV) as f:
        rows = list(csv.DictReader(f))

    combos = sorted(set((r["k"], r["alpha"]) for r in rows),
                     key=lambda t: (float(t[0]), float(t[1])))

    # ---- raw long-format, for figures ----
    raw_path = os.path.join(OUT_DIR, "end2end_santoslarge_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_table", "k", "alpha", "pool_size", "metric", "value_ms"])
        for r in rows:
            pool = int(float(r["k"]) * float(r["alpha"]))
            for mname, field in METRICS.items():
                w.writerow([r["q_table"], r["k"], r["alpha"], pool, mname,
                            float(r[field]) * 1000])

    # ---- per-combo stats ----
    per_combo: Dict[tuple, Dict] = {}
    for k, alpha in combos:
        sub = [r for r in rows if r["k"] == k and r["alpha"] == alpha]
        pool = int(float(k) * float(alpha))
        per_combo[(k, alpha)] = {"pool": pool, "n": len(sub)}
        for mname, field in METRICS.items():
            per_combo[(k, alpha)][mname] = _stats_ms([float(r[field]) for r in sub])

    # ---- resolve by pool size (elementwise max across duplicate combos) ----
    by_pool: Dict[int, List[tuple]] = {}
    for (k, alpha), d in per_combo.items():
        by_pool.setdefault(d["pool"], []).append(((k, alpha), d))

    resolved_path = os.path.join(OUT_DIR, "end2end_santoslarge_by_poolsize.csv")
    with open(resolved_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool_size", "metric", "mean_ms", "median_ms", "max_ms", "min_ms",
                    "source_combos"])
        for pool in sorted(by_pool):
            entries = by_pool[pool]
            combo_str = "+".join("k{}a{}".format(k, a) for (k, a), _ in entries)
            for mname in METRICS:
                mean_v = max(d[mname].mean for _, d in entries)
                median_v = max(d[mname].median for _, d in entries)
                max_v = max(d[mname].max for _, d in entries)
                min_v = max(d[mname].min for _, d in entries)
                w.writerow([pool, mname, mean_v, median_v, max_v, min_v, combo_str])

    return per_combo, by_pool, raw_path, resolved_path


if __name__ == "__main__":
    per_combo, by_pool, raw_path, resolved_path = run()
    print("wrote", raw_path)
    print("wrote", resolved_path)
    for pool in sorted(by_pool):
        combos_here = [c for c, _ in by_pool[pool]]
        print("pool={} from {}".format(pool, combos_here))
