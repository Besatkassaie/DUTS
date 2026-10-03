"""Unclamped shared-cohort selection for WDC -- santosLarge-style rigor, applied here.

**The gap this fixes**: ``wdc_tau_sweep_shared.py``'s cohort-selection grid (``k in
{10,...,50} x alpha in {10,...,60}``, max pool 3000) only checked FEASIBILITY against
whatever pool ``P`` resulted -- clamped (``P = D`` when ``|D| < alpha*k``) or not. It never
required ``|D| >= max(alpha*k)`` the way ``santoslarge_maxd_selection.py``'s cohort did
(``|D| >= 60`` for its ``k in {10,15} x alpha in {2,3,4}`` grid, chosen specifically "so
every retained query stays unclamped across all six cells" -- see
``experiments/beamer/duts_presentation.tex``'s santosLarge grid slide). Measured directly:
87.1% of the 900 (query, cell) pairs in the original WDC grid were already clamped at
``top_n=5000``, even the best query (``|D|=563``) had 20/30 cells clamped. This module
applies the SAME unclamped-across-the-whole-grid discipline to WDC, on ``tier_10k`` and
``tier_100k`` jointly (``tier_1m`` out of scope here).

**Method** (three steps, same top_n=5000 throughout -- unchanged from the existing shared
selection):

1. **Expand the candidate pool.** ``wdc_maxd_selection_shared.py::run(n_candidates=N)``
   draws candidates by continuing the SAME ``random.Random(seed=42)`` shuffle over
   ``tier_10k``'s table universe that the original 60-candidate run used -- the first 60
   of a larger draw are (almost) the same 60 already in
   ``wdc_shared_maxd_selection_report.csv`` (a small amount of non-determinism was
   observed and is documented in ``wdc_unclamped_selection_notes.md``: 5/60 rows differ
   in which tied-proxy-score (attr, value) was picked, not in whether real |D| was
   measured correctly). This module draws 210 total (60 old + 150 new) -- enough to give
   the grid below a healthy margin over the >=30 target.
2. **Derive the grid from the observed distribution, santosLarge-style**, instead of
   reusing the old ambitious 30-cell grid. Uses ``min(D_10k, D_100k)`` per candidate as
   the binding constraint (the smaller tier is always at least as hard to clear).
3. **Retain every candidate with ``min(D_10k, D_100k) >= max(alpha*k)``** -- guaranteed
   unclamped on BOTH tiers, at every cell of the chosen grid, exactly mirroring
   santosLarge's criterion.

Chosen grid: ``k in {5, 10, 20} x alpha in {5, 10}`` (6 cells, pools
25/50/100/50/100/200 -- max 200) -- same cell count as santosLarge's grid, deliberately
more ambitious in absolute pool size (santosLarge's max was 60) since WDC's realized
|D| distribution supports it: median 52, but a healthy right tail (p75=280, p90=416,
max=562) once you widen the candidate pool. Retention: 64/210 -- comfortably clears the
>=30 target and exceeds santosLarge's retention count (47/78) in absolute terms.

Outputs:
  - ``experiments/results/wdc_unclamped_candidate_pool.csv`` -- all 210 candidates
    considered, one row per candidate (table, attr, value, D_10k, D_100k).
  - ``experiments/results/wdc_unclamped_final_cohort.csv`` -- the retained cohort.
  - ``experiments/results/wdc_unclamped_selection_notes.md`` -- the threshold curve,
    chosen grid, and honest tradeoffs.

Usage:
    /u6/bkassaie/.conda/envs/TableUnionNew/bin/python -m experiments.wdc.wdc_maxd_selection_unclamped
"""
import csv
import os
from typing import List, NamedTuple, Tuple

from .wdc_maxd_selection_shared import run as run_maxd_selection

REPO = "/u6/bkassaie/DUTS"
N_CANDIDATES = 210  # 60 old (near-reproduced) + 150 new
TOP_N = 5000  # unchanged from wdc_maxd_selection_shared.py / wdc_tau_sweep_shared.py

# Grid derived from the observed min(D_10k, D_100k) distribution over 210 candidates
# (median 52, p75 280, p90 416, max 562) -- same 6-cell shape as santosLarge's grid,
# ceiling raised to 200 (vs. santosLarge's 60) since WDC's tail supports it.
KS = (5, 10, 20)
ALPHAS = (5.0, 10.0)
GRID_CELLS = [(k, a, int(k * a)) for k in KS for a in ALPHAS]
MAX_POOL = max(pool for _, _, pool in GRID_CELLS)  # 200
THRESHOLD_CANDIDATES = (20, 30, 40, 50, 60, 80, 100, 150, 200, 300)

CANDIDATE_POOL_CSV = os.path.join(REPO, "experiments/results/wdc_unclamped_candidate_pool.csv")
FINAL_COHORT_CSV = os.path.join(REPO, "experiments/results/wdc_unclamped_final_cohort.csv")
NOTES_MD = os.path.join(REPO, "experiments/results/wdc_unclamped_selection_notes.md")


class PoolRow(NamedTuple):
    q_table: str
    attr: int
    value: str
    n_D_tier_10k: int
    n_D_tier_100k: int
    min_D: int


def build_candidate_pool(n_candidates: int = N_CANDIDATES, verbose: bool = True) -> List[PoolRow]:
    rows = run_maxd_selection(n_candidates=n_candidates, top_n=TOP_N, verbose=verbose)
    return [
        PoolRow(r.q_table, r.attr, r.value, r.n_D_tier_10k, r.n_D_tier_100k,
                min(r.n_D_tier_10k, r.n_D_tier_100k))
        for r in rows
    ]


def threshold_curve(pool: List[PoolRow], thresholds=THRESHOLD_CANDIDATES) -> List[Tuple[int, int]]:
    return [(t, sum(1 for r in pool if r.min_D >= t)) for t in thresholds]


def select_final_cohort(pool: List[PoolRow], max_pool: int = MAX_POOL) -> List[PoolRow]:
    """Unclamped on BOTH tiers at every grid cell <=> min(D_10k, D_100k) >= max(alpha*k)."""
    return [r for r in pool if r.min_D >= max_pool]


def write_pool_csv(pool: List[PoolRow], path: str = CANDIDATE_POOL_CSV) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PoolRow._fields)
        for r in pool:
            w.writerow(r)


def write_cohort_csv(cohort: List[PoolRow], path: str = FINAL_COHORT_CSV) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_table", "attr", "value", "n_D_tier_10k", "n_D_tier_100k"])
        for r in cohort:
            w.writerow([r.q_table, r.attr, r.value, r.n_D_tier_10k, r.n_D_tier_100k])


def write_notes(pool: List[PoolRow], cohort: List[PoolRow], curve: List[Tuple[int, int]],
                 path: str = NOTES_MD) -> None:
    lines = []
    lines.append("# WDC unclamped shared cohort -- selection notes\n")
    lines.append(
        "Fixes the methodology gap vs. santosLarge: the original `wdc_tau_sweep_shared.py` "
        "grid (`k in {10..50} x alpha in {10..60}`, max pool 3000) never required `|D| >= "
        "max(alpha*k)`, so 87.1% of its (query, cell) pairs were silently clamped at "
        "`top_n=5000` (even the best query, |D|=563, had 20/30 cells clamped). This mirrors "
        "santosLarge's `|D| >= max(alpha*k)` cohort criterion instead.\n"
    )
    lines.append("## Candidate pool\n")
    lines.append(
        "%d candidates (60 reproduced from the original seed=42 draw + 150 new, same "
        "continued shuffle over `tier_10k`'s table universe), `top_n=5000` fixed "
        "throughout, `(attr, value)` chosen by the existing maxD proxy-rank + top-20 "
        "real-retrieval-verify method (`wdc_maxd_selection_shared.py`), evaluated against "
        "`tier_100k`.\n" % len(pool)
    )
    lines.append(
        "**Reproducibility caveat**: 5 of the first 60 rows differ from the original "
        "`wdc_shared_maxd_selection_report.csv` in which tied-proxy-score `(attr, value)` "
        "was picked for that table (not in whether real |D| was measured correctly) -- a "
        "pre-existing tie-break non-determinism in `_proxy_rank`/the selection loop, not "
        "introduced here. Real |D| values throughout are freshly, correctly measured.\n"
    )
    lines.append(
        "**Non-monotonicity**: 24/210 (11.4%) candidates have `D_10k > D_100k`, despite "
        "`tier_10k` being a literal subset of `tier_100k`. This is the same, already-"
        "documented `top_n` budget-crowding effect (`tier_100k` has 10x more tables "
        "competing for the same fixed `top_n=5000` semantic-probe slots) -- not a bug. "
        "Using `min(D_10k, D_100k)` as the binding constraint is unaffected by direction.\n"
    )
    lines.append("## Threshold curve (candidates retained vs. `min(D_10k, D_100k)` threshold)\n")
    lines.append("| threshold | retained | fraction |")
    lines.append("|---:|---:|---:|")
    for t, n in curve:
        lines.append("| %d | %d/%d | %.1f%% |" % (t, n, len(pool), 100.0 * n / len(pool)))
    lines.append("")
    lines.append("## Chosen grid\n")
    lines.append(
        "`k in {5, 10, 20} x alpha in {5, 10}` (6 cells, same count as santosLarge's grid) "
        "-- pools: %s (max %d).\n" % (
            sorted(set(p for _, _, p in GRID_CELLS)), MAX_POOL
        )
    )
    lines.append(
        "Chosen over the original 30-cell grid (max pool 3000, unachievable unclamped on "
        "`tier_10k` at this corpus/retrieval scale -- max observed |D| there is 1450 even "
        "at `top_n=30000`) and over keeping santosLarge's exact ceiling (60), since WDC's "
        "distribution (median 52, p75 280, p90 416, max 562) supports a materially larger "
        "pool once the candidate draw is widened to 210.\n"
    )
    lines.append("## Retained cohort\n")
    lines.append(
        "**%d of %d candidates** (%.1f%%) retained -- `min(D_10k, D_100k) >= %d`, "
        "guaranteeing unclamped Stage 1 on BOTH tiers at every one of the 6 grid cells. "
        "Compare santosLarge: 47/78 (60.3%%) at its own threshold/grid.\n" % (
            len(cohort), len(pool), 100.0 * len(cohort) / len(pool), MAX_POOL
        )
    )
    n_10k_alone = sum(1 for r in pool if r.n_D_tier_10k >= MAX_POOL)
    n_100k_alone = sum(1 for r in pool if r.n_D_tier_100k >= MAX_POOL)
    lines.append(
        "`tier_10k`-alone retention (`D_10k >= %d`, ignoring `tier_100k`): %d/%d. "
        "`tier_100k`-alone retention (`D_100k >= %d`, ignoring `tier_10k`): %d/%d. "
        "Joint (both tiers, the actual cohort criterion): %d/%d.\n" % (
            MAX_POOL, n_10k_alone, len(pool),
            MAX_POOL, n_100k_alone, len(pool),
            len(cohort), len(pool),
        )
    )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    pool = build_candidate_pool()
    write_pool_csv(pool)
    print("wrote", CANDIDATE_POOL_CSV, "(%d rows)" % len(pool))

    curve = threshold_curve(pool)
    for t, n in curve:
        print("threshold %4d -> %d/%d retained" % (t, n, len(pool)))

    cohort = select_final_cohort(pool)
    write_cohort_csv(cohort)
    print("wrote", FINAL_COHORT_CSV, "(%d rows, unclamped on both tiers at max pool %d)" % (
        len(cohort), MAX_POOL))

    write_notes(pool, cohort, curve)
    print("wrote", NOTES_MD)
