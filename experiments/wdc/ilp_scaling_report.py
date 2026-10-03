"""Stage 2 ILP scaling report vs. pool size (``alpha*k``), WDC tier_10k/tier_100k
(user-requested ad hoc experiment, 2026-08-14 -- not part of ``PLAN-integration.md``'s
phase sequence).

Answers: at fixed retrieval breadth (``top_n=5000``) and fixed ``alpha=10``, how does Stage 2's
ILP solve time scale as ``k`` grows from 10 to 100 (so the *requested* pool ``alpha*k`` grows from
100 to 1000), and how often does Stage 2 actually find a feasible solution -- for both WDC tiers.

Reuses ``topn_alpha_k_sweep.run_topn_sweep`` unmodified (singleton ``top_ns``/``alphas``, the same
union-mode retrieval, per-table dedup ON, ``sigma=0.6`` -- the paper-matching default, same as
every other canonical WDC table produced so far, NOT that module's own ``sigma=0.30`` default).

**"Falls short" tracking, per the user's explicit ask:** two distinct ways a query can fail to
contribute a full-strength cell at a given ``k``, both reported:

- **Skipped** (pre-Stage-1): ``|D_union| < max(ks) = 100``, checked once per query against the
  LARGEST ``k`` in this sweep -- so it's the same skip decision, and the same count, at every row
  of a tier's table (``run_topn_sweep``'s own pre-filter, not this script's addition).
- **Clamped** (per ``k`` row): ``alpha*k >= |D_union|`` for a query that DID pass the skip filter
  -- Stage 1 could not select a genuine ``alpha*k`` pool and fell back to ``P = D`` entirely (no
  Dinkelbach run for that cell). This is the metric that actually moves with ``k`` in this sweep
  (larger ``k`` -> larger requested pool -> more queries clamped) and is the direct answer to
  "how many queries fall short of the requested pool at this k". Stage 2 still runs on the
  (smaller-than-requested) ``P`` in a clamped cell, so its ILP time is still real and included in
  the mean -- clamping is reported as a separate count, not excluded from the timing average.
"""
import os
from typing import List, NamedTuple, Sequence

from .topn_alpha_k_sweep import run_topn_sweep

TIERS = ("tier_10k", "tier_100k")
TOP_N = 5000
ALPHA = 10.0
KS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
SIGMA = 0.6


class IlpScalingRow(NamedTuple):
    tier: str
    k: int
    pool_requested: int
    n_queries: int
    n_skipped: int
    n_clamped: int
    n_feasible: int
    mean_stage2_ms: float


def build_report(
    tiers: Sequence[str] = TIERS, ks: Sequence[int] = KS,
    top_n: int = TOP_N, alpha: float = ALPHA, sigma: float = SIGMA,
    n_queries: int = 30, verbose: bool = True,
) -> List[IlpScalingRow]:
    out: List[IlpScalingRow] = []
    for tier in tiers:
        rows, _build_times, selection_reports = run_topn_sweep(
            tier, top_ns=(top_n,), ks=ks, alphas=(alpha,), n_queries=n_queries,
            sigma=sigma, verbose=verbose,
        )
        n_selected = selection_reports[top_n]["n_queries_selected"]
        for k in ks:
            cell = [r for r in rows if r.k == k]
            n_this_k = len({r.q_table for r in cell})
            n_skipped = n_selected - n_this_k
            n_clamped = sum(1 for r in cell if r.clamped)
            n_feasible = sum(1 for r in cell if r.feasible)
            mean_ms = (sum(r.stage2_time_s for r in cell) / len(cell) * 1000) if cell else 0.0
            out.append(IlpScalingRow(
                tier=tier, k=k, pool_requested=int(alpha * k),
                n_queries=n_this_k, n_skipped=n_skipped, n_clamped=n_clamped,
                n_feasible=n_feasible, mean_stage2_ms=mean_ms,
            ))
        if verbose:
            for row in out:
                if row.tier != tier:
                    continue
                print(
                    "[{}] k={:3d} pool={:4d} queries={:2d} skipped={:2d} clamped={:2d} "
                    "feasible={:2d}/{:2d} mean_stage2={:.3f}ms".format(
                        row.tier, row.k, row.pool_requested, row.n_queries, row.n_skipped,
                        row.n_clamped, row.n_feasible, row.n_queries, row.mean_stage2_ms,
                    ), flush=True,
                )
    return out


def write_csv(rows: List[IlpScalingRow], name: str = "wdc_ilp_scaling_report",
              output_dir: str = "experiments/results") -> str:
    import csv
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(IlpScalingRow._fields)
        for r in rows:
            w.writerow(r)
    return path


if __name__ == "__main__":
    rows = build_report()
    path = write_csv(rows)
    print("wrote", path)
