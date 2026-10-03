"""Full per-step timing breakdown across a k-sweep at alpha=10 fixed, tier_100k only, top_n=5000,
sigma=0.60 (user-requested, 2026-08-18). Extends wdc_ilp_scaling_report.csv's tier_100k k-sweep
(k in {10,20,...,100,639}, same grid) -- that CSV only carries n_feasible/n_clamped and a single-run
mean_stage2_ms; this module reruns the identical grid via topn_alpha_k_sweep.run_topn_sweep with the
same 5-repeats-pooled-per-k methodology already used and explained in step_timing_deepdive.py (see
that module's docstring for the full right-skew and HNSW-non-determinism rationale -- same reasoning
applies here unchanged, just swept over k instead of a single point), and reports the full per-step
breakdown (probe / union_combine / union_lookup / stage1 / scoring / stage2 / end_to_end) as
mean/median/min/max for every k, plus n_feasible/n_clamped/n_skipped per k.

**Why each k is run as its own singleton sweep, not one run_topn_sweep(ks=ALL_KS) call:**
run_topn_sweep's query-eligibility filter is ``len(D) < max(ks)`` -- if all 11 k values were passed
in one call, the filter would use max_k=639 for EVERY k, silently skipping small-k queries that
wdc_ilp_scaling_report.csv's original per-k runs (each with its own max_k=k) did not skip, making
this table's k=10..100 rows disagree with the existing CSV for no algorithmic reason. Each k is
therefore swept with its own ``run_topn_sweep(ks=(k,), ...)`` call, preserving that CSV's per-k
selection semantics exactly.

**What IS shared across all 11 k's and all 5 repeats:** the HNSW+overlap context (``build_wdc_context``)
is built exactly ONCE for tier_100k up front and passed via ``ctx=`` into every one of the 11*5=55
``run_topn_sweep`` calls -- ctx depends only on (tier, sigma), never on k, so rebuilding it per-k
would be 11x wasted work and would also reintroduce the HNSW add_items() construction
non-determinism documented in step_timing_deepdive.py (querying an already-built index is
deterministic; only construction is thread-order-sensitive).
"""
import os
import statistics
from typing import Dict, List, Sequence, Tuple

from .context import build_wdc_context
from .step_timing_deepdive import STEPS, _fmt, _stats_ms
from .topn_alpha_k_sweep import SweepRow, run_topn_sweep

TIER = "tier_100k"
TOP_N = 5000
SIGMA = 0.6
ALPHA = 10.0
N_REPEATS = 5
KS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 639)


def run_repeated_for_k(ctx, tier: str, k: int, alpha: float, top_n: int, sigma: float,
                        n_repeats: int) -> dict:
    all_rows: List[SweepRow] = []
    per_repeat_mean: Dict[str, List[float]] = {field: [] for field, _label in STEPS}
    n_selected = n_clamped = n_feasible = None

    for rep in range(n_repeats):
        rows, _build_times, selection_reports = run_topn_sweep(
            tier, top_ns=(top_n,), ks=(k,), alphas=(alpha,), sigma=sigma, verbose=False,
            ctx=ctx,
        )
        sel = selection_reports[top_n]["n_queries_selected"]
        clamped = sum(1 for r in rows if r.clamped)
        feasible = sum(1 for r in rows if r.feasible)
        print(
            "[{} k={}] rep {}/{}: {} queries, clamped={}, feasible={}".format(
                tier, k, rep + 1, n_repeats, len(rows), clamped, feasible,
            ), flush=True,
        )
        if n_selected is None:
            n_selected, n_clamped, n_feasible = sel, clamped, feasible
        else:
            assert (sel, clamped, feasible) == (n_selected, n_clamped, n_feasible), (
                "non-deterministic query selection/clamping/feasibility across repeats at k={}: "
                "rep 0 had ({}, {}, {}), rep {} had ({}, {}, {})".format(
                    k, n_selected, n_clamped, n_feasible, rep, sel, clamped, feasible,
                )
            )
        all_rows.extend(rows)
        for field, _label in STEPS:
            per_repeat_mean[field].append(
                statistics.mean(getattr(r, field) * 1000.0 for r in rows)
            )

    n_queries = len(all_rows) // n_repeats
    n_skipped = n_selected - n_queries
    step_stats = {field: _stats_ms(all_rows, field) for field, _label in STEPS}
    step_repeat_range = {
        field: (min(vals), max(vals)) for field, vals in per_repeat_mean.items()
    }
    return dict(
        tier=tier, k=k, alpha=alpha, top_n=top_n, sigma=sigma, n_repeats=n_repeats,
        n_selected=n_selected, n_queries=n_queries, n_skipped=n_skipped,
        n_clamped=n_clamped, n_feasible=n_feasible, n_pooled=len(all_rows),
        step_stats=step_stats, step_repeat_range=step_repeat_range,
    )


def run_k_sweep(tier: str = TIER, ks: Sequence[int] = KS, alpha: float = ALPHA,
                 top_n: int = TOP_N, sigma: float = SIGMA,
                 n_repeats: int = N_REPEATS) -> List[dict]:
    shared_ctx, build_times = build_wdc_context(tier, sigma=sigma)
    print(
        "[{}] index build: {:.2f}s over {} tables, sigma={}".format(
            tier, build_times.total_index_s, build_times.n_tables, sigma,
        ), flush=True,
    )
    results = []
    for k in ks:
        results.append(run_repeated_for_k(shared_ctx, tier, k, alpha, top_n, sigma, n_repeats))
    return results


def _table_tex(results: List[dict]) -> str:
    tier_label = results[0]["tier"].replace("_", "\\_")
    top_n = results[0]["top_n"]
    sigma = results[0]["sigma"]
    alpha = results[0]["alpha"]
    n_repeats = results[0]["n_repeats"]

    lines = []
    lines.append("\\begin{longtable}{llrrrr}")
    lines.append(
        "\\caption{{Per-step wall-clock time vs. $k$, {} at $\\alpha$={:.0f} fixed, "
        "\\texttt{{top\\_n}}={}, $\\sigma$={:.2f}, pooled over {} repeated runs per $k$.}}"
        "\\label{{tab:wdc-k-sweep-timing-{}}} \\\\".format(
            tier_label, alpha, top_n, sigma, n_repeats, results[0]["tier"],
        )
    )
    lines.append("\\toprule")
    lines.append("$k$ & Step & Mean (ms) & Median (ms) & Min (ms) & Max (ms) \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append(
        "\\multicolumn{6}{l}{\\textit{(continued)}} \\\\"
    )
    lines.append("\\toprule")
    lines.append("$k$ & Step & Mean (ms) & Median (ms) & Min (ms) & Max (ms) \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\midrule")
    lines.append(
        "\\multicolumn{6}{r}{\\textit{continued on next page}} \\\\"
    )
    lines.append("\\endfoot")
    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")

    for r in results:
        pool = int(r["alpha"] * r["k"])
        lines.append(
            "\\multicolumn{{6}}{{l}}{{\\textbf{{$k$={}}} (pool={}, {}/{} queries, {} skipped, "
            "{} clamped, {} feasible)\\tnote{{a}}}} \\\\".format(
                r["k"], pool, r["n_queries"], r["n_selected"], r["n_skipped"],
                r["n_clamped"], r["n_feasible"],
            )
        )
        for field, label in STEPS:
            s = r["step_stats"][field]
            note = "\\tnote{b}" if field in ("stage1_time_s", "stage2_time_s") else ""
            lines.append(
                " & {}{} & {} & {} & {} & {} \\\\".format(
                    label, note, _fmt(s["mean"]), _fmt(s["median"]), _fmt(s["min"]), _fmt(s["max"]),
                )
            )
        lines.append("\\midrule")

    lines.append("\\end{longtable}")

    notes = []
    notes.append("\\begin{minipage}{\\linewidth}")
    notes.append("\\footnotesize")
    notes.append("\\begin{tablenotes}")
    notes.append(
        "\\item[a] Per $k$: {} independent repeats of the SAME query set at that $k$ "
        "(query eligibility/skip is $|D_{{union}}|\\geq k$, so which queries are eligible "
        "differs by $k$ -- $k$=100 skips 1 more query than $k$=10..90; $k$=639 skips several "
        "more, see the skipped counts above). \\emph{{n\\_selected}}, \\emph{{n\\_skipped}}, "
        "\\emph{{n\\_clamped}}, and \\emph{{n\\_feasible}} are identical across all {} repeats "
        "at a given $k$ (enforced by assertion); only step wall-clock cost is re-measured each "
        "repeat, against ONE shared context (semantic HNSW index + overlap posting-list index) "
        "built once for {} and reused across every $k$ and every repeat.".format(
            n_repeats, n_repeats, tier_label,
        )
    )
    notes.append(
        "\\item[b] Stage 1/Stage 2 solve times are heavily right-skewed per query (most queries "
        "solve in low single-digit ms; a few take far longer), so Mean/Median/Min/Max are all "
        "computed by pooling every query's timing across all 5 repeats into one sample per $k$ "
        "-- the same repeated-and-pooled methodology used in and explained fully by "
        "\\texttt{wdc\\_step\\_timing\\_deepdive.tex} (right-skew instability + the HNSW "
        "add\\_items() construction non-determinism found there); not repeated in full here to "
        "avoid duplicating that explanation across 11 $k$ values."
    )
    notes.append("\\end{tablenotes}")
    notes.append("\\end{minipage}")
    return "\n".join(lines + notes)


def build_and_write(output_dir: str = "experiments/results",
                     name: str = "wdc_k_sweep_timing_deepdive") -> Tuple[str, List[dict]]:
    results = run_k_sweep()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name + ".tex")
    with open(path, "w") as f:
        f.write("% \\usepackage{booktabs,longtable,threeparttable} required "
                 "(threeparttable's tnote/minipage style reused inside a longtable here since "
                 "threeparttable itself does not support longtable)\n\n")
        f.write(_table_tex(results))
        f.write("\n")
    return path, results


if __name__ == "__main__":
    path, results = build_and_write()
    for r in results:
        print(
            "[{}] k={} pool={} queries={} (x{} repeats, {} pooled) skipped={} clamped={} "
            "feasible={}".format(
                r["tier"], r["k"], int(r["alpha"] * r["k"]), r["n_queries"], r["n_repeats"],
                r["n_pooled"], r["n_skipped"], r["n_clamped"], r["n_feasible"],
            )
        )
    print("wrote", path)
