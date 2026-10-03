"""Full per-step timing breakdown at two singleton (tier, k) settings, both pushing the requested
pool close to |D_union| (user-requested, 2026-08-17) -- tier_10k, k=170 (alpha*k=1700) and
tier_100k, k=639 (alpha*k=6390), both at top_n=5000, sigma=0.60, alpha=10 fixed. These are the same
two singleton points already recorded (aggregate-only) in wdc_ilp_scaling_report.csv's k=170/639
rows, added there outside the standard KS=(10..100) grid (see wdc_ilp_scaling_tables.tex note [f]).

wdc_ilp_scaling_report.csv only carries mean Stage 2 time; this module reruns those exact two
settings via topn_alpha_k_sweep.run_topn_sweep and reports the full per-step breakdown (probe /
union_combine / union_lookup / stage1 / scoring / stage2 / end_to_end) as mean/median/min/max, plus
n_feasible/n_clamped/n_skipped context, as two separate LaTeX tables.

**Why N_REPEATS repeated runs, not one (added after the first single-run version of this table
disagreed with wdc_ilp_scaling_report.csv's existing Stage 2 mean for the same cell -- 14.26ms vs.
6.64ms at tier_10k k=170, 30.74ms vs. 41.88ms at tier_100k k=639):** every step's mean/median/min/max
here is heavily right-skewed (e.g. tier_100k Stage 2: median 3.68ms but max 387ms in the first
single run) -- most queries solve fast, a handful are slow (large branch-and-bound trees, or a
table falling in a cold part of the OS page cache), and which specific queries land in that slow
tail is sensitive to machine load at the moment the run happens. With a right-skewed distribution
the MEAN is disproportionately dragged around by whichever slow queries happen to be slowest on a
given run -- a single run's mean is not a reliable estimate on its own. ``run_repeated`` reruns each
setting's PIPELINE STEPS (retrieval/Stage 1/scoring/Stage 2) ``N_REPEATS=5`` times, resampling
system-load jitter independently each time, and POOLS every query's timing across all 5 repeats
into one sample (150 observations for tier_10k, 145 for tier_100k) before computing
mean/median/min/max -- equivalent to a mean-of-means for the mean column (equal sample size per
repeat), but also gives median/min/max far more data, so one unlucky repeat can't dominate them.

**A second, independent bug this surfaced, now worked around (not fixed at the source):** the first
version of this function rebuilt the HNSW index fresh on every repeat and asserted the resulting
query selection/clamping/feasibility counts were identical across repeats -- true for tier_10k (5/5
repeats: 30 queries, 19 clamped, 9 feasible) but FALSE for tier_100k (rep 1: 29 queries/23
clamped/7 feasible; rep 2: 28 queries/22 clamped/7 feasible), tripping the assertion. Root cause:
``dutsx/adapters/semantic.py``'s ``HnswRetriever`` calls ``hnswlib.Index.add_items()`` without
``num_threads=1``, so it defaults to multi-threaded insertion -- documented hnswlib behavior is that
concurrent insertion order (OS-scheduling-dependent) makes the constructed graph itself
non-deterministic across builds even with a fixed ``random_seed``, once there's enough data for
threading to matter (tier_100k's ~518K embedded columns vs. tier_10k's ~52K, which is presumably
why tier_10k happened not to show it in these particular 5 reps -- not a guarantee it never would).
That shifts which borderline neighbors get retrieved, which shifts a borderline query's
``|D_union|``, which flips its skip/clamp status. Querying an ALREADY-BUILT index is deterministic;
only construction is thread-order-sensitive. Fix applied HERE: build the index ONCE per tier
(``shared_ctx`` below) and reuse it across all 5 timing repeats, via ``run_topn_sweep``'s new
optional ``ctx=``/``build_times=`` passthrough -- every repeat now queries the identical graph, so
the "same 30 (or 29) problem instances every repeat" assumption this table's pooling relies on is
actually true, and enforced by the same assertion. This also isolates step-timing jitter (what we
want to measure) from index-construction jitter (what we don't), and is ~5x faster. The underlying
non-determinism in ``HnswRetriever`` construction itself is NOT fixed here -- that's a shared-code
change with build-time/performance tradeoffs across every WDC experiment, a separate decision.
"""
import os
import statistics
from typing import Dict, List, Sequence, Tuple

from .context import build_wdc_context
from .topn_alpha_k_sweep import SweepRow, run_topn_sweep

TOP_N = 5000
SIGMA = 0.6
N_REPEATS = 5
SETTINGS = (
    ("tier_10k", 170, 10.0),
    ("tier_100k", 639, 10.0),
)

STEPS = (
    ("probe_time_s", "Probe (semantic + overlap)"),
    ("union_combine_time_s", "Union combine"),
    ("union_lookup_time_s", "Union lookup (N/n)"),
    ("stage1_time_s", "Stage 1 (Dinkelbach)"),
    ("scoring_time_s", "Scoring (U)"),
    ("stage2_time_s", "Stage 2 (ILP)"),
    ("end_to_end_s", "End-to-end total"),
)


def _stats_ms(rows: Sequence[SweepRow], field: str) -> Dict[str, float]:
    vals = [getattr(r, field) * 1000.0 for r in rows]
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def run_repeated(tier: str, k: int, alpha: float, top_n: int = TOP_N, sigma: float = SIGMA,
                  n_repeats: int = N_REPEATS) -> dict:
    all_rows: List[SweepRow] = []
    per_repeat_mean: Dict[str, List[float]] = {field: [] for field, _label in STEPS}
    n_selected = n_clamped = n_feasible = None

    # Built once and reused across repeats -- see module docstring: hnswlib's default
    # multi-threaded add_items() makes index CONSTRUCTION non-deterministic across builds
    # (confirmed: an earlier version of this function rebuilt the index every repeat and
    # tier_100k's query selection/clamping genuinely differed rep-to-rep). Querying an
    # already-built index is deterministic; only construction is thread-order-sensitive.
    shared_ctx, _shared_build_times = build_wdc_context(tier, sigma=sigma)

    for rep in range(n_repeats):
        rows, _build_times, selection_reports = run_topn_sweep(
            tier, top_ns=(top_n,), ks=(k,), alphas=(alpha,), sigma=sigma, verbose=False,
            ctx=shared_ctx,
        )
        sel = selection_reports[top_n]["n_queries_selected"]
        clamped = sum(1 for r in rows if r.clamped)
        feasible = sum(1 for r in rows if r.feasible)
        print(
            "[{}] rep {}/{}: {} queries, clamped={}, feasible={}".format(
                tier, rep + 1, n_repeats, len(rows), clamped, feasible,
            ), flush=True,
        )
        if n_selected is None:
            n_selected, n_clamped, n_feasible = sel, clamped, feasible
        else:
            assert (sel, clamped, feasible) == (n_selected, n_clamped, n_feasible), (
                "non-deterministic query selection/clamping/feasibility across repeats: "
                "rep 0 had ({}, {}, {}), rep {} had ({}, {}, {})".format(
                    n_selected, n_clamped, n_feasible, rep, sel, clamped, feasible,
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


def _fmt(v: float) -> str:
    return "{:.2f}".format(v)


def _table_tex(result: dict) -> str:
    tier_label = result["tier"].replace("_", "\\_")
    pool = int(result["alpha"] * result["k"])
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{threeparttable}")
    lines.append(
        "\\caption{{Per-step wall-clock time, {} at $k$={}, $\\alpha$={:.0f} "
        "($\\alpha{{\\times}}k$={}), \\texttt{{top\\_n}}={}, $\\sigma$={:.2f}, "
        "pooled over {} repeated runs.}}".format(
            tier_label, result["k"], result["alpha"], pool, result["top_n"], result["sigma"],
            result["n_repeats"],
        )
    )
    lines.append("\\label{{tab:wdc-step-timing-{}}}".format(result["tier"]))
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Step & Mean (ms)\\tnote{a} & Median (ms) & Min (ms) & Max (ms) \\\\")
    lines.append("\\midrule")
    for field, label in STEPS:
        s = result["step_stats"][field]
        note = "\\tnote{b}" if field == "stage1_time_s" or field == "stage2_time_s" else ""
        sep = "\\midrule" if field == "union_lookup_time_s" else None
        lines.append(
            "{}{} & {} & {} & {} & {} \\\\".format(
                label, note, _fmt(s["mean"]), _fmt(s["median"]), _fmt(s["min"]), _fmt(s["max"]),
            )
        )
        if sep:
            lines.append(sep)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\footnotesize")
    lines.append(
        "\\item[a] {} independent repeats of the identical {}-query problem set ({} selected, "
        "{} skipped by the pre-Stage-1 filter $|D_{{union}}|\\geq k$={} -- both counts identical "
        "on every repeat), against one shared, once-built index (only the pipeline steps are "
        "re-timed per repeat, resampling system-load jitter -- see note [b] for why the index "
        "itself is built once, not per repeat). All {}$\\times${}={} per-query timings pooled into "
        "one sample before computing "
        "mean/median/min/max, rather than averaging 5 already-skewed per-run means -- see note "
        "[b]. {} of {} queries were clamped every repeat ($\\alpha{{\\times}}k \\geq "
        "|D_{{union}}|$, Stage 1 skipped, $P=D$, contributing 0ms to Stage 1) and {}/{} reached a "
        "Stage 2-feasible solution ($F_R \\geq \\tau$ achievable in $P$) every repeat.".format(
            result["n_repeats"], result["n_queries"], result["n_selected"], result["n_skipped"],
            result["k"], result["n_queries"], result["n_repeats"], result["n_pooled"],
            result["n_clamped"], result["n_queries"], result["n_feasible"], result["n_queries"],
        )
    )
    stage1_lo, stage1_hi = result["step_repeat_range"]["stage1_time_s"]
    stage2_lo, stage2_hi = result["step_repeat_range"]["stage2_time_s"]
    lines.append(
        "\\item[b] \\textbf{{Why repeated and pooled, not a single run:}} Stage 1/Stage 2 solve "
        "times are heavily right-skewed per query (most queries solve in low single-digit ms; a "
        "few take far longer -- large branch-and-bound trees, or a cold page-cache hit) rather "
        "than normally distributed, so a single run's MEAN is dominated by whichever slow "
        "queries happen to land in that run's tail -- a property of machine load at run time, not "
        "of the algorithm. The {} problem instances themselves are identical every repeat "
        "(enforced by an assertion: n\\_selected/n\\_clamped/n\\_feasible match exactly across all "
        "{} repeats) -- only the WALL-CLOCK COST of running the same instances is re-measured each "
        "repeat, against one index built once (querying an already-built HNSW index is "
        "deterministic; only its construction is thread-order-sensitive and was confirmed, during "
        "development of this table, to occasionally change tier\\_100k's retrieved/clamped query "
        "set across independent rebuilds -- worked around here by building once and reusing). "
        "Direct evidence of the swing pure timing jitter alone still causes even with the "
        "instances held fixed: Stage 1's own per-repeat mean ranged from {}ms to {}ms across the "
        "{} repeats, and Stage 2's from {}ms to {}ms -- either extreme alone, reported as \"the\" "
        "mean, would misrepresent typical cost by a factor of 2--3$\\times$. Pooling all {} "
        "queries $\\times$ {} repeats into one sample before taking mean/median/min/max (rather "
        "than reporting one arbitrary repeat, or averaging the 5 already-skewed per-run means) is "
        "what the Mean/Median columns above report; Min/Max are the single fastest/slowest "
        "individual query observed across the full pooled sample, not per-repeat "
        "extremes.".format(
            result["n_queries"], result["n_repeats"], _fmt(stage1_lo), _fmt(stage1_hi),
            result["n_repeats"], _fmt(stage2_lo), _fmt(stage2_hi), result["n_queries"],
            result["n_repeats"],
        )
    )
    lines.append(
        "\\item[c] Retrieval subtotal (probe + union combine + union lookup) = "
        "{} ms mean. Stage 1 + scoring + Stage 2 + retrieval subtotal need not sum exactly to "
        "the end-to-end total due to independent \\texttt{{time.perf\\_counter()}} calls "
        "per phase.".format(_fmt(sum(
            result["step_stats"][f]["mean"]
            for f in ("probe_time_s", "union_combine_time_s", "union_lookup_time_s")
        )))
    )
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_and_write(output_dir: str = "experiments/results",
                     name: str = "wdc_step_timing_deepdive") -> Tuple[str, List[dict]]:
    results = [run_repeated(tier, k, alpha) for tier, k, alpha in SETTINGS]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name + ".tex")
    with open(path, "w") as f:
        f.write("% \\usepackage{booktabs,threeparttable} required\n\n")
        f.write("\n\n".join(_table_tex(r) for r in results))
        f.write("\n")
    return path, results


if __name__ == "__main__":
    path, results = build_and_write()
    for r in results:
        print(
            "[{}] k={} pool={} queries={} (x{} repeats, {} pooled) clamped={} feasible={}".format(
                r["tier"], r["k"], int(r["alpha"] * r["k"]), r["n_queries"], r["n_repeats"],
                r["n_pooled"], r["n_clamped"], r["n_feasible"],
            )
        )
    print("wrote", path)
