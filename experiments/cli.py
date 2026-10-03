"""Reproducible CLI entry point for the Phase E experiment harness
(PLAN-integration.md §4 Phase E, deliverable 2).

    PY=/u6/bkassaie/.conda/envs/TableUnionNew/bin/python
    $PY -m experiments.cli all --output-dir experiments/results

    $PY -m experiments.cli alpha-sweep --output-dir experiments/results
    $PY -m experiments.cli retrieval-ablation --output-dir experiments/results

Every sweep is seeded (``--seed``, default 42) and config-driven (adapter
names, k/alpha grids, F*/delta all come from CLI flags with documented
defaults) -- a rerun with the same flags reproduces the same CSV byte-for-byte
modulo wall-clock timing columns (``wall_time_s``/``*_time_s``), which are
measurements of THIS machine, not part of the reproducibility contract.
``--limit-queries`` truncates the santos query list deterministically (first
N by CSV row order) for a fast smoke run; omit it for the real 48-query sweep.
"""
import argparse
import os
import sys
import time

from . import analysis, context as ctx_mod
from .alpha_sweep import run_alpha_sweep
from .fraction_reachability import run_fraction_reachability, summarise, write_rows
from .groundtruth_eval import run_groundtruth_eval
from .lp_precheck import run_lp_precheck_experiment, run_lp_precheck_synthetic_demo
from .retrieval_ablation import run_retrieval_ablation
from .stage1_skip_ablation import run_stage1_skip_ablation
from .schema import ResultRow, rows_to_dataframe, write_csv, write_parquet

DEFAULT_ALPHAS = (1, 2, 3, 5, 10)
DEFAULT_KS = (5, 10, 20)
DEFAULT_F_STAR = 0.3
DEFAULT_DELTA = 0.15
DEFAULT_TOP_N = 100


def _write(rows, name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, name + ".csv")
    write_csv(rows, csv_path)
    print("wrote {} ({} rows)".format(csv_path, len(rows)))
    try:
        write_parquet(rows, os.path.join(output_dir, name + ".parquet"))
    except Exception as exc:  # noqa: BLE001 -- parquet is a bonus, never load-bearing
        print("  (parquet skipped: {})".format(exc))


def _limit(tasks, skipped, limit):
    if limit is None:
        return tasks, skipped
    return tasks[:limit], skipped


def cmd_alpha_sweep(args) -> "list[ResultRow]":
    print("=== alpha sweep: benchmark={} alpha in {} x k in {} ===".format(
        args.benchmark, list(args.alphas), list(args.ks)))
    bench_paths = ctx_mod.paths_for(args.benchmark)
    ctx, adapters = ctx_mod.build_santos_context(
        unionability=args.unionability, paths=bench_paths)
    base_tasks, skipped = ctx_mod.load_base_tasks(
        k=args.ks[0], alpha=args.alphas[0], F_star=args.f_star, delta=args.delta,
        include_query=not args.no_include_query, top_n=args.top_n, paths=bench_paths,
    )
    base_tasks, skipped = _limit(base_tasks, skipped, args.limit_queries)
    print("{} usable queries ({} skipped)".format(len(base_tasks), len(skipped)))

    rows, instances = run_alpha_sweep(
        base_tasks, ctx, adapters, alphas=args.alphas, ks=args.ks, seed=args.seed,
    )
    _write(rows, "{}_alpha_sweep".format(args.benchmark), args.output_dir)

    df = rows_to_dataframe(rows)
    print("\n-- infeasibility by cause, per (k, alpha) --")
    print(analysis.infeasibility_by_cause(df, ["k", "alpha"]).to_string(index=False))
    print("\n-- quality vs cost, per (k, alpha) --")
    print(analysis.quality_vs_cost_by_config(df, ["k", "alpha"]).to_string(index=False))

    violations = analysis.check_efficiency_invariant(df)
    print("\n-- efficiency invariant (n_unionability_computations == n_P == min(alpha*k, n_D)) --")
    print("{} violation(s) out of {} rows that reached Stage 1".format(
        len(violations), len(df[df["infeasibility_cause"] != "insufficient_candidates"])))
    if len(violations):
        print(violations.to_string(index=False))

    return rows, instances, adapters


def cmd_lp_precheck(args, instances, adapters) -> "list[ResultRow]":
    print("\n=== LP pre-check savings, replayed over {} Stage2 instances from the alpha sweep ===".format(
        len(instances)))
    rows = run_lp_precheck_experiment(instances, adapters, seed=args.seed)
    _write(rows, "{}_lp_precheck".format(args.benchmark), args.output_dir)
    df = rows_to_dataframe(rows)
    summary = df.groupby("lp_precheck").agg(
        mean_ilp_time_s=("ilp_time_s", "mean"),
        mean_lp_time_s=("lp_time_s", "mean"),
        n_infeasible=("feasible", lambda s: (~s).sum()),
        n=("feasible", "count"),
    )
    print(summary.to_string())

    print("\n=== LP pre-check savings, SYNTHETIC supplementary demo (real santos pools "
          "are too small -- ~10 candidates -- to exercise this path; see docstring) ===")
    synth_rows = []
    for pool_size in (100, 400, 1000, 3000):
        synth_rows.extend(run_lp_precheck_synthetic_demo(
            seed=args.seed, n_instances=30, pool_size=pool_size,
        ))
    _write(synth_rows, "lp_precheck_synthetic", args.output_dir)
    synth_df = rows_to_dataframe(synth_rows)
    synth_summary = synth_df.groupby(["n_P", "feasible", "lp_precheck"]).agg(
        mean_ilp_time_s=("ilp_time_s", "mean"), n=("ilp_time_s", "count"),
    )
    print(synth_summary.to_string())

    return rows, synth_rows


def cmd_retrieval_ablation(args) -> "list[ResultRow]":
    print("\n=== retrieval ablation ({}): semantic in {{hnsw,exact_scan}} x overlap in {{inverted_index,null}} ===".format(args.benchmark))
    rows = run_retrieval_ablation(
        k=args.k, alpha=args.alpha, F_star=args.f_star, delta=args.delta,
        include_query=not args.no_include_query, top_n=args.top_n, seed=args.seed,
        unionability=args.unionability, task_limit=args.limit_queries,
        paths=ctx_mod.paths_for(args.benchmark),
    )
    _write(rows, "{}_retrieval_ablation".format(args.benchmark), args.output_dir)
    df = rows_to_dataframe(rows)
    print(analysis.infeasibility_by_cause(df, ["config_id"]).to_string(index=False))
    print(df.groupby("config_id")[["n_sem", "n_ovl", "n_pair", "n_D"]].mean().to_string())
    return rows


def cmd_groundtruth_eval(args) -> "list[ResultRow]":
    paths = ctx_mod.paths_for(args.benchmark)
    print("\n=== groundtruth eval: benchmark={} k={} alpha={} ===".format(
        args.benchmark, args.k, args.alpha))
    rows = run_groundtruth_eval(
        paths, k=args.k, alpha=args.alpha, F_star=args.f_star, delta=args.delta,
        include_query=not args.no_include_query, top_n=args.top_n,
        unionability=args.unionability, seed=args.seed, task_limit=args.limit_queries,
    )
    _write(rows, "{}_groundtruth_eval_k{}".format(args.benchmark, args.k), args.output_dir)
    df = rows_to_dataframe(rows)

    n = len(df)
    n_feasible = int(df["feasible"].sum())
    print("\nfeasible: {}/{} queries ({:.0%})".format(n_feasible, n, n_feasible / n if n else 0))
    print(df["infeasibility_cause"].fillna("FEASIBLE").value_counts().to_string())

    print("\n-- per-stage funnel (mean counts) --")
    print(df[["n_sem", "n_ovl", "n_pair", "n_D", "n_P", "n_R"]].mean().to_string())

    print("\n-- precision / recall by stage (mean over queries with a groundtruth entry) --")
    have_gt = df[df["gt_size"].notna()]
    for stage in ("d", "p", "r"):
        pcol, rcol = "precision_" + stage, "recall_" + stage
        print("  {}: precision={:.3f}  recall={:.3f}  (n={})".format(
            stage.upper(), have_gt[pcol].mean(), have_gt[rcol].mean(), len(have_gt)))

    print("\n-- runtime --")
    print("  mean wall_time_s={:.4f}  max={:.4f}".format(
        df["wall_time_s"].mean(), df["wall_time_s"].max()))
    return rows


def cmd_stage1_skip_ablation(args) -> "list[ResultRow]":
    print("\n=== stage1-skip ablation: two_stage vs skip_stage1, benchmark={} k={} alpha={} ===".format(
        args.benchmark, args.k, args.alpha))
    rows = run_stage1_skip_ablation(
        k=args.k, alpha=args.alpha, F_star=args.f_star, delta=args.delta,
        include_query=not args.no_include_query, top_n=args.top_n, seed=args.seed,
        unionability=args.unionability, benchmark=args.benchmark,
        task_limit=args.limit_queries,
    )
    _write(rows, "{}_stage1_skip_ablation".format(args.benchmark), args.output_dir)
    df = rows_to_dataframe(rows)

    df["condition"] = df["config_id"].str.split("__").str[0]
    print("\n-- feasibility by condition --")
    print(df.groupby("condition")["feasible"].agg(["sum", "count"]).to_string())

    print("\n-- runtime by condition (scoring_time_s isolated from ilp_time_s) --")
    print(df.groupby("condition")[["scoring_time_s", "ilp_time_s", "wall_time_s"]]
          .mean().to_string())

    print("\n-- candidate-set size and quality by condition (feasible rows only) --")
    feas = df[df["feasible"]]
    print(feas.groupby("condition")[["n_P", "sum_U", "F_R"]].mean().to_string())

    # per-query paired comparison: does skip_stage1 ever do worse (fewer scored,
    # lower sum_U) than two_stage despite optimizing over a superset of it?
    two = df[df["condition"] == "two_stage"].set_index("q_table")
    skip = df[df["condition"] == "skip_stage1"].set_index("q_table")
    both_feasible = two.index[(two["feasible"]) & (two.index.isin(skip.index)) & (skip.loc[two.index]["feasible"])]
    if len(both_feasible):
        worse = both_feasible[skip.loc[both_feasible]["sum_U"].values < two.loc[both_feasible]["sum_U"].values - 1e-9]
        print("\n-- sum_U monotonicity check (skip_stage1 optimizes over D superset P; "
              "should never score lower) --")
        print("{} of {} paired-feasible queries violate skip_stage1.sum_U >= two_stage.sum_U".format(
            len(worse), len(both_feasible)))
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="experiments.cli", description=__doc__)
    p.add_argument("command", choices=[
        "alpha-sweep", "lp-precheck", "retrieval-ablation", "groundtruth-eval",
        "stage1-skip-ablation", "fraction-reachability", "all"])
    p.add_argument("--benchmark", default="santos",
                    help="which benchmark to run against: santos | santos2 | santos3 | santos4. "
                         "Honoured by every command; result filenames are prefixed with it.")
    p.add_argument("--output-dir", default="experiments/results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--f-star", type=float, default=DEFAULT_F_STAR)
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--no-include-query", action="store_true")
    p.add_argument("--unionability", default="pinned_match",
                    choices=["pinned_match", "starmie_verify", "constant"])
    p.add_argument("--limit-queries", type=int, default=None,
                    help="truncate the santos query list (smoke-test aid; omit for the real sweep)")
    # retrieval-ablation-only knobs (a single fixed operating point)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=2.0)
    return p


def cmd_fraction_reachability(args) -> None:
    """Is F* attainable at all? Brute-forces every k-subset of each query's D
    and reports the attainable range, independent of what the pipeline chose."""
    print("\n=== fraction reachability: benchmark={} k={} F*={} delta={} ===".format(
        args.benchmark, args.k, args.f_star, args.delta))
    rows = run_fraction_reachability(
        benchmark=args.benchmark, k=args.k, alpha=args.alpha, F_star=args.f_star,
        delta=args.delta, include_query=not args.no_include_query, top_n=args.top_n,
        unionability=args.unionability, task_limit=args.limit_queries,
    )
    path = write_rows(rows, "{}_fraction_reachability_k{}".format(args.benchmark, args.k),
                      args.output_dir)
    print("wrote {} ({} rows)".format(path, len(rows)))
    s = summarise(rows)
    for key in ("queries_total", "analysed", "skipped_insufficient", "tau_reachable",
                "fstar_bracketed", "fstar_within_delta", "median_best_gap",
                "mean_best_gap", "median_F_min", "median_F_max", "median_F_Q"):
        v = s.get(key)
        print("  {:<24} {}".format(key, round(v, 4) if isinstance(v, float) else v))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()

    if args.command == "alpha-sweep":
        cmd_alpha_sweep(args)
    elif args.command == "lp-precheck":
        # standalone lp-precheck needs its own alpha-sweep pass to harvest instances from
        _, instances, adapters = cmd_alpha_sweep(args)
        cmd_lp_precheck(args, instances, adapters)
    elif args.command == "retrieval-ablation":
        cmd_retrieval_ablation(args)
    elif args.command == "groundtruth-eval":
        cmd_groundtruth_eval(args)
    elif args.command == "stage1-skip-ablation":
        cmd_stage1_skip_ablation(args)
    elif args.command == "fraction-reachability":
        cmd_fraction_reachability(args)
    elif args.command == "all":
        _, instances, adapters = cmd_alpha_sweep(args)
        cmd_lp_precheck(args, instances, adapters)
        cmd_retrieval_ablation(args)

    print("\ntotal wall time: {:.1f}s".format(time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
