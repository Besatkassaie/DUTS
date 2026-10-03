"""CLI driver for ``topn_alpha_k_sweep.run_topn_sweep`` -- one tier per invocation so the two tiers
can run as separate background processes.

Usage:
    python -m experiments.wdc.run_sweep --tier tier_10k
    python -m experiments.wdc.run_sweep --tier tier_100k
"""
import argparse
import time

from .topn_alpha_k_sweep import (
    DEFAULT_ALPHAS, DEFAULT_KS, DEFAULT_SIGMA, DEFAULT_TOP_NS,
    run_topn_sweep, write_rows, write_selection_reports,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--out-name", default=None, help="defaults to 'wdc_sweep_<tier>'")
    args = ap.parse_args()

    out_name = args.out_name or ("wdc_sweep_%s" % args.tier)
    t0 = time.time()
    rows, build_times, selection_reports = run_topn_sweep(
        args.tier, top_ns=DEFAULT_TOP_NS, ks=DEFAULT_KS, alphas=DEFAULT_ALPHAS, sigma=args.sigma,
    )
    csv_path = write_rows(rows, out_name)
    report_path = write_selection_reports(selection_reports, out_name)
    elapsed = time.time() - t0
    print("wrote {} rows to {}".format(len(rows), csv_path), flush=True)
    print("wrote selection reports to {}".format(report_path), flush=True)
    print("build_times: {}".format(build_times), flush=True)
    print("total elapsed: {:.1f}s".format(elapsed), flush=True)


if __name__ == "__main__":
    main()
