"""Correctness check for a WDC ``MetadataStore`` pickle against ``CsvSynopsis``, the repo's
existing independent oracle (same role ``tests/test_dutsx_synopsis.py`` gives it for the santos
pickles -- ``CsvSynopsis`` recomputes everything from raw CSVs with pandas and never imports
``TableMetadata``/``MetadataStore``, so agreement between the two is a real correctness signal, not
a tautology).

Two checks, matching that test file's pattern:

1. ``n_rows`` -- exhaustive for ``tier_10k`` (10,000 tables, cheap), a large random sample for
   ``tier_100k`` (5,000 tables -- an exhaustive pass would mean 100,000 fresh pandas reads with no
   caching payoff, on the same order as the tier's own ~40s CsvSynopsis-driven inverted-index build;
   a 5,000-table sample is still a strong statistical check without paying that twice).
2. ``distribution`` (``N_i`` inputs) -- a random sample of (table, categorical attr) pairs per tier,
   comparing the full ``{value: count}`` histogram, not just a summary statistic.

Usage:
    python scripts/wdc_verify_metadata_store.py --tier tier_10k
    python scripts/wdc_verify_metadata_store.py --tier tier_100k
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dutsx.adapters.synopsis import CsvSynopsis, MetadataStoreSynopsis  # noqa: E402

WDC_ROOT = "/u6/bkassaie/wdc_data"
THETA_CAT = 50


def verify(tier: str, n_rows_sample: int, n_dist_pairs: int, seed: int = 0) -> bool:
    csv_dir = os.path.join(WDC_ROOT, "tiers", tier, "csv")
    pkl_path = os.path.join(WDC_ROOT, "indexes", "%s_metadata.pkl" % tier)

    cs = CsvSynopsis([csv_dir], theta_cat=THETA_CAT)
    ms = MetadataStoreSynopsis(pkl_path, theta_cat=THETA_CAT)

    tables = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    rng = random.Random(seed)

    ok = True

    # --- n_rows -----------------------------------------------------------
    row_sample = tables if len(tables) <= n_rows_sample else rng.sample(tables, n_rows_sample)
    n_rows_mismatches = []
    for t in row_sample:
        a, b = cs.n_rows(t), ms.n_rows(t)
        if a != b:
            n_rows_mismatches.append((t, a, b))
    print("[%s] n_rows: checked %d/%d tables, %d mismatches%s" % (
        tier, len(row_sample), len(tables), len(n_rows_mismatches),
        " (exhaustive)" if row_sample is tables else "",
    ))
    for t, a, b in n_rows_mismatches[:10]:
        print("    MISMATCH n_rows(%r): CsvSynopsis=%d MetadataStoreSynopsis=%d" % (t, a, b))
    ok = ok and not n_rows_mismatches

    # --- distribution -------------------------------------------------------
    table_sample = rng.sample(tables, min(n_dist_pairs, len(tables)))
    dist_checked, dist_mismatches = 0, []
    for t in table_sample:
        cat_attrs = cs.categorical_attrs(t)
        if not cat_attrs:
            continue
        attr = rng.choice(cat_attrs)
        d_cs = cs.distribution(t, attr)
        d_ms = ms.distribution(t, attr)
        dist_checked += 1
        if d_cs != d_ms:
            dist_mismatches.append((t, attr, d_cs, d_ms))
    print("[%s] distribution: checked %d (table, attr) pairs, %d mismatches" % (
        tier, dist_checked, len(dist_mismatches),
    ))
    for t, attr, d_cs, d_ms in dist_mismatches[:5]:
        print("    MISMATCH distribution(%r, %r):" % (t, attr))
        print("      CsvSynopsis          =", d_cs)
        print("      MetadataStoreSynopsis=", d_ms)
    ok = ok and not dist_mismatches

    print("[%s] %s" % (tier, "PASS" if ok else "FAIL"))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--n-rows-sample", type=int, default=5000)
    ap.add_argument("--n-dist-pairs", type=int, default=300)
    args = ap.parse_args()
    ok = verify(args.tier, args.n_rows_sample, args.n_dist_pairs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
