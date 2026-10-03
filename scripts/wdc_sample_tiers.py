"""Sample nested WDC table-count tiers from indexed archives.

Builds `tier_10k` (10,000 members), `tier_100k` (tier_10k plus 90,000 more), `tier_1m`
(tier_100k plus 900,000 more) -- each tier a strict superset of the smaller ones, so tier-to-tier
deltas are directly attributable to added tables and smaller tiers' downstream artifacts (CSVs,
embeddings) are reusable inside larger tiers rather than rebuilt from scratch.

Each manifest entry embeds (archive, member_name, offset, size) directly -- avoids reloading a
full ~114MB per-archive index file just to resolve a handful of members later.

Usage:
    python scripts/wdc_sample_tiers.py \
        --indexes /u6/bkassaie/wdc_data/raw/00.tar.index.json /u6/bkassaie/wdc_data/raw/01.tar.index.json \
        --archive-tars /u6/bkassaie/wdc_data/raw/00.tar /u6/bkassaie/wdc_data/raw/01.tar \
        --out-dir /u6/bkassaie/wdc_data/tiers --seed 42
"""
import argparse
import json
import os
import random

TIER_SIZES = [("tier_10k", 10_000), ("tier_100k", 100_000), ("tier_1m", 1_000_000)]


def load_pool(index_paths, archive_tar_paths):
    """-> list of (archive_tar_path, member_name, offset, size), stable order."""
    assert len(index_paths) == len(archive_tar_paths)
    pool = []
    for index_path, tar_path in zip(index_paths, archive_tar_paths):
        with open(index_path) as f:
            index = json.load(f)
        for name, info in index.items():
            pool.append((tar_path, name, info["offset"], info["size"]))
    return pool


def sample_nested_tiers(pool, seed=42):
    total_needed = TIER_SIZES[-1][1]
    if len(pool) < total_needed:
        raise ValueError(
            "pool has %d members, need >= %d for the largest tier" % (len(pool), total_needed)
        )
    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)  # a single shuffled order -- prefixes of it ARE the nested tiers

    tiers = {}
    for tier_name, size in TIER_SIZES:
        member_indices = order[:size]
        tiers[tier_name] = [pool[i] for i in member_indices]
    return tiers


def write_tier_manifest(tier_name, entries, out_dir):
    tier_dir = os.path.join(out_dir, tier_name)
    os.makedirs(tier_dir, exist_ok=True)
    manifest_path = os.path.join(tier_dir, "manifest.json")
    manifest = [
        {"archive_tar": archive_tar, "member": member, "offset": offset, "size": size}
        for archive_tar, member, offset, size in entries
    ]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print("wrote %s (%d members) -> %s" % (tier_name, len(manifest), manifest_path))
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indexes", nargs="+", required=True)
    ap.add_argument("--archive-tars", nargs="+", required=True)
    ap.add_argument("--out-dir", default="/u6/bkassaie/wdc_data/tiers")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = load_pool(args.indexes, args.archive_tars)
    print("pool size: %d members across %d archives" % (len(pool), len(args.indexes)))

    tiers = sample_nested_tiers(pool, seed=args.seed)
    for tier_name, _ in TIER_SIZES:
        write_tier_manifest(tier_name, tiers[tier_name], args.out_dir)

    # sanity: confirm nesting (tier_10k members subset of tier_100k, etc.)
    keysets = {name: {(a, n) for a, n, o, s in tiers[name]} for name, _ in TIER_SIZES}
    assert keysets["tier_10k"] <= keysets["tier_100k"] <= keysets["tier_1m"], "nesting violated!"
    print("nesting verified: tier_10k subset tier_100k subset tier_1m")


if __name__ == "__main__":
    main()
