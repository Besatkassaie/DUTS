"""Chunked wrapper around ``wdc_extract_via_santos_checkpoint.py`` for tiers too large to
extract in one process.

**Why this exists**: a single unchunked run of ``extract()`` on ``tier_1m`` (1,000,000 tables)
was OOM-killed (``exit -9``) after 148 minutes at 64% progress -- the reused, deliberately
UNMODIFIED ``starmie/extractVectors.py`` accumulates every extracted embedding in memory before
writing its one output pickle, and that doesn't scale to 1M tables (RSS climbed unbounded: 11GB
-> 22.5GB -> 32.7GB and rising when killed; no swap configured on this machine). ``tier_100k``
(100,000 tables) is PROVEN to work in one process (~623s, `RESULTS-wdc.md` §2) -- so this
script never touches ``extractVectors.py`` itself (still completely unmodified, same precedent as
every other benchmark in this repo): it only splits the input CSV directory into
``chunk_size``-table chunks (symlink directories, no copying), calls the existing
``extract()`` once per chunk as a fresh subprocess (so memory never accumulates ACROSS chunks),
and concatenates the resulting pickles (the on-disk format is a plain
``List[(table_name, ndarray)]`` -- see ``dutsx/adapters/unionability.py::load_starmie_vectors`` --
so merging is just list concatenation, no key-collision handling needed since table names are
unique across chunks by construction).

Usage:
    python scripts/wdc_extract_via_santos_checkpoint_chunked.py --tier tier_1m \
        --csv-dir /u6/bkassaie/wdc_data/tiers/tier_1m/csv \
        --out /u6/bkassaie/wdc_data/vectors/tier_1m_roberta.pkl \
        --chunk-size 100000
"""
import argparse
import json
import os
import pickle
import shutil
import time

from wdc_extract_via_santos_checkpoint import extract

CHUNK_ROOT_SUFFIX = "_extract_chunks_tmp"


def _make_chunk_dirs(csv_dir: str, chunk_root: str, chunk_size: int):
    csv_files = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    n = len(csv_files)
    n_chunks = (n + chunk_size - 1) // chunk_size
    chunk_dirs = []
    for i in range(n_chunks):
        chunk_files = csv_files[i * chunk_size:(i + 1) * chunk_size]
        chunk_dir = os.path.join(chunk_root, "chunk_%02d" % i)
        os.makedirs(chunk_dir, exist_ok=True)
        for fname in chunk_files:
            link_path = os.path.join(chunk_dir, fname)
            if not os.path.exists(link_path):
                os.symlink(os.path.join(os.path.abspath(csv_dir), fname), link_path)
        chunk_dirs.append((chunk_dir, len(chunk_files)))
    return chunk_dirs


def run(tier: str, csv_dir: str, out_pkl: str, chunk_size: int = 100_000,
        keep_chunk_pickles: bool = False) -> dict:
    csv_dir = os.path.abspath(csv_dir)
    out_pkl = os.path.abspath(out_pkl)
    chunk_root = csv_dir.rstrip("/") + CHUNK_ROOT_SUFFIX
    os.makedirs(chunk_root, exist_ok=True)

    print("splitting %s into %d-table chunks under %s ..." % (csv_dir, chunk_size, chunk_root),
          flush=True)
    chunk_dirs = _make_chunk_dirs(csv_dir, chunk_root, chunk_size)
    print("%d chunks: %s" % (len(chunk_dirs), [n for _, n in chunk_dirs]), flush=True)

    chunk_pkls = []
    t_total0 = time.time()
    for i, (chunk_dir, n_tables) in enumerate(chunk_dirs):
        chunk_out = os.path.join(chunk_root, "chunk_%02d.pkl" % i)
        chunk_pkls.append(chunk_out)
        if os.path.exists(chunk_out):
            print("[%d/%d] chunk_%02d already extracted, skipping" % (
                i + 1, len(chunk_dirs), i), flush=True)
            continue
        print("[%d/%d] extracting chunk_%02d (%d tables) ..." % (
            i + 1, len(chunk_dirs), i, n_tables), flush=True)
        t0 = time.time()
        stats = extract("%s_chunk_%02d" % (tier, i), chunk_dir, chunk_out)
        print("  done in %.1fs: %s" % (time.time() - t0, stats), flush=True)

    print("merging %d chunk pickles ..." % len(chunk_pkls), flush=True)
    merged = []
    for p in chunk_pkls:
        with open(p, "rb") as f:
            merged.extend(pickle.load(f))
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(merged, f)

    total_elapsed = time.time() - t_total0
    stats = {"tier": tier, "n_tables": len(merged), "n_chunks": len(chunk_dirs),
              "elapsed_s": round(total_elapsed, 1)}
    timing_out = out_pkl.rsplit(".", 1)[0] + ".timing.json"
    with open(timing_out, "w") as f:
        json.dump(stats, f, indent=2)
    print("wrote %s (%d table vectors)" % (out_pkl, len(merged)), flush=True)
    print(json.dumps(stats, indent=2))

    if not keep_chunk_pickles:
        shutil.rmtree(chunk_root)
        print("removed chunk staging dir %s" % chunk_root, flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--keep-chunk-pickles", action="store_true")
    args = ap.parse_args()
    run(args.tier, args.csv_dir, args.out, args.chunk_size, args.keep_chunk_pickles)


if __name__ == "__main__":
    main()
