"""Convert a WDC tier's sampled JSON table records to one CSV per table.

This is the single conversion step both consumers read from: `dutsx.adapters.synopsis.CsvSynopsis`
(DUTS-side synopsis/overlap machinery) and the embedding-extraction staging step
(`scripts/wdc_extract_via_santos_checkpoint.py`) both just need a flat directory of CSVs -- no
JSON-native adapter code needed anywhere (see the plan's "Key discovery" section).

Schema, confirmed empirically against real downloaded WDC data (not assumed from corpus docs --
see plan Phase 1's "verify empirically" step, run against archive 00.tar.gz during planning):
  - `relation` is column-major (`relation[col][row]`) regardless of `tableOrientation` -- no
    transposition needed for VERTICAL tables.
  - `headerRowIndex` is always 0 when `hasHeader=True`, and -1 when `hasHeader=False` -- clean,
    not the variable field originally assumed from the corpus's own documentation.
  - Across a 20,000-record sample: zero parse errors, zero ragged relation columns, zero empty
    relations. The tolerant/fallback paths below are a defensive safety net (mirroring
    `CsvSynopsis`'s and santosLarge's precedent), not something expected to fire often.

Usage:
    python scripts/wdc_json_to_csv.py --tier-manifest /u6/bkassaie/wdc_data/tiers/tier_10k/manifest.json \
        --out-dir /u6/bkassaie/wdc_data/tiers/tier_10k/csv
"""
import argparse
import csv
import json
import os
import re
import tarfile
import time
from collections import Counter

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_filename(member_name: str) -> str:
    """member names contain '/' (e.g. '0/<hash>.json') -- flatten to one safe CSV filename."""
    base = _UNSAFE.sub("_", member_name)
    if not base.endswith(".json"):
        base += ".json"
    return base[: -len(".json")] + ".csv"


def dedupe_headers(headers):
    seen = Counter()
    out = []
    for h in headers:
        h = h if h else "col"
        seen[h] += 1
        out.append(h if seen[h] == 1 else "%s.%d" % (h, seen[h] - 1))
    return out


class ParseOutcome:
    OK = "ok"
    RECOVERED = "recovered"
    FAILED = "failed"


def parse_wdc_record(raw: bytes):
    """-> (outcome, headers, rows) where headers/rows are None on FAILED."""
    try:
        rec = json.loads(raw)
    except Exception:
        return ParseOutcome.FAILED, None, None

    relation = rec.get("relation")
    if not relation or not isinstance(relation, list) or len(relation) == 0:
        return ParseOutcome.FAILED, None, None

    row_lens = {len(col) for col in relation}
    outcome = ParseOutcome.OK
    if len(row_lens) != 1:
        # ragged columns -- not observed in the 20k-record schema sample, but handle
        # defensively: truncate every column to the shortest, tracked as a recovery not hidden.
        min_len = min(row_lens)
        if min_len == 0:
            return ParseOutcome.FAILED, None, None
        relation = [col[:min_len] for col in relation]
        outcome = ParseOutcome.RECOVERED

    n_rows_total = len(relation[0])
    if n_rows_total == 0:
        return ParseOutcome.FAILED, None, None

    has_header = bool(rec.get("hasHeader", False))
    header_idx = rec.get("headerRowIndex", -1)

    if has_header and header_idx is not None and 0 <= header_idx < n_rows_total:
        headers = dedupe_headers([str(col[header_idx]) for col in relation])
        data_row_indices = [r for r in range(n_rows_total) if r != header_idx]
    else:
        headers = ["col_%d" % i for i in range(len(relation))]
        data_row_indices = list(range(n_rows_total))

    if not data_row_indices:
        return ParseOutcome.FAILED, None, None

    rows = [[relation[c][r] for c in range(len(relation))] for r in data_row_indices]
    return outcome, headers, rows


def convert_tier(manifest_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # group by archive tar so each is opened (and its fileobj reused) exactly once
    by_archive = {}
    for entry in manifest:
        by_archive.setdefault(entry["archive_tar"], []).append(entry)

    n_ok = 0
    n_recovered = 0
    n_failed = 0
    seen_names = set()
    t0 = time.time()

    for archive_tar, entries in by_archive.items():
        with tarfile.open(archive_tar, mode="r:") as tf:
            for entry in entries:
                tf.fileobj.seek(entry["offset"])
                raw = tf.fileobj.read(entry["size"])
                outcome, headers, rows = parse_wdc_record(raw)

                if outcome == ParseOutcome.FAILED:
                    n_failed += 1
                    continue
                if outcome == ParseOutcome.RECOVERED:
                    n_recovered += 1
                else:
                    n_ok += 1

                fname = safe_filename(entry["member"])
                assert fname not in seen_names, "filename collision: %s" % fname
                seen_names.add(fname)

                with open(os.path.join(out_dir, fname), "w", newline="") as out_f:
                    writer = csv.writer(out_f)
                    writer.writerow(headers)
                    writer.writerows(rows)

    elapsed = time.time() - t0
    stats = {
        "n_total": len(manifest),
        "n_ok": n_ok,
        "n_recovered": n_recovered,
        "n_failed": n_failed,
        "elapsed_s": round(elapsed, 2),
    }
    with open(os.path.join(out_dir, "_conversion_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    stats = convert_tier(args.tier_manifest, args.out_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
