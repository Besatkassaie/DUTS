"""Build a prebuilt ``MetadataStore`` synopsis pickle for a WDC tier.

Why this exists: ``dutsx/adapters/synopsis.py::CsvSynopsis`` (WDC's current, and only, synopsis
source -- see that module's docstring) never persists a computed histogram to disk. Its
``distribution()`` recomputes ``pandas.Series.value_counts()`` from scratch on EVERY call, even for
a ``(table, attr)`` pair already queried earlier in the same process -- confirmed by inspection to be
the dominant cost in the WDC scalability sweep (72-90% of end-to-end time, ``RESULTS-wdc-sweep.md``
§6). ``dutsx/adapters/synopsis.py::MetadataStoreSynopsis`` already exists and gives O(1) dict lookups
against a prebuilt pickle -- it's what the santos-family benchmarks use -- but no such pickle has
ever been built for WDC (grepped ``experiments/wdc/``, ``scripts/``, ``dutsx/`` for any call to
``MetadataStore.build_metadata_from_csv``/``.save()``: zero hits). This script closes that gap.

Reuses ``/u6/bkassaie/starmie_fair/TableMetadata.py::MetadataStore`` COMPLETELY UNMODIFIED --
``build_metadata_from_csv(folder_path)`` computes a full ``Counter(col_values)`` histogram for
EVERY column of every CSV in the folder (its own docstring's claim, confirmed by reading
``_process_csv_file``: no domain-size filter at build time despite the "categorical" naming --
filtering happens later, at ``MetadataStoreSynopsis.categorical_attrs()``'s ``theta_cat`` check),
then ``.save(filepath)`` pickles the whole store. Semantics match ``CsvSynopsis`` exactly: both use
raw string values with empty cells as a real ``""`` bucket (``Counter`` over `csv.reader`'s raw
strings here vs. ``pandas`` ``value_counts(dropna=False)`` there), so ``N_i``/``n_i`` should agree --
verified separately, not just assumed (see ``scripts/wdc_verify_metadata_store.py``).

Usage:
    python scripts/wdc_build_metadata_store.py --tier tier_10k \
        --csv-dir /u6/bkassaie/wdc_data/tiers/tier_10k/csv \
        --out /u6/bkassaie/wdc_data/indexes/tier_10k_metadata.pkl
"""
import argparse
import io
import json
import os
import sys
import time

STARMIE_FAIR_ROOT = "/u6/bkassaie/starmie_fair"


def _process_csv_file_tolerant(store, filepath: str, table_name: str) -> bool:
    """Reimplements ``MetadataStore._process_csv_file`` (TableMetadata.py:235-267) line-for-line
    for the common path, adding one guard: strip embedded NUL bytes before handing the text to
    ``csv.reader``, which raises ``_csv.Error: line contains NUL`` and cannot be told to tolerate
    them (unlike pandas' C parser, which ``CsvSynopsis`` already has a fallback for -- this is the
    same class of rare corrupt-row issue, a different parser). Found on 2/100,000 tier_100k files
    (0 on tier_10k): genuine embedded ``\\x00`` bytes inside quoted field values (e.g.
    ``"v\\x00deosxvip.com"``), not an encoding artifact. Returns True if this file needed the
    NUL-strip fallback (tracked by the caller, mirroring ``CsvSynopsis.parse_recovered``/
    ``_conversion_stats.json``'s ``n_recovered`` precedent)."""
    import csv

    from TableMetadata import TableMetadata  # noqa: E402  (path set by caller)
    from collections import Counter

    with open(filepath, "rb") as f:
        raw = f.read()
    recovered = b"\x00" in raw
    text = raw.decode("utf-8", errors="replace").replace("\x00", "")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return recovered

    metadata = TableMetadata(table_name)
    headers = rows[0]
    data_rows = rows[1:]
    metadata.num_records = len(data_rows)
    metadata.num_columns = len(headers)
    metadata.column_names = headers
    metadata._column_index = {name: i for i, name in enumerate(headers)}

    for col_idx, col_name in enumerate(headers):
        col_values = [row[col_idx] if col_idx < len(row) else "" for row in data_rows]
        if col_values:
            metadata.add_categorical_column(col_name, dict(Counter(col_values)))

    store._metadata[table_name] = metadata
    return recovered


def build(tier: str, csv_dir: str, out_pkl: str, timing_out: str = None) -> dict:
    if STARMIE_FAIR_ROOT not in sys.path:
        sys.path.insert(0, STARMIE_FAIR_ROOT)
    from TableMetadata import MetadataStore  # noqa: E402  (path set above)

    csv_dir = os.path.abspath(csv_dir)
    out_pkl = os.path.abspath(out_pkl)
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)

    files = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    n_files = len(files)

    t0 = time.time()
    store = MetadataStore()
    n_recovered = 0
    n_failed = 0
    failed_files = []
    for fname in files:
        filepath = os.path.join(csv_dir, fname)
        try:
            store._process_csv_file(filepath, fname)  # noqa: SLF001 -- same call
                                                        # build_metadata_from_csv makes per-file,
                                                        # just looped here so one bad file doesn't
                                                        # abort the whole batch.
        except Exception:
            try:
                if _process_csv_file_tolerant(store, filepath, fname):
                    n_recovered += 1
            except Exception as e:
                # Genuinely unparseable even by the tolerant fallback (e.g. a field
                # exceeding csv's field-size limit, found on tier_1m -- neither
                # tier_10k nor tier_100k ever hit this) -- skip and count it rather
                # than crashing the whole batch, same tolerance already applied to
                # NUL-byte rows above.
                n_failed += 1
                failed_files.append((fname, str(e)))
    store._update_global_stats()  # noqa: SLF001 -- normally called at the end of
                                   # build_metadata_from_csv; replicated since that method
                                   # was bypassed above for per-file error recovery.
    build_elapsed = time.time() - t0

    t0 = time.time()
    store.save(out_pkl)
    save_elapsed = time.time() - t0

    stats = {
        "tier": tier, "n_files": n_files, "n_tables_stored": len(store._metadata),
        "n_recovered": n_recovered, "n_failed": n_failed, "failed_files": failed_files,
        "build_elapsed_s": round(build_elapsed, 2), "save_elapsed_s": round(save_elapsed, 2),
        "total_elapsed_s": round(build_elapsed + save_elapsed, 2),
        "pkl_size_bytes": os.path.getsize(out_pkl),
    }
    timing_out = timing_out or (out_pkl.rsplit(".", 1)[0] + ".timing.json")
    with open(timing_out, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(args.tier, args.csv_dir, args.out)


if __name__ == "__main__":
    main()
