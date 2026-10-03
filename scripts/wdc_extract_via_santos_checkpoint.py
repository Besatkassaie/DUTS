"""Extract WDC table embeddings by reusing the existing trained santos checkpoint.

Reuses `/u6/bkassaie/starmie/extractVectors.py` COMPLETELY UNMODIFIED. That script hardcodes
`hp.benchmark="santos"` right after argument parsing (extractVectors.py:93-96), regardless of the
--benchmark flag, and always loads `results/santos/model_drop_col_tfidf_entity_column_0.pt` -- so
the only way to embed a different dataset with it is to physically stage that dataset's tables into
`data/santos/{query,datalake}` and run it there, exactly the workflow already used (manually) for
every non-santos dataset in that repo (santos2/3/4, santos_fair) -- `data/santos_org/` is the
archive of prior staged variants, and `data/santos/` itself is a transient staging slot, absent when
nothing is being staged.

This gets real *trained* embeddings (not vanilla-untrained-RoBERTa noise) at zero training cost --
santos-trained, applied out-of-domain to WDC's web tables, an honest cross-domain caveat rather than
an undertrained-model one (see the plan's "Resolved design decisions" #3).

Safety: only `data/santos/` is ever touched. If it already exists when this script starts (e.g. a
manual run is mid-flight), it's renamed aside and restored afterward -- `data/santos_org/` and
`results/santos/` are never written to at any point.

Usage:
    python scripts/wdc_extract_via_santos_checkpoint.py --tier tier_10k \
        --csv-dir /u6/bkassaie/wdc_data/tiers/tier_10k/csv \
        --out /u6/bkassaie/wdc_data/vectors/tier_10k_roberta.pkl
"""
import argparse
import json
import os
import shutil
import subprocess
import time

STARMIE_ROOT = "/u6/bkassaie/starmie"
SANTOS_DIR = os.path.join(STARMIE_ROOT, "data", "santos")
CONDA_ENV_PYTHON = "/u6/bkassaie/.conda/envs/tableunion/bin/python"
OUTPUT_PKL_NAME = "cl_datalake_drop_col_tfidf_entity_column_0.pkl"
QUERY_PLACEHOLDER_COUNT = 3  # just enough that get_df() never sees an empty glob


def backup_existing_dir(target_dir: str):
    """If ``target_dir`` already exists, rename it aside. Returns the backup
    path (or None if nothing needed backing up). ``target_dir`` is a
    parameter (not hardcoded to SANTOS_DIR) so this is unit-testable against
    a tmp_path without ever touching the real staging directory."""
    if not os.path.exists(target_dir):
        return None
    backup_path = target_dir + ".bak_%d" % int(time.time())
    os.rename(target_dir, backup_path)
    print("backed up existing %s -> %s" % (target_dir, backup_path))
    return backup_path


def restore_or_absent(target_dir: str, backup_path):
    """Undo ``backup_existing_dir()``: restore the backup, or (if there was
    none) just make sure the staged dir is gone -- matches whatever state
    existed before staging started."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    if backup_path is not None:
        os.rename(backup_path, target_dir)
        print("restored %s -> %s" % (backup_path, target_dir))
    else:
        print("no prior %s existed -- left absent, as found" % target_dir)


def _stage(csv_dir: str, target_dir: str = SANTOS_DIR):
    os.makedirs(target_dir)
    datalake_dir = os.path.join(target_dir, "datalake")
    query_dir = os.path.join(target_dir, "query")
    os.makedirs(datalake_dir)
    os.makedirs(query_dir)
    os.makedirs(os.path.join(target_dir, "vectors"))  # extractVectors.py writes here but never
                                                        # creates it -- must pre-exist (found by
                                                        # running this against real data).

    csv_files = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    if not csv_files:
        raise RuntimeError("no CSV files found in %s" % csv_dir)

    for fname in csv_files:
        os.symlink(os.path.join(csv_dir, fname), os.path.join(datalake_dir, fname))

    # placeholder only -- this side's output pickle is discarded (see module
    # docstring); it exists purely so extractVectors.py's get_df() doesn't
    # see an empty directory glob.
    for fname in csv_files[:QUERY_PLACEHOLDER_COUNT]:
        os.symlink(os.path.join(csv_dir, fname), os.path.join(query_dir, fname))

    return len(csv_files)


def extract(tier: str, csv_dir: str, out_pkl: str, timing_out: str = None) -> dict:
    csv_dir = os.path.abspath(csv_dir)
    out_pkl = os.path.abspath(out_pkl)
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)

    backup_path = backup_existing_dir(SANTOS_DIR)
    try:
        n_staged = _stage(csv_dir)
        print("staged %d tables into %s" % (n_staged, SANTOS_DIR))

        t0 = time.time()
        result = subprocess.run(
            [CONDA_ENV_PYTHON, "extractVectors.py"],
            cwd=STARMIE_ROOT, capture_output=True, text=True,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(result.stdout[-4000:])
            print(result.stderr[-4000:])
            raise RuntimeError("extractVectors.py failed (exit %d)" % result.returncode)

        produced = os.path.join(SANTOS_DIR, "vectors", OUTPUT_PKL_NAME)
        if not os.path.exists(produced):
            raise RuntimeError("expected output not found: %s" % produced)
        shutil.move(produced, out_pkl)

        stats = {"tier": tier, "n_tables": n_staged, "elapsed_s": round(elapsed, 1)}
        timing_out = timing_out or (out_pkl.rsplit(".", 1)[0] + ".timing.json")
        with open(timing_out, "w") as f:
            json.dump(stats, f, indent=2)
        print(json.dumps(stats, indent=2))
        return stats
    finally:
        restore_or_absent(SANTOS_DIR, backup_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    extract(args.tier, args.csv_dir, args.out)


if __name__ == "__main__":
    main()
