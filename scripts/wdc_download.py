"""Resumable downloader for the WDC Web Table Corpus 2015, English relational subset.

51 tar.gz archives (`00.tar.gz`..`50.tar.gz`), ~1.4GB each, ~69GB total, hosted at Mannheim
(confirmed via HEAD request during planning: 200 OK, Accept-Ranges: bytes, Content-Length present).
Downloads to a `.partial` file and atomically renames on completion, so a rerun skips archives
already marked complete in the manifest and resumes a partial one via HTTP Range.

Usage:
    python scripts/wdc_download.py --archives 2 --dest /u6/bkassaie/wdc_data/raw
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request

BASE_URL = "https://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed"
CHUNK_SIZE = 1 << 20  # 1MB


def _manifest_path(dest: str) -> str:
    return os.path.join(dest, "_download_manifest.json")


def _load_manifest(dest: str) -> dict:
    path = _manifest_path(dest)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_manifest(dest: str, manifest: dict) -> None:
    path = _manifest_path(dest)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers["Content-Length"])


def download_archive(index: int, dest: str, manifest: dict) -> None:
    name = "%02d.tar.gz" % index
    url = "%s/%s" % (BASE_URL, name)
    final_path = os.path.join(dest, name)
    partial_path = final_path + ".partial"

    if manifest.get(name, {}).get("status") == "complete" and os.path.exists(final_path):
        print("skip (already complete): %s" % name)
        return

    total = _remote_size(url)
    resume_from = os.path.getsize(partial_path) if os.path.exists(partial_path) else 0
    if resume_from >= total:
        resume_from = 0  # stale/corrupt partial larger than remote; restart

    req = urllib.request.Request(url)
    if resume_from > 0:
        req.add_header("Range", "bytes=%d-" % resume_from)
        print("resuming %s from byte %d/%d" % (name, resume_from, total))
    else:
        print("downloading %s (%d bytes)" % (name, total))

    mode = "ab" if resume_from > 0 else "wb"
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp, open(partial_path, mode) as out:
        written = resume_from
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)

    if written != total:
        manifest[name] = {"status": "partial", "bytes": written, "expected": total}
        _save_manifest(dest, manifest)
        raise RuntimeError(
            "%s: downloaded %d bytes, expected %d (rerun to resume)" % (name, written, total)
        )

    os.replace(partial_path, final_path)
    elapsed = time.time() - t0
    manifest[name] = {"status": "complete", "bytes": total, "elapsed_s": round(elapsed, 1)}
    _save_manifest(dest, manifest)
    print("done: %s (%d bytes, %.1fs)" % (name, total, elapsed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", type=int, default=2, help="number of archives (00..N-1) to fetch")
    ap.add_argument("--dest", type=str, default="/u6/bkassaie/wdc_data/raw")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    manifest = _load_manifest(args.dest)
    for i in range(args.archives):
        download_archive(i, args.dest, manifest)


if __name__ == "__main__":
    main()
