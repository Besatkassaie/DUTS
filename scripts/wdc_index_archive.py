"""Unwrap a downloaded WDC archive and build a member index for random access.

Each downloaded `NN.tar.gz` is a gzip-compressed outer tar with exactly ONE member: an
uncompressed inner `NN.tar` (several GB) that holds the ~1M individual JSON table files
(confirmed empirically against archive 00 during planning -- not assumed from corpus docs).

gzip streams are not efficiently seekable to an arbitrary byte offset (reaching a member near
the end would mean decompressing almost the whole archive), so this script:

  1. Unwraps the outer .tar.gz once, writing the inner .tar to disk as a single file (NOT
     exploding into ~1M loose files -- avoids the exact NFS-inode-storm the plan's Phase 0
     was designed to prevent).
  2. Builds a `{member_name: {"offset": int, "size": int}}` index over that INNER,
     uncompressed tar, which genuinely supports fast `seek()`-based random access.

Downstream readers (Phase 1's JSON->CSV converter) reopen the inner .tar and use
`TarFile.extractfile()` with this index for O(1) member access, never `extractall()`.

Usage:
    python scripts/wdc_index_archive.py --archive /u6/bkassaie/wdc_data/raw/00.tar.gz
"""
import argparse
import json
import os
import tarfile
import time


def unwrap_inner_tar(archive_path: str) -> str:
    """Extract the single inner .tar member to disk (idempotent). Returns its path."""
    inner_path = archive_path[: -len(".tar.gz")] + ".tar" if archive_path.endswith(".tar.gz") \
        else archive_path + ".inner.tar"
    if os.path.exists(inner_path):
        return inner_path

    tmp_path = inner_path + ".partial"
    with tarfile.open(archive_path, mode="r:gz") as outer:
        members = [m for m in outer.getmembers() if m.isfile()]
        if len(members) != 1:
            raise RuntimeError(
                "%s: expected exactly 1 inner member, found %d (%s)"
                % (archive_path, len(members), [m.name for m in members[:5]])
            )
        inner_member = members[0]
        t0 = time.time()
        with outer.extractfile(inner_member) as src, open(tmp_path, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
        print("unwrapped %s (%d bytes) in %.1fs" % (
            inner_member.name, inner_member.size, time.time() - t0
        ))
    os.replace(tmp_path, inner_path)
    return inner_path


def build_index(inner_tar_path: str) -> dict:
    """-> {member_name: {"offset": int, "size": int}} for every regular-file member."""
    index = {}
    with tarfile.open(inner_tar_path, mode="r:") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            index[member.name] = {"offset": member.offset_data, "size": member.size}
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="path to a downloaded NN.tar.gz")
    ap.add_argument("--index-out", default=None, help="defaults to <inner_tar>.index.json")
    args = ap.parse_args()

    inner_path = unwrap_inner_tar(args.archive)
    index_out = args.index_out or (inner_path + ".index.json")

    t0 = time.time()
    index = build_index(inner_path)
    elapsed = time.time() - t0

    with open(index_out, "w") as f:
        json.dump(index, f)

    print("indexed %d members from %s in %.1fs -> %s" % (
        len(index), inner_path, elapsed, index_out
    ))


if __name__ == "__main__":
    main()
