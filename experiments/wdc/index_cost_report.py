"""Per-tier, per-index build-cost report (user-requested ad hoc experiment, 2026-08-14 --
not part of ``PLAN-integration.md``'s phase sequence).

Answers: for each WDC tier, how long does each index (HNSW, posting-list/inverted-index) take
to build -- excluding model inference (embedding extraction is a separate, already-timed, offline
GPU step; see ``WdcBuildTimes.embedding_extraction_s``) -- how many attributes does each index
actually cover vs. how many attributes exist in total, and how large is each index on disk.

Neither adapter persists to disk in normal operation (both are rebuilt in-memory from the vectors
pkl / synopsis on every ``build_wdc_context`` call), so "size on disk" is measured by actually
serializing each once, here, for reporting purposes only (the scratch files are deleted immediately
after measuring):

- HNSW: ``hnswlib.Index.save_index(path)`` -- a real, native binary format.
- Posting list: no native serialization exists (``InvertedIndexOverlap._postings`` is a plain
  ``Dict[str, Set[Tuple[str, int]]]``) -- pickled to measure a representative byte size. This is a
  proxy (pickle framing overhead included), not a purpose-built on-disk format, so it is NOT an
  apples-to-apples comparison with HNSW's native size -- flagged in the table's footnotes too.

"Attributes indexed" differs by index by design, not by bug: ``HnswRetriever`` indexes every
column with a nonzero embedding (no ``theta_cat`` filter at all -- see
``dutsx/adapters/semantic.py::_flatten``), while ``InvertedIndexOverlap`` only indexes categorical
columns (``synopsis.categorical_attrs``, ``theta_cat``-bounded). Both are reported against the same
"attributes total" denominator (``WdcBuildTimes.n_columns_total``, from
``experiments/context.py::column_coverage_stats``).
"""
import csv
import os
import pickle
import tempfile
from typing import List, NamedTuple

from .context import build_wdc_context

TIERS = ("tier_10k", "tier_100k")


class IndexCostRow(NamedTuple):
    tier: str
    n_tables: int
    n_attrs_total: int
    n_attrs_indexed_hnsw: int
    n_attrs_indexed_postlist: int
    setup_load_s: float
    synopsis_build_s: float
    hnsw_build_s: float
    hnsw_size_bytes: int
    postlist_build_s: float
    postlist_size_bytes: int


def _hnsw_disk_size(semantic_obj) -> int:
    fd, path = tempfile.mkstemp(suffix=".hnsw")
    os.close(fd)
    try:
        semantic_obj.index.save_index(path)
        return os.path.getsize(path)
    finally:
        os.remove(path)


def _postlist_disk_size(overlap_obj) -> int:
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            pickle.dump(dict(overlap_obj._postings), f, protocol=pickle.HIGHEST_PROTOCOL)
        return os.path.getsize(path)
    finally:
        os.remove(path)


def build_report(tiers=TIERS, verbose: bool = True) -> List[IndexCostRow]:
    rows: List[IndexCostRow] = []
    for tier in tiers:
        ctx, bt = build_wdc_context(tier)
        hnsw_bytes = _hnsw_disk_size(ctx.semantic)
        postlist_bytes = _postlist_disk_size(ctx.overlap)
        row = IndexCostRow(
            tier=tier,
            n_tables=bt.n_tables,
            n_attrs_total=bt.n_columns_total,
            n_attrs_indexed_hnsw=ctx.semantic.n_indexed,
            n_attrs_indexed_postlist=ctx.overlap.n_pairs_indexed,
            setup_load_s=bt.load_resources_s,
            synopsis_build_s=bt.synopsis_build_s,
            hnsw_build_s=bt.hnsw_build_s,
            hnsw_size_bytes=hnsw_bytes,
            postlist_build_s=bt.inverted_index_build_s,
            postlist_size_bytes=postlist_bytes,
        )
        rows.append(row)
        if verbose:
            print(
                "[{}] tables={} attrs_total={} hnsw: indexed={} build={:.3f}s "
                "size={:.1f}MB | postlist: indexed={} build={:.3f}s size={:.1f}MB".format(
                    tier, row.n_tables, row.n_attrs_total,
                    row.n_attrs_indexed_hnsw, row.hnsw_build_s, row.hnsw_size_bytes / 1e6,
                    row.n_attrs_indexed_postlist, row.postlist_build_s, row.postlist_size_bytes / 1e6,
                ), flush=True,
            )
    return rows


def write_rows(rows: List[IndexCostRow], name: str = "wdc_index_cost_report",
               output_dir: str = "experiments/results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(IndexCostRow._fields)
        for r in rows:
            w.writerow(r)
    return path


if __name__ == "__main__":
    rows = build_report()
    path = write_rows(rows)
    print("wrote", path)
