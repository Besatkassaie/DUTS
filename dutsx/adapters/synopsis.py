"""``SynopsisSource`` adapters (PLAN-integration.md §2, Phase A).

Two implementations, deliberately built independently of each other so one
can serve as a staleness/correctness oracle for the other:

* ``MetadataStoreSynopsis`` wraps starmie_fair's ``TableMetadata.MetadataStore``,
  loaded from a ``.pkl``.
* ``CsvSynopsis`` recomputes everything from the CSVs directly, with pandas,
  using the exact same domain-size rule as starmie_fair's
  ``experimental_setup.py::_find_categorical_columns``
  (``0 < nunique(dropna()) <= theta_cat``). It does not import or wrap
  ``MetadataStore``/``TableMetadata`` in any way.

Phase A findings (santos, ``data/santos/indexes/metadata_datalake.pkl`` +
``metadata_query.pkl``, both dated Dec 2025; CSVs from 2021):

1. **No staleness.** Every one of the 550 datalake tables' ``n_i`` (row
   count) in the pickle matches a fresh CSV row count exactly (0
   mismatches), likewise all 50 query tables. A 40-table random sample of
   full ``{value: count}`` histograms also matched exactly, byte-for-byte.
   The Dec-2025/2021 date gap does not indicate drift here -- the pickle
   appears to have been re-serialized from the same CSVs, not built from
   different data.

2. **The pickle's own "categorical" filter is NOT ``theta_cat=50``, despite
   ``experimental_config.json``'s ``domain_size_threshold: 50``.** Every one
   of the 2220 distributions actually stored in ``metadata_datalake.pkl``
   that ``MetadataStoreSynopsis.categorical_attrs`` currently surfaces (see
   its docstring for a second, independent bug fixed 2026-08-13 -- domain
   size used to wrongly include the ``""`` empty-cell bucket, inflating this
   count to 2345 with 125 spurious all-empty "categorical" columns; the
   numbers here are already corrected) has domain size <= 10 (checked
   exhaustively -- the histogram of stored domain sizes has mass at 1..10 and
   nothing above). Columns with domain size in (10, 50] genuinely exist in
   the raw CSVs (e.g. ``311_calls_historic_data_0.csv``'s ``issue_type``,
   domain size 33) and are silently absent from ``categorical_distributions``
   -- not filtered by a downstream consumer, absent from the pickle itself.
   Recomputing with the *actually configured* ``theta_cat=50`` from scratch
   (``CsvSynopsis(theta_cat=50)``) gives 3583 surviving attributes and 52
   tables with zero categorical attributes; the pickle as shipped reports
   2220 and 96 respectively. A caller who trusts
   ``MetadataStoreSynopsis.categorical_attrs`` as "the theta_cat=50 answer"
   is silently working with a stricter, undocumented threshold (empirically
   ~10) baked in at some earlier build. This also means the two classes in
   this module are NOT expected to agree on ``categorical_attrs`` for the
   same table/threshold -- that disagreement is the whole point of keeping
   ``CsvSynopsis`` independent (Phase A's staleness-oracle requirement).

   Root cause not fully recovered: the currently-checked-in
   ``TableMetadata.py`` computes distributions for ALL columns
   unconditionally (its own docstring's claim) and its ``MetadataStore``
   pickles a ``_global_stats`` dict keyed
   ``total_columns_with_distributions``; the shipped pickle's
   ``_global_stats`` is keyed ``total_categorical_columns`` instead --
   different key names prove the pickle was built by a since-edited version
   of this class, one that filtered by domain size before storing. No
   script in this repo currently reproduces that build (grepped for
   ``metadata_datalake``/``metadata_query`` filenames and for a `10`
   threshold constant; neither matched). Treat this as an open item, not
   solved provenance.

3. **``biodiversity_2.csv``'s empty ``categorical_distributions`` is not a
   bug in either the current or the build-time code.** Its three columns
   (``scientific name``, ``family name``, ``common name``) have domain sizes
   257, 91, 257 over 257 rows -- every one exceeds *both* the configured
   ``theta_cat=50`` and the pickle's empirical ~10 cutoff. The table
   genuinely has zero categorical columns by any threshold in play; this is
   §8's "tables with no categorical attribute" trade-off, not a filtering
   defect. (96 of 550 datalake tables hit this under the pickle's actual
   filter; 52 of 550 under the correctly-configured ``theta_cat=50``.)
"""
import os
from typing import Dict, List, Optional, Union

import pandas as pd

from ..ports import AttrRef

DEFAULT_STARMIE_FAIR_ROOT = "/u6/bkassaie/starmie_fair"


def _ensure_starmie_fair_on_path(root: Optional[str] = None) -> str:
    """Put starmie_fair on ``sys.path`` so ``TableMetadata`` is importable.

    starmie_fair (``/u6/bkassaie/starmie_fair``) is read-only upstream data,
    not an installed package (CLAUDE.md) -- this mirrors the
    ``sys.path.insert`` pattern starmie_fair's own scripts use (e.g.
    ``fair_hnsw_eval/fair_starmie_search.py``).
    """
    import sys

    resolved = root or os.environ.get("STARMIE_FAIR_ROOT", DEFAULT_STARMIE_FAIR_ROOT)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return resolved


class MetadataStoreSynopsis:
    """``SynopsisSource`` backed by starmie_fair's ``MetadataStore.load(pkl_path)``.

    Trusts whatever filtering was actually baked into the pickle at build
    time for ``categorical_attrs`` -- see this module's docstring, finding 2:
    that filtering is NOT the configured ``theta_cat=50`` for
    ``data/santos/indexes/metadata_datalake.pkl``. Pass ``theta_cat`` to
    additionally cap any *surviving* stored distribution at that domain
    size; it can only narrow what the pickle already kept, never recover
    what the pickle never stored.
    """

    def __init__(
        self,
        pkl_path: str,
        theta_cat: Optional[int] = None,
        starmie_fair_root: Optional[str] = None,
    ):
        _ensure_starmie_fair_on_path(starmie_fair_root)
        from TableMetadata import MetadataStore  # noqa: E402  (path set above)

        self.pkl_path = pkl_path
        self.theta_cat = theta_cat
        self._store = MetadataStore.load(pkl_path)

    def n_rows(self, table: str) -> int:
        return self._store.get_num_records(table)

    def n_columns(self, table: str) -> int:
        """Total column count (not just categorical ones) -- beyond the ``SynopsisSource``
        protocol, same purpose as ``CsvSynopsis.n_columns`` (used by
        ``experiments/context.py::column_coverage_stats``). ``TableMetadata.num_columns`` is set
        from the header row at build time (``_process_csv_file``), independent of any
        categorical-domain filtering."""
        metadata = self._store.get_metadata(table)
        return metadata.num_columns if metadata else 0

    def distribution(self, table: str, attr: AttrRef) -> Dict[str, int]:
        dist = self._store.get_distribution(table, attr)
        return dict(dist) if dist else {}

    def categorical_attrs(self, table: str) -> List[int]:
        """A column is categorical iff ``0 < domain_size <= theta_cat`` (when ``theta_cat`` is
        given), matching ``CsvSynopsis.categorical_attrs``'s semantics exactly -- ``domain_size``
        excludes the ``""`` empty-cell bucket.

        **Bug fixed 2026-08-13**: this used to take ``domain_size = len(categorical_distributions[col])``
        directly, which counts ``""`` as a value like any other. Since ``TableMetadata._process_csv_file``
        stores a distribution for every column unconditionally (module docstring, finding 2) including
        all-empty ones, an all-empty column got ``domain_size=1`` (just the ``""`` bucket) and was
        wrongly classified categorical -- confirmed on WDC ``tier_10k`` (no prior domain-size
        pre-filter baked into the pickle, unlike the santos pickles this class was originally built
        against, where an earlier undocumented ~10-domain-size build filter likely masked the same
        bug for most tables): 274/2000 sampled tables had at least one such spurious column, inflating
        ``theta_cat=50`` categorical-column counts by ~4% tier-wide. ``0 < domain_size`` was also never
        checked at all when ``theta_cat is None`` -- now always enforced, matching ``CsvSynopsis``.
        """
        metadata = self._store.get_metadata(table)
        if metadata is None:
            return []
        out = []
        for col_name in metadata.categorical_columns:
            idx = metadata.get_column_index(col_name)
            if idx is None:
                continue
            dist = metadata.categorical_distributions.get(col_name, {})
            domain_size = sum(1 for v in dist if v != "")
            if domain_size == 0:
                continue
            if self.theta_cat is not None and domain_size > self.theta_cat:
                continue
            out.append(idx)
        return sorted(out)

    def has_table(self, table: str) -> bool:
        return self._store.get_metadata(table) is not None

    def n_tables(self) -> int:
        return len(self._store._metadata)  # noqa: SLF001 -- no public len() accessor


class CsvSynopsis:
    """``SynopsisSource`` that recomputes everything directly from CSVs.

    Genuinely independent of ``MetadataStore``/``TableMetadata`` -- this is
    the staleness oracle for ``MetadataStoreSynopsis`` (PLAN-integration.md
    Phase A accept criterion), so it must never import or wrap them.

    ``categorical_attrs`` mirrors starmie_fair's
    ``experimental_setup.py::_find_categorical_columns`` exactly: a column is
    categorical iff ``0 < nunique(dropna()) <= theta_cat``. ``distribution``
    counts every raw string value including empty cells (matching
    ``TableMetadata._process_csv_file``'s ``Counter(col_values)``, which does
    not drop empties) so ``N_of(dist, M)`` in ``duts/stats.py`` sees the same
    universe of tuples as ``n_rows`` -- dropping empties from
    ``distribution`` while ``n_rows`` counts them would silently make
    ``sum(dist.values()) < n_rows`` even when every value is populated in
    ``M``, which is exactly the "off by the missing bucket" mistake C2 warns
    against for ``N_i``.
    """

    def __init__(self, table_dirs: Union[str, List[str]], theta_cat: int = 50):
        self._table_dirs = [table_dirs] if isinstance(table_dirs, str) else list(table_dirs)
        self.theta_cat = theta_cat
        self._cache: Dict[str, pd.DataFrame] = {}
        self.parse_recovered: set = set()  # tables recovered via the tolerant reader in _df

    def _resolve_path(self, table: str) -> str:
        for d in self._table_dirs:
            candidate = os.path.join(d, table)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"table {table!r} not found in any of {self._table_dirs}"
        )

    def _df(self, table: str) -> pd.DataFrame:
        if table not in self._cache:
            # dtype=str + keep_default_na=False: read every cell as its raw
            # string, empty cells as "" rather than NaN. categorical_attrs
            # below does its own dropna-equivalent (excluding "") to match
            # experimental_setup.py's semantics for THAT computation only;
            # distribution() intentionally keeps "" as a real bucket (see
            # class docstring).
            path = self._resolve_path(table)
            try:
                self._cache[table] = pd.read_csv(
                    path, dtype=str, keep_default_na=False
                )
            except pd.errors.ParserError:
                # A malformed row must not take down a whole benchmark. Exactly
                # one of santosLarge's 11086 tables trips the C parser
                # (BEIS_exemption_2016-17_Q2_ICT.csv, buffer overflow on a bad
                # quote); santos/santos2/santos3/santos4 have none. Retry with
                # the tolerant python engine, skipping only the offending rows.
                # The recovered table has FEWER rows than the file, so n_i and
                # N_i for it are slight undercounts -- recorded here rather
                # than silently, and tracked in self.parse_recovered.
                self._cache[table] = pd.read_csv(
                    path, dtype=str, keep_default_na=False,
                    engine="python", on_bad_lines="skip",
                )
                self.parse_recovered.add(table)
        return self._cache[table]

    def _col_name(self, table: str, attr: AttrRef) -> str:
        df = self._df(table)
        if isinstance(attr, int):
            if not (0 <= attr < len(df.columns)):
                raise IndexError(f"attr index {attr} out of range for table {table!r}")
            return df.columns[attr]
        if attr not in df.columns:
            raise KeyError(f"attr {attr!r} not a column of table {table!r}")
        return attr

    def n_rows(self, table: str) -> int:
        return len(self._df(table))

    def n_columns(self, table: str) -> int:
        """Total column count (not just categorical ones) -- beyond the
        ``SynopsisSource`` protocol, used by ``experiments/context.py``'s
        column-coverage stat (fraction of columns theta_cat classifies as
        categorical). Reuses the same ``_df`` cache as every other method
        here, so it costs nothing beyond the read ``categorical_attrs``
        already pays."""
        return len(self._df(table).columns)

    def distribution(self, table: str, attr: AttrRef) -> Dict[str, int]:
        df = self._df(table)
        col = self._col_name(table, attr)
        counts = df[col].value_counts(dropna=False)
        return {str(k): int(v) for k, v in counts.items()}

    def categorical_attrs(self, table: str) -> List[int]:
        df = self._df(table)
        out = []
        for idx, col in enumerate(df.columns):
            domain_size = int(df[col][df[col] != ""].nunique())
            if 0 < domain_size <= self.theta_cat:
                out.append(idx)
        return out
