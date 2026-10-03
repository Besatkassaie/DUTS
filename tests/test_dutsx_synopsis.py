"""Tests for the ``SynopsisSource`` port and its two Phase A adapters
(PLAN-integration.md §4 Phase A).

Runs against the real santos data at ``/u6/bkassaie/starmie_fair`` (read-only
upstream, CLAUDE.md) rather than fixtures -- the whole point of this phase is
verifying the adapters against real ``.pkl``/CSV data, matching the repo's
existing "brute force / independent oracle" testing standard rather than
narrow example-based tests. The suite skips cleanly if that data isn't
mounted at the expected path.

``CsvSynopsis`` is used throughout as the independent oracle for
``MetadataStoreSynopsis`` -- it recomputes everything from CSVs with pandas
and never imports ``TableMetadata``/``MetadataStore``.
"""
import csv
import os
import random
from collections import Counter

import pytest

from dutsx.adapters.synopsis import CsvSynopsis, MetadataStoreSynopsis
from dutsx.ports import SynopsisSource
from dutsx.registry import build, build_from_config

STARMIE_FAIR_ROOT = "/u6/bkassaie/starmie_fair"
SANTOS_ROOT = os.path.join(STARMIE_FAIR_ROOT, "data", "santos")
DATALAKE_DIR = os.path.join(SANTOS_ROOT, "datalake")
QUERY_DIR = os.path.join(SANTOS_ROOT, "query")
DATALAKE_PKL = os.path.join(SANTOS_ROOT, "indexes", "metadata_datalake.pkl")
QUERY_PKL = os.path.join(SANTOS_ROOT, "indexes", "metadata_query.pkl")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(DATALAKE_DIR) and os.path.isfile(DATALAKE_PKL)),
    reason="starmie_fair santos data not available at the expected read-only path",
)


@pytest.fixture(scope="module")
def datalake_files():
    return sorted(os.listdir(DATALAKE_DIR))


@pytest.fixture(scope="module")
def metadata_store_synopsis():
    return MetadataStoreSynopsis(DATALAKE_PKL)


@pytest.fixture(scope="module")
def csv_synopsis_theta50():
    return CsvSynopsis([DATALAKE_DIR], theta_cat=50)


# ---------------------------------------------------------------------------
# Registry: config-driven selection (Phase A deliverable 3)
# ---------------------------------------------------------------------------

def test_registry_builds_metadata_store_adapter():
    adapter = build("synopsis", "metadata_store", pkl_path=DATALAKE_PKL)
    assert isinstance(adapter, MetadataStoreSynopsis)


def test_registry_builds_csv_adapter():
    adapter = build("synopsis", "csv", table_dirs=[DATALAKE_DIR], theta_cat=50)
    assert isinstance(adapter, CsvSynopsis)


def test_registry_unknown_adapter_name_raises_keyerror():
    with pytest.raises(KeyError):
        build("synopsis", "not_a_real_adapter")


def test_registry_unknown_port_raises_keyerror():
    with pytest.raises(KeyError):
        build("not_a_real_port", "metadata_store")


def test_registry_config_driven_selection_is_a_string_change():
    """The PLAN-integration.md §5 config shape: swapping the adapter is
    purely a change to cfg['adapters']['synopsis'], same call site."""
    cfg = {"adapters": {"synopsis": "metadata_store"}}
    a = build_from_config(cfg, "synopsis", pkl_path=DATALAKE_PKL)
    cfg["adapters"]["synopsis"] = "csv"
    b = build_from_config(cfg, "synopsis", table_dirs=[DATALAKE_DIR])
    assert isinstance(a, MetadataStoreSynopsis)
    assert isinstance(b, CsvSynopsis)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_both_adapters_satisfy_synopsis_source_protocol(
    metadata_store_synopsis, csv_synopsis_theta50
):
    assert isinstance(metadata_store_synopsis, SynopsisSource)
    assert isinstance(csv_synopsis_theta50, SynopsisSource)


# ---------------------------------------------------------------------------
# CsvSynopsis correctness -- checked against an INDEPENDENT stdlib-csv
# reimplementation written here, not against CsvSynopsis's own pandas logic,
# so this is a genuine oracle rather than a tautology.
# ---------------------------------------------------------------------------

def _read_via_stdlib_csv(path):
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0] if rows else []
    data = rows[1:] if rows else []
    return header, data


def test_csv_synopsis_n_rows_matches_independent_stdlib_reader(datalake_files):
    rng = random.Random(0)
    sample = rng.sample(datalake_files, 40)
    cs = CsvSynopsis([DATALAKE_DIR])
    for table in sample:
        _, data = _read_via_stdlib_csv(os.path.join(DATALAKE_DIR, table))
        assert cs.n_rows(table) == len(data), table


def test_csv_synopsis_n_columns_matches_independent_stdlib_reader(datalake_files):
    rng = random.Random(2)
    sample = rng.sample(datalake_files, 40)
    cs = CsvSynopsis([DATALAKE_DIR])
    for table in sample:
        header, _ = _read_via_stdlib_csv(os.path.join(DATALAKE_DIR, table))
        assert cs.n_columns(table) == len(header), table


def test_csv_synopsis_distribution_matches_independent_stdlib_reader(datalake_files):
    """distribution() counts every raw value including empty cells (matching
    TableMetadata._process_csv_file's Counter(col_values), not pandas'
    default NA-dropping) -- verified here against a from-scratch Counter."""
    rng = random.Random(1)
    sample = rng.sample(datalake_files, 25)
    cs = CsvSynopsis([DATALAKE_DIR])
    for table in sample:
        header, data = _read_via_stdlib_csv(os.path.join(DATALAKE_DIR, table))
        for idx in range(len(header)):
            vals = [row[idx] if idx < len(row) else "" for row in data]
            expected = dict(Counter(vals))
            assert cs.distribution(table, idx) == expected, (table, idx)


def test_csv_synopsis_distribution_by_name_matches_by_index(datalake_files):
    rng = random.Random(2)
    sample = rng.sample(datalake_files, 15)
    cs = CsvSynopsis([DATALAKE_DIR])
    for table in sample:
        header, _ = _read_via_stdlib_csv(os.path.join(DATALAKE_DIR, table))
        for idx, name in enumerate(header):
            assert cs.distribution(table, idx) == cs.distribution(table, name), (table, name)


def test_csv_synopsis_categorical_attrs_property_exhaustive_over_columns(datalake_files):
    """Brute-force check over every column of a random sample: an index is
    returned by categorical_attrs iff its true (dropna) domain size is in
    (0, theta_cat] -- checked both directions, not just for returned indices."""
    rng = random.Random(3)
    sample = rng.sample(datalake_files, 30)
    theta_cat = 50
    cs = CsvSynopsis([DATALAKE_DIR], theta_cat=theta_cat)
    for table in sample:
        header, data = _read_via_stdlib_csv(os.path.join(DATALAKE_DIR, table))
        returned = set(cs.categorical_attrs(table))
        for idx in range(len(header)):
            vals = {row[idx] for row in data if idx < len(row) and row[idx] != ""}
            domain_size = len(vals)
            should_survive = 0 < domain_size <= theta_cat
            assert (idx in returned) == should_survive, (table, idx, domain_size)


def test_csv_synopsis_unknown_table_raises():
    cs = CsvSynopsis([DATALAKE_DIR])
    with pytest.raises(FileNotFoundError):
        cs.n_rows("definitely_not_a_real_table.csv")


def test_csv_synopsis_out_of_range_attr_index_raises(datalake_files):
    cs = CsvSynopsis([DATALAKE_DIR])
    table = datalake_files[0]
    with pytest.raises(IndexError):
        cs.distribution(table, 10_000)


# ---------------------------------------------------------------------------
# Staleness check (Phase A deliverable, PLAN-integration.md's explicit
# accept criterion): does metadata_datalake.pkl (dated Dec 2025) still match
# a fresh read of the CSVs (from 2021)?
# ---------------------------------------------------------------------------

def test_staleness_n_rows_matches_csv_for_every_datalake_table(
    metadata_store_synopsis, datalake_files
):
    """Full scan, not a sample -- this IS the staleness acceptance check."""
    cs = CsvSynopsis([DATALAKE_DIR])
    mismatches = []
    for table in datalake_files:
        pkl_n = metadata_store_synopsis.n_rows(table)
        csv_n = cs.n_rows(table)
        if pkl_n != csv_n:
            mismatches.append((table, pkl_n, csv_n))
    assert mismatches == [], (
        f"{len(mismatches)}/{len(datalake_files)} tables disagree on n_i "
        f"between metadata_datalake.pkl and a fresh CSV count: {mismatches[:10]}"
    )


def test_staleness_n_rows_matches_csv_for_every_query_table():
    if not os.path.isfile(QUERY_PKL):
        pytest.skip("metadata_query.pkl not available")
    ms = MetadataStoreSynopsis(QUERY_PKL)
    cs = CsvSynopsis([QUERY_DIR])
    files = sorted(os.listdir(QUERY_DIR))
    mismatches = [
        (t, ms.n_rows(t), cs.n_rows(t)) for t in files if ms.n_rows(t) != cs.n_rows(t)
    ]
    assert mismatches == []


def test_staleness_distributions_match_for_random_sample(
    metadata_store_synopsis, datalake_files
):
    """Full {value: count} histograms, not just n_i -- a stronger staleness
    check than the row-count-only one PLAN-integration.md names."""
    rng = random.Random(4)
    sample = rng.sample(datalake_files, 40)
    cs = CsvSynopsis([DATALAKE_DIR])
    checked = 0
    for table in sample:
        for attr in metadata_store_synopsis.categorical_attrs(table):
            pkl_dist = metadata_store_synopsis.distribution(table, attr)
            csv_dist = cs.distribution(table, attr)
            assert pkl_dist == csv_dist, (table, attr)
            checked += 1
    assert checked > 0  # sanity: the sample actually exercised some columns


# ---------------------------------------------------------------------------
# Acceptance report numbers (Phase A): tables, attrs surviving theta_cat=50,
# tables with no categorical attribute -- pinned as regression fixtures
# since santos is static data, following the repo's convention of pinning
# concrete found instances (NOTES.md: alpha_induced, C4 counterexample).
# ---------------------------------------------------------------------------

def test_santos_table_count_is_550(datalake_files):
    assert len(datalake_files) == 550


def test_santos_theta_cat_50_counts_from_fresh_csv(csv_synopsis_theta50, datalake_files):
    """The real theta_cat=50 answer, independent of MetadataStore -- see
    dutsx/adapters/synopsis.py's module docstring, finding 2."""
    total_attrs = 0
    tables_no_cat = 0
    for table in datalake_files:
        n = len(csv_synopsis_theta50.categorical_attrs(table))
        total_attrs += n
        if n == 0:
            tables_no_cat += 1
    assert total_attrs == 3583
    assert tables_no_cat == 52


def test_metadata_store_pickle_filter_is_stricter_than_configured_theta_cat(
    metadata_store_synopsis, datalake_files
):
    """Documents (as an executable assertion, not just prose) that the
    pickle's own baked-in filter is NOT theta_cat=50: every stored
    distribution has domain size <= 10, and the pickle undercounts both
    total attributes and "has at least one categorical attribute" tables
    relative to a correctly-configured theta_cat=50 recompute."""
    total_attrs = 0
    tables_no_cat = 0
    max_domain_size = 0
    for table in datalake_files:
        md = metadata_store_synopsis._store.get_metadata(table)
        attrs = metadata_store_synopsis.categorical_attrs(table)
        total_attrs += len(attrs)
        if len(attrs) == 0:
            tables_no_cat += 1
        for col_name in md.categorical_columns:
            max_domain_size = max(max_domain_size, len(md.categorical_distributions[col_name]))

    # 2220/96, not the pre-fix 2345/95 -- MetadataStoreSynopsis.categorical_attrs used to count
    # the "" empty-cell bucket toward domain_size (bug fixed 2026-08-13, see that method's
    # docstring), so 125 all-empty columns in the shipped pickle were wrongly counted categorical.
    # This test's OWN point (the pickle's baked-in filter isn't theta_cat=50) still holds -- these
    # are the corrected numbers for that same claim, not a relaxation of it.
    assert total_attrs == 2220
    assert tables_no_cat == 96
    assert max_domain_size <= 10  # empirically ~10, not the configured 50

    # And the discrepancy against the correctly-configured theta_cat=50:
    assert total_attrs < 3583
    assert tables_no_cat > 52


def test_biodiversity_2_has_no_categorical_attrs_under_either_threshold(
    metadata_store_synopsis, csv_synopsis_theta50
):
    """Resolves the open question: MetadataStore's docstring claims
    distributions for ALL columns, but biodiversity_2.csv comes back empty.
    Its 3 columns (scientific name / family name / common name) have domain
    sizes 257/91/257 over 257 rows -- every one exceeds theta_cat=50 (the
    configured threshold) AND the pickle's stricter empirical ~10 cutoff.
    Not a bug in either the current TableMetadata.py or the build-time
    version that produced this pickle -- the table genuinely has zero
    categorical columns by any threshold in play."""
    table = "biodiversity_2.csv"
    assert metadata_store_synopsis.categorical_attrs(table) == []
    assert csv_synopsis_theta50.categorical_attrs(table) == []

    header, data = _read_via_stdlib_csv(os.path.join(DATALAKE_DIR, table))
    domain_sizes = []
    for idx in range(len(header)):
        vals = {row[idx] for row in data if idx < len(row) and row[idx] != ""}
        domain_sizes.append(len(vals))
    assert header == ["scientific name", "family name", "common name"]
    assert domain_sizes == [257, 91, 257]
    assert all(d > 50 for d in domain_sizes)  # exceeds every threshold considered
