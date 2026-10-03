"""Tests for ``scripts.wdc_json_to_csv``'s WDC-record parser (WDC scalability study Phase 1).

Fixture-based against hand-crafted JSON records matching the schema confirmed empirically
during planning (against real downloaded archive 00.tar.gz -- see the module docstring): the
tolerant-fallback paths (ragged columns) were never observed live in a 20,000-record sample, so
they're only exercisable here via constructed fixtures, not real data.
"""
import json

from scripts.wdc_json_to_csv import ParseOutcome, dedupe_headers, parse_wdc_record, safe_filename


def _rec(**overrides):
    base = {
        "relation": [["h1", "a", "b"], ["h2", "x", "y"]],  # column-major, 2 cols x 3 rows
        "hasHeader": True,
        "headerRowIndex": 0,
        "tableOrientation": "HORIZONTAL",
    }
    base.update(overrides)
    return json.dumps(base).encode("utf-8")


def test_normal_record_parses_with_header():
    outcome, headers, rows = parse_wdc_record(_rec())
    assert outcome == ParseOutcome.OK
    assert headers == ["h1", "h2"]
    assert rows == [["a", "x"], ["b", "y"]]


def test_no_header_uses_synthetic_column_names_and_keeps_all_rows():
    raw = _rec(hasHeader=False, headerRowIndex=-1)
    outcome, headers, rows = parse_wdc_record(raw)
    assert outcome == ParseOutcome.OK
    assert headers == ["col_0", "col_1"]
    assert rows == [["h1", "h2"], ["a", "x"], ["b", "y"]]


def test_vertical_orientation_relation_still_column_major_no_transpose():
    # confirmed empirically: relation layout is orientation-independent
    raw = _rec(tableOrientation="VERTICAL")
    outcome, headers, rows = parse_wdc_record(raw)
    assert outcome == ParseOutcome.OK
    assert headers == ["h1", "h2"]
    assert rows == [["a", "x"], ["b", "y"]]


def test_ragged_columns_truncated_and_marked_recovered():
    raw = json.dumps({
        "relation": [["h1", "a", "b"], ["h2", "x"]],  # col 0 has 3 rows, col 1 has 2
        "hasHeader": True,
        "headerRowIndex": 0,
    }).encode("utf-8")
    outcome, headers, rows = parse_wdc_record(raw)
    assert outcome == ParseOutcome.RECOVERED
    assert headers == ["h1", "h2"]
    assert rows == [["a", "x"]]  # truncated to the shorter column's length


def test_malformed_json_is_failed():
    outcome, headers, rows = parse_wdc_record(b"{not valid json")
    assert outcome == ParseOutcome.FAILED
    assert headers is None and rows is None


def test_missing_relation_is_failed():
    raw = json.dumps({"hasHeader": True}).encode("utf-8")
    outcome, headers, rows = parse_wdc_record(raw)
    assert outcome == ParseOutcome.FAILED


def test_empty_relation_is_failed():
    raw = json.dumps({"relation": [], "hasHeader": True}).encode("utf-8")
    outcome, headers, rows = parse_wdc_record(raw)
    assert outcome == ParseOutcome.FAILED


def test_zero_row_relation_is_failed():
    raw = json.dumps({"relation": [[], []], "hasHeader": False}).encode("utf-8")
    outcome, headers, rows = parse_wdc_record(raw)
    assert outcome == ParseOutcome.FAILED


def test_header_only_table_with_no_data_rows_is_failed():
    # 1 row total, that row IS the header -> zero data rows left
    raw = json.dumps({
        "relation": [["h1"], ["h2"]],
        "hasHeader": True,
        "headerRowIndex": 0,
    }).encode("utf-8")
    outcome, headers, rows = parse_wdc_record(raw)
    assert outcome == ParseOutcome.FAILED


def test_dedupe_headers_handles_empty_and_duplicate_values():
    assert dedupe_headers(["a", "", "a", ""]) == ["a", "col", "a.1", "col.1"]


def test_safe_filename_flattens_path_separators():
    name = safe_filename("0/1438042988458.74_20150728002308-00254-ip-10-236-191-2_867263717_6.json")
    assert "/" not in name
    assert name.endswith(".csv")
