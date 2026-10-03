"""Santos-style table-union-search groundtruth (``<benchmark>_small_benchmark_
groundtruth.csv``, present for all four santos variants) and precision/recall
against it.

**What this measures, and what it doesn't.** The groundtruth's own
``intent_col_index`` (the join/entity column two tables are considered
unionable *on*) is almost always a DIFFERENT column from ``V_D`` (the
protected/distributional attribute DUTS optimizes for) -- verified: only 4/48
santos base queries have ``intent_col_index == protected_attribute_id``, 44
differ. So precision/recall here does NOT test whether DUTS found the
"right" column; the column identity is intentionally ignored (grouped by
``query_table`` only). It tests something orthogonal and still real: **does
distribution-first retrieval and selection (favoring tables that help satisfy
`tau`) throw away tables the benchmark considers genuinely unionable, or let
through tables it doesn't?**

The groundtruth includes a self-match row for every query (the query's own
name as one of its ``data_lake_table`` matches), which is not a data error --
every santos query table also has a same-named file physically present in
`datalake/` (verified for all 50/50 santos queries), so a candidate pipeline
CAN legitimately retrieve/select the query's own row.
"""
import csv
from collections import defaultdict
from typing import Dict, FrozenSet, Iterable, Set


def load_groundtruth(path: str) -> Dict[str, FrozenSet[str]]:
    """``{query_table: frozenset(data_lake_table)}``, grouped across all
    ``intent_col_index`` values for that query (see module docstring for
    why the column is ignored). ``utf-8-sig`` handles the file's BOM."""
    groups: Dict[str, Set[str]] = defaultdict(set)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            groups[row["query_table"]].add(row["data_lake_table"])
    return {q: frozenset(tables) for q, tables in groups.items()}


def precision_recall(retrieved: Iterable[str], truth: FrozenSet[str]):
    """``(precision, recall, hits)`` for one query's one stage.

    ``precision`` is ``None`` when ``retrieved`` is empty (undefined, not
    zero -- an empty result says nothing about the retriever's accuracy).
    ``recall`` is ``None`` when ``truth`` is empty (no groundtruth for this
    query -- shouldn't happen for the four santos variants, but guarded
    rather than raising ``ZeroDivisionError`` on unfamiliar data).
    """
    retrieved_set = set(retrieved)
    hits = len(retrieved_set & truth)
    precision = (hits / len(retrieved_set)) if retrieved_set else None
    recall = (hits / len(truth)) if truth else None
    return precision, recall, hits
