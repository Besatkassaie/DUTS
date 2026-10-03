"""Deliberately re-select each santosLarge query's ``(protected_attribute,
protected_value)`` pair to maximize retrieved ``|D|``, replacing the random
assignment in ``starmie_fair/data/protected_attributes_santosLarge.csv``
(``experimental_setup.py``, ``selection_strategy='random'``, seed 42), which
picks a random categorical column then ``random.choice(unique_values)`` with
no regard for how common that value is elsewhere in the 11,086-table
datalake. That's why only 19 of 78 queries reach ``|D|>=50`` today
(``santoslarge_study.py``'s own docstring).

**Method.** Per query table, enumerate every ``(categorical attr, value)``
pair from its own synopsis (excluding the empty-string bucket -- a degenerate
"protected value" meaning missing data, not a real one). Rank all candidates
by a free proxy -- ``len(InvertedIndexOverlap._postings[value])``, the number
of ``(table, attr)`` pairs that would pass the overlap filter for that value,
computable with no retrieval at all since the index is already built. Verify
the top ``TOP_K`` candidates by actual retrieval (same ``top_n=5000``,
same HNSW+overlap context every santosLarge script uses) and keep whichever
gives the single largest real post-intersection ``|D|`` -- no target
threshold, no early exit (user's choice: maximize, not "clear a bar").

**This is a deliberate, documented departure from the original random
selection** -- it systematically favors common/generic values over rare
ones, which is the whole point (it's what makes |D| large), but it changes
what these queries represent and must be captioned as such wherever the
resulting numbers are shown, not silently substituted.

Output: a new protected-attributes CSV in the same 3-column format
(``q_name,protected_attribute_id,protected_value``), written under
``experiments/results/`` -- NEVER under ``starmie_fair/`` (CLAUDE.md: nothing
in this repo writes to that upstream repo) -- plus a before/after ``|D|``
report per query for transparency.
"""
import csv
import os
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

from dutsx.retrieval import retrieve_candidates
from dutsx.runner import retrieve_unscored_candidates, QueryTask

from . import context as ctx_mod
from .santoslarge_study import build_context

REPO = "/u6/bkassaie/DUTS"
TOP_N = 5000
TOP_K = 20         # candidates verified by real retrieval, per query table
EMPTY_VALUE = ""   # excluded: missing-data bucket, not a genuine value


class MaxDRow(NamedTuple):
    q_table: str
    orig_attr: int
    orig_value: str
    orig_n_D: int
    new_attr: int
    new_value: str
    new_n_D: int              # exact, via retrieve_unscored_candidates + drop_empty
    n_candidates_considered: int
    n_candidates_verified: int


def _proxy_rank(synopsis, overlap, q_table: str) -> List[Tuple[int, str, int]]:
    """All (attr, value, proxy_count) for q_table's categorical attrs,
    ranked descending by proxy_count. proxy_count = |postings[value]| --
    free, no retrieval."""
    postings = overlap._postings  # noqa: SLF001 -- offline analysis script, not production path
    candidates = []
    for attr in synopsis.categorical_attrs(q_table):
        dist = synopsis.distribution(q_table, attr)
        for value in dist:
            if value == EMPTY_VALUE:
                continue
            candidates.append((attr, value, len(postings.get(value, ()))))
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates


def _real_n_D(ctx, q_table: str, attr: int, value: str, top_n: int) -> int:
    query_vec = ctx.query_vectors[q_table][attr]
    _, telemetry = retrieve_candidates(ctx.semantic, ctx.overlap, query_vec, {value}, top_n)
    return telemetry.n_d


def _exact_n_D(ctx, q_table: str, attr: int, value: str, top_n: int) -> int:
    """The |D| the real experiment will actually see: retrieve_unscored_candidates
    + drop_empty (n_i=0 candidates dropped), matching santoslarge_study.py exactly."""
    from duts.stats import drop_empty
    task = QueryTask(q_table=q_table, attr=attr, M={value}, F_star=0.3, delta=0.15,
                      k=10, alpha=2.0, include_query=True, top_n=top_n)
    D, _, _, _ = retrieve_unscored_candidates(task, ctx)
    return len(drop_empty(D))


def run(protected_csv_in: Optional[str] = None, top_k: int = TOP_K,
        top_n: int = TOP_N, verbose: bool = True) -> List[MaxDRow]:
    paths = ctx_mod.paths_for("santosLarge")
    protected_csv_in = protected_csv_in or paths.protected_csv

    t0 = time.perf_counter()
    ctx, _paths, build_times = build_context()
    if verbose:
        print("context built in {:.1f}s ({} datalake tables)".format(
            time.perf_counter() - t0, build_times.n_datalake_tables))

    with open(protected_csv_in) as f:
        orig_rows = list(csv.DictReader(f))

    rows: List[MaxDRow] = []
    for i, r in enumerate(orig_rows):
        q_table = r["q_name"]
        orig_attr = int(r["protected_attribute_id"])
        orig_value = r["protected_value"]

        orig_n_D = _real_n_D(ctx, q_table, orig_attr, orig_value, top_n)

        candidates = _proxy_rank(ctx.synopsis, ctx.overlap, q_table)
        top = candidates[:top_k]

        best_attr, best_value, best_n_D = orig_attr, orig_value, orig_n_D
        for attr, value, _proxy in top:
            n_D = _real_n_D(ctx, q_table, attr, value, top_n)
            if n_D > best_n_D:
                best_attr, best_value, best_n_D = attr, value, n_D

        exact_n_D = _exact_n_D(ctx, q_table, best_attr, best_value, top_n)

        rows.append(MaxDRow(
            q_table=q_table, orig_attr=orig_attr, orig_value=orig_value, orig_n_D=orig_n_D,
            new_attr=best_attr, new_value=best_value, new_n_D=exact_n_D,
            n_candidates_considered=len(candidates), n_candidates_verified=len(top),
        ))
        if verbose:
            print("[{}/{}] {}: |D| {} -> {}".format(
                i + 1, len(orig_rows), q_table, orig_n_D, exact_n_D))

    return rows


def write_outputs(rows: List[MaxDRow], output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    protected_path = os.path.join(output_dir, "protected_attributes_santosLarge_maxD.csv")
    with open(protected_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_name", "protected_attribute_id", "protected_value"])
        for r in rows:
            w.writerow([r.q_table, r.new_attr, r.new_value])

    report_path = os.path.join(output_dir, "santoslarge_maxd_selection_report.csv")
    with open(report_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MaxDRow._fields)
        for r in rows:
            w.writerow(r)

    return protected_path, report_path


if __name__ == "__main__":
    rows = run()
    protected_path, report_path = write_outputs(rows, os.path.join(REPO, "experiments/results"))
    print("\nwrote", protected_path)
    print("wrote", report_path)
    ns = [r.new_n_D for r in rows]
    olds = [r.orig_n_D for r in rows]
    print("\nold |D|: min={} median={} max={}".format(min(olds), sorted(olds)[len(olds)//2], max(olds)))
    print("new |D|: min={} median={} max={}".format(min(ns), sorted(ns)[len(ns)//2], max(ns)))
    print("queries with new |D| >= 50: {}/{}".format(sum(1 for n in ns if n >= 50), len(ns)))
