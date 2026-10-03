"""Tests for the Phase E experiment harness (PLAN-integration.md §4 Phase E).

Two kinds, matching the ``test_dutsx_*.py`` convention:

* Stub/synthetic tests (no real data dependency) -- schema stability, row
  construction, aggregation correctness in ``analysis.py``, and determinism
  of the alpha-sweep driver given a seed. Run unconditionally, fast.
* A couple of tiny real-data smoke tests (2-3 queries, a small k/alpha grid)
  that exercise the actual CLI/driver wiring end to end against real santos
  data. Skip cleanly if that data isn't mounted. Deliberately NOT the full
  48-query x 15-config sweep -- that's what ``experiments/results/*.csv``
  (committed, produced by a real run) is for.
"""
import numpy as np
import pandas as pd
import pytest

from duts.stage2_ilp import solve as stage2_solve
from duts.types import CandidateStats
from dutsx.runner import QueryTask, RunnerContext, run_query

from experiments import analysis, context as ctx_mod
from experiments.alpha_sweep import Stage2Instance, run_alpha_sweep
from experiments.lp_precheck import run_lp_precheck_experiment, run_lp_precheck_synthetic_demo
from experiments.retrieval_ablation import run_retrieval_ablation
from experiments.stage1_skip_ablation import run_query_skip_stage1, run_stage1_skip_ablation
from experiments.schema import FIELDS, ResultRow, row_from_result, rows_to_dataframe, write_csv

# ---------------------------------------------------------------------------
# Stub adapters -- same shape as tests/test_dutsx_runner.py's, kept local so
# this file has no cross-test-module dependency.
# ---------------------------------------------------------------------------


class _StubSynopsis(object):
    def __init__(self, n_rows, distributions):
        self._n_rows = n_rows
        self._dist = distributions

    def n_rows(self, table):
        return self._n_rows[table]

    def distribution(self, table, attr):
        return self._dist.get((table, attr), {})

    def categorical_attrs(self, table):
        return [0]


class _StubSemantic(object):
    def __init__(self, results):
        self._results = results

    def query(self, vec, top_n):
        return self._results[:top_n]


class _StubOverlap(object):
    def __init__(self, pairs):
        self._pairs = pairs

    def query(self, M):
        return dict(self._pairs)


class _StubUnionability(object):
    def __init__(self, value=1.0):
        self.value = value
        self.n_scored = 0

    def score(self, q_table, c_table, pin):
        self.n_scored += 1
        return self.value


def _make_ctx(n_tables, n_i=10, N_i=5, q_n=10, q_N=5, u_value=1.0):
    tables = ["t{}".format(i) for i in range(n_tables)]
    n_rows = {t: n_i for t in tables}
    n_rows["Q"] = q_n
    dist = {}
    for t in tables:
        dist[(t, 0)] = {"m": N_i, "x": max(n_i - N_i, 0)}
    dist[("Q", 0)] = {"m": q_N, "x": max(q_n - q_N, 0)}
    synopsis = _StubSynopsis(n_rows, dist)
    semantic = _StubSemantic([(t, 0, 1.0 - i * 1e-6) for i, t in enumerate(tables)])
    overlap = _StubOverlap({(t, 0): 1 for t in tables})
    unionability = _StubUnionability(value=u_value)
    query_vectors = {"Q": np.zeros((1, 4))}
    ctx = RunnerContext(
        synopsis=synopsis, semantic=semantic, overlap=overlap,
        unionability=unionability, query_vectors=query_vectors,
    )
    return ctx


def _base_task(k=3, alpha=2, F_star=0.3, delta=0.3):
    return QueryTask(
        q_table="Q", attr=0, M={"m"}, F_star=F_star, delta=delta, k=k, alpha=alpha,
    )


ADAPTERS = {"synopsis": "stub", "semantic": "stub", "overlap": "stub", "unionability": "stub"}


# ---------------------------------------------------------------------------
# schema.py -- stability and row construction
# ---------------------------------------------------------------------------


def test_result_row_field_set_is_stable():
    """Pins the column contract -- a change here is a schema break that every
    downstream CSV consumer (and the committed experiments/results/*.csv)
    would silently disagree with."""
    expected = {
        "experiment", "config_id", "seed", "q_table", "attr", "M_size",
        "k", "alpha", "F_star", "delta", "include_query", "top_n",
        "adapter_synopsis", "adapter_semantic", "adapter_overlap", "adapter_unionability",
        "lp_precheck", "n_sem", "n_ovl", "n_pair",
        "feasible", "infeasibility_cause", "error",
        "n_D", "n_P", "n_R", "F_P", "F_R", "delta_R", "sum_U",
        "n_unionability_computations", "dinkelbach_iterations_stage1",
        "retrieval_time_s", "stage1_time_s", "effective_alpha",
        "ilp_time_s", "lp_precheck_ran", "lp_feasible", "lp_time_s",
        "wall_time_s",
        # groundtruth precision/recall (experiments/groundtruth_eval.py) -- None
        # for every other experiment's rows, see schema.py's ResultRow docstring.
        "gt_size", "d_hits", "p_hits", "r_hits",
        "precision_d", "recall_d", "precision_p", "recall_p", "precision_r", "recall_r",
        # stage1-skip ablation (experiments/stage1_skip_ablation.py) -- None elsewhere.
        "scoring_time_s",
    }
    assert set(FIELDS) == expected


def test_row_from_result_feasible_case():
    ctx = _make_ctx(n_tables=20)
    task = _base_task(k=3, alpha=2)
    result = run_query(task, ctx)
    row = row_from_result("unit", "cfg", 42, task, result, 0.01, ADAPTERS)

    assert row.experiment == "unit" and row.config_id == "cfg" and row.seed == 42
    assert row.q_table == "Q" and row.k == 3 and row.alpha == 2
    assert row.feasible == result.telemetry.feasible
    assert row.n_D == result.telemetry.n_D
    assert row.n_unionability_computations == result.telemetry.n_unionability_computations
    assert row.error is None


def test_row_from_result_error_case():
    task = _base_task()
    row = row_from_result(
        "unit", "cfg", 42, task, None, 0.02, ADAPTERS, error="KeyError: boom",
    )
    assert row.feasible is False
    assert row.error == "KeyError: boom"
    assert row.infeasibility_cause == "error"
    assert row.n_D == 0


def test_write_csv_roundtrip(tmp_path):
    ctx = _make_ctx(n_tables=20)
    task = _base_task()
    result = run_query(task, ctx)
    row = row_from_result("unit", "cfg", 42, task, result, 0.01, ADAPTERS)

    out = tmp_path / "rows.csv"
    write_csv([row], str(out))
    df = pd.read_csv(str(out))
    assert list(df.columns) == list(FIELDS)
    assert len(df) == 1
    assert df.iloc[0]["q_table"] == "Q"


# ---------------------------------------------------------------------------
# stage1_skip_ablation.py -- skip-Stage-1 composition, on stub adapters
# ---------------------------------------------------------------------------


def test_run_query_skip_stage1_scores_all_of_D_not_just_alpha_k():
    """The whole point of the ablation: skip_stage1 scores every retrieved
    candidate, while two_stage (dutsx.runner.run_query) scores only
    alpha*k -- distinguishable whenever alpha*k < |D|."""
    ctx = _make_ctx(n_tables=30)
    task = _base_task(k=5, alpha=1)  # pool_size = 5 << |D| = 30

    result_two = run_query(task, ctx)
    result_skip, scoring_time_s = run_query_skip_stage1(task, ctx)

    assert result_two.telemetry.n_unionability_computations == 5
    assert result_skip.telemetry.n_unionability_computations == 30
    assert scoring_time_s >= 0.0


def test_run_query_skip_stage1_never_scores_worse_than_two_stage():
    """Stage 2 in skip_stage1 optimizes ΣU over a superset of what two_stage
    sees (D ⊇ P) subject to the identical constraint -- its optimum can only
    be >= two_stage's (PLAN.md's Stage 2 exactness + monotonicity under
    superset feasible sets), never strictly worse."""
    ctx = _make_ctx(n_tables=30, u_value=1.0)
    task = _base_task(k=5, alpha=1)

    result_two = run_query(task, ctx)
    result_skip, _ = run_query_skip_stage1(task, ctx)

    assert result_two.telemetry.feasible and result_skip.telemetry.feasible
    assert result_skip.telemetry.sum_U >= result_two.telemetry.sum_U - 1e-9


def test_run_query_skip_stage1_matches_two_stage_when_D_fits_in_the_pool():
    """When |D| <= alpha*k, two_stage's C5 clamp makes P == D -- so both
    conditions score and solve over the identical candidate set and must
    agree exactly (same |R|, same sum_U)."""
    ctx = _make_ctx(n_tables=8)
    task = _base_task(k=5, alpha=2)  # pool_size = 10 > |D| = 8

    result_two = run_query(task, ctx)
    result_skip, _ = run_query_skip_stage1(task, ctx)

    assert result_two.telemetry.n_P == result_skip.telemetry.n_P == 8
    assert result_two.telemetry.sum_U == pytest.approx(result_skip.telemetry.sum_U)


def test_run_query_skip_stage1_insufficient_candidates_matches_two_stage():
    ctx = _make_ctx(n_tables=3)
    task = _base_task(k=5, alpha=1)

    result_two = run_query(task, ctx)
    result_skip, scoring_time_s = run_query_skip_stage1(task, ctx)

    assert result_two.telemetry.infeasibility_cause == "insufficient_candidates"
    assert result_skip.telemetry.infeasibility_cause == "insufficient_candidates"
    assert scoring_time_s == 0.0


def test_stage1_skip_ablation_row_pair_shares_config_suffix():
    """run_stage1_skip_ablation emits two rows per query, distinguished by a
    condition prefix on config_id (schema.py has no dedicated 'condition'
    column -- see experiments/cli.py's cmd_stage1_skip_ablation, which
    derives one by splitting on '__')."""
    ctx = _make_ctx(n_tables=20)
    task = _base_task(k=3, alpha=2)._replace(q_table="Q")

    rows = []
    for label, fn in (("two_stage", run_query), ("skip_stage1", run_query_skip_stage1)):
        if fn is run_query:
            result = fn(task, ctx)
        else:
            result, _ = fn(task, ctx)
        rows.append(row_from_result(
            "stage1_skip_ablation", "{}__k3_a2".format(label), 42, task, result, 0.01, ADAPTERS,
        ))
    assert rows[0].config_id == "two_stage__k3_a2"
    assert rows[1].config_id == "skip_stage1__k3_a2"
    assert rows[0].q_table == rows[1].q_table == "Q"


# ---------------------------------------------------------------------------
# alpha_sweep.py -- driver correctness and determinism, on stub adapters
# ---------------------------------------------------------------------------


def test_alpha_sweep_produces_one_row_per_query_per_condition():
    ctx = _make_ctx(n_tables=30)
    base = [_base_task(k=1, alpha=1)._replace(q_table="Q")]
    rows, instances = run_alpha_sweep(
        base, ctx, ADAPTERS, alphas=(1, 2), ks=(3, 5), seed=7, verbose=False,
    )
    assert len(rows) == len(base) * 2 * 2  # 1 query x 2 alphas x 2 ks
    config_ids = {r.config_id for r in rows}
    assert config_ids == {"k=3_alpha=1", "k=3_alpha=2", "k=5_alpha=1", "k=5_alpha=2"}


def test_alpha_sweep_harvests_stage2_instances_only_when_pool_nonempty():
    ctx = _make_ctx(n_tables=3)  # too few for k=5 -- insufficient_candidates, n_P=0
    base = [_base_task(k=5, alpha=1)._replace(q_table="Q")]
    rows, instances = run_alpha_sweep(base, ctx, ADAPTERS, alphas=(1,), ks=(5,), verbose=False)
    assert rows[0].infeasibility_cause == "insufficient_candidates"
    assert instances == []  # nothing harvested -- Stage 1 never ran


def test_alpha_sweep_is_deterministic_given_same_inputs():
    """Two independent runs of the same stub sweep must agree on every
    column except wall time -- the reproducibility contract (CLAUDE.md C5:
    determinism is required, not cosmetic; this is the Phase E analogue)."""
    def run():
        ctx = _make_ctx(n_tables=30, n_i=10, N_i=6, q_n=10, q_N=6)
        base = [_base_task(k=4, alpha=2, F_star=0.5, delta=0.2)._replace(q_table="Q")]
        rows, _ = run_alpha_sweep(base, ctx, ADAPTERS, alphas=(1, 2, 3), ks=(4,), seed=42, verbose=False)
        return rows

    rows_a, rows_b = run(), run()
    non_timing_fields = [f for f in FIELDS if not f.endswith("_time_s")]
    a = [tuple(getattr(r, f) for f in non_timing_fields) for r in rows_a]
    b = [tuple(getattr(r, f) for f in non_timing_fields) for r in rows_b]
    assert a == b


def test_error_in_run_query_is_captured_not_raised():
    """A malformed task (bad attr index) must produce an error row, not kill
    the whole sweep -- the real 48-query sweep cannot afford one bad query
    to abort everything."""
    ctx = _make_ctx(n_tables=10)
    bad_task = _base_task()._replace(q_table="does-not-exist")
    rows, instances = run_alpha_sweep([bad_task], ctx, ADAPTERS, alphas=(1,), ks=(3,), verbose=False)
    assert len(rows) == 1
    assert rows[0].error is not None
    assert rows[0].feasible is False
    assert instances == []


# ---------------------------------------------------------------------------
# lp_precheck.py -- replays a synthetic Stage2Instance, no real data needed
# ---------------------------------------------------------------------------


def test_lp_precheck_experiment_produces_two_rows_per_instance_and_agrees():
    pool = [
        CandidateStats(table="t{}".format(i), N=i % 4, n=10, U=float(i)) for i in range(8)
    ]
    task = _base_task(k=3, alpha=2, F_star=0.2, delta=0.5)
    inst = Stage2Instance(config_id="k=3_alpha=2", task=task, N_Q=1, n_Q=5, P=pool)

    rows = run_lp_precheck_experiment([inst], ADAPTERS, seed=1, verbose=False)

    assert len(rows) == 2
    lp_true = [r for r in rows if r.lp_precheck][0]
    lp_false = [r for r in rows if not r.lp_precheck][0]
    assert lp_true.feasible == lp_false.feasible
    assert lp_true.n_R == lp_false.n_R
    assert lp_true.ilp_time_s is not None and lp_true.ilp_time_s >= 0
    assert lp_true.lp_precheck_ran is True
    assert lp_false.lp_precheck_ran is False

    # cross-check against calling duts.stage2_ilp.solve directly
    tau = task.F_star - task.delta
    direct = stage2_solve(pool, task.k, tau, inst.N_Q, inst.n_Q, task.include_query)
    assert lp_true.feasible == direct.feasible


def test_lp_precheck_experiment_handles_infeasible_instance():
    """N=0 for every candidate, high F_star -- guaranteed Stage 2 infeasible;
    both variants must agree it's infeasible (LP/ILP equivalence, NOTES.md)."""
    pool = [CandidateStats(table="t{}".format(i), N=0, n=10, U=1.0) for i in range(6)]
    task = _base_task(k=3, alpha=2, F_star=0.9, delta=0.01)
    inst = Stage2Instance(config_id="cfg", task=task, N_Q=0, n_Q=10, P=pool)

    rows = run_lp_precheck_experiment([inst], ADAPTERS, seed=1, verbose=False)
    assert all(r.feasible is False for r in rows)
    assert all(r.n_R == 0 for r in rows)


def test_lp_precheck_synthetic_demo_produces_both_feasible_and_infeasible_instances():
    """Tiny synthetic pool (well below the 100-3000 scale used for the real
    demo, kept fast for the test suite) -- checks the alternation actually
    produces both outcomes and that direct-ILP vs LP-precheck agree on
    feasibility for every instance (the same equivalence invariant)."""
    rows = run_lp_precheck_synthetic_demo(seed=1, n_instances=6, pool_size=20, verbose=False)
    assert len(rows) == 12  # 6 instances x 2 variants
    feasible_flags = {r.feasible for r in rows}
    assert feasible_flags == {True, False}  # both outcomes exercised

    by_instance = {}
    for r in rows:
        by_instance.setdefault(r.q_table, []).append(r)
    for q_table, instance_rows in by_instance.items():
        assert len(instance_rows) == 2
        assert instance_rows[0].feasible == instance_rows[1].feasible, q_table


def test_lp_precheck_synthetic_demo_is_deterministic_given_seed():
    a = run_lp_precheck_synthetic_demo(seed=7, n_instances=4, pool_size=20, verbose=False)
    b = run_lp_precheck_synthetic_demo(seed=7, n_instances=4, pool_size=20, verbose=False)
    non_timing = [f for f in FIELDS if not f.endswith("_time_s")]
    a_vals = [tuple(getattr(r, f) for f in non_timing) for r in a]
    b_vals = [tuple(getattr(r, f) for f in non_timing) for r in b]
    assert a_vals == b_vals


# ---------------------------------------------------------------------------
# analysis.py -- aggregation correctness on a small constructed table
# ---------------------------------------------------------------------------


def _row(**overrides) -> ResultRow:
    base = dict(
        experiment="unit", config_id="cfg", seed=42, q_table="q", attr="0", M_size=1,
        k=5, alpha=2, F_star=0.3, delta=0.15, include_query=True, top_n=100,
        adapter_synopsis="csv", adapter_semantic="hnsw", adapter_overlap="inverted_index",
        adapter_unionability="pinned_match", lp_precheck=True,
        n_sem=10, n_ovl=10, n_pair=10,
        feasible=True, infeasibility_cause=None, error=None,
        n_D=20, n_P=10, n_R=5, F_P=0.4, F_R=0.35, delta_R=-0.05, sum_U=12.5,
        n_unionability_computations=10, dinkelbach_iterations_stage1=3,
        retrieval_time_s=0.004, stage1_time_s=0.0001, effective_alpha=2.0,
        ilp_time_s=0.002, lp_precheck_ran=True, lp_feasible=True,
        lp_time_s=0.0005, wall_time_s=0.01,
    )
    base.update(overrides)
    return ResultRow(**base)


def test_infeasibility_by_cause_counts_correctly():
    rows = [
        _row(k=5, alpha=2, feasible=True, infeasibility_cause=None),
        _row(k=5, alpha=2, feasible=False, infeasibility_cause="stage2_infeasible"),
        _row(k=5, alpha=2, feasible=False, infeasibility_cause="stage2_infeasible"),
        _row(k=5, alpha=3, feasible=False, infeasibility_cause="insufficient_candidates"),
    ]
    df = rows_to_dataframe(rows)
    pivot = analysis.infeasibility_by_cause(df, ["k", "alpha"])

    row_5_2 = pivot[(pivot["k"] == 5) & (pivot["alpha"] == 2)].iloc[0]
    assert row_5_2["feasible"] == 1
    assert row_5_2["stage2_infeasible"] == 2
    assert row_5_2["total"] == 3

    row_5_3 = pivot[(pivot["k"] == 5) & (pivot["alpha"] == 3)].iloc[0]
    assert row_5_3["insufficient_candidates"] == 1
    assert row_5_3["total"] == 1


def test_quality_vs_cost_only_averages_quality_over_feasible_rows():
    rows = [
        _row(k=5, alpha=2, feasible=True, sum_U=10.0, F_R=0.4, n_unionability_computations=10),
        _row(k=5, alpha=2, feasible=False, infeasibility_cause="stage2_infeasible",
             sum_U=None, F_R=None, n_unionability_computations=10),
    ]
    df = rows_to_dataframe(rows)
    summary = analysis.quality_vs_cost_by_config(df, ["k", "alpha"])
    r = summary.iloc[0]
    assert r["n_total"] == 2
    assert r["n_feasible"] == 1
    assert r["mean_sum_U"] == 10.0  # NOT averaged with the infeasible row's None
    assert r["mean_n_unionability_computations"] == 10.0


def test_efficiency_invariant_passes_on_valid_data():
    rows = [
        _row(k=5, alpha=2, n_D=20, n_P=10, n_unionability_computations=10,
             infeasibility_cause=None, feasible=True),
        _row(k=5, alpha=10, n_D=20, n_P=20, n_unionability_computations=20,  # clamped: alpha*k=50 > n_D=20
             infeasibility_cause="stage2_infeasible", feasible=False),
        _row(k=5, alpha=2, n_D=3, n_P=0, n_unionability_computations=0,       # |D|<k, never reached Stage 1
             infeasibility_cause="insufficient_candidates", feasible=False),
    ]
    df = rows_to_dataframe(rows)
    violations = analysis.check_efficiency_invariant(df)
    assert len(violations) == 0


def test_efficiency_invariant_catches_a_real_violation():
    rows = [
        _row(k=5, alpha=2, n_D=20, n_P=10, n_unionability_computations=9,  # WRONG: should be 10
             infeasibility_cause=None, feasible=True),
    ]
    df = rows_to_dataframe(rows)
    violations = analysis.check_efficiency_invariant(df)
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Real-data smoke tests -- tiny, skip cleanly if starmie_fair isn't mounted.
# ---------------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    not ctx_mod.santos_data_available(),
    reason="starmie_fair santos data not available at the expected read-only path",
)


@pytestmark_real
def test_real_alpha_sweep_smoke():
    """2 queries x 2 conditions -- confirms the real driver + real adapters
    wire together correctly, without paying for the full 48 x 15 sweep."""
    ctx, adapters = ctx_mod.build_santos_context()
    base_tasks, skipped = ctx_mod.load_santos_base_tasks(
        k=3, alpha=2, F_star=0.3, delta=0.2, top_n=100,
    )
    base_tasks = base_tasks[:2]
    assert len(base_tasks) == 2

    rows, instances = run_alpha_sweep(
        base_tasks, ctx, adapters, alphas=(1, 2), ks=(3,), seed=42, verbose=False,
    )
    assert len(rows) == 4  # 2 queries x 1 k x 2 alphas
    for r in rows:
        assert r.error is None
        assert r.n_D >= 0

    if instances:
        lp_rows = run_lp_precheck_experiment(instances, adapters, seed=42, verbose=False)
        assert len(lp_rows) == 2 * len(instances)


@pytestmark_real
def test_column_coverage_stats_on_real_santos_datalake():
    """Generic instrumentation added for the WDC scalability study.

    Empirically (this test), santos's own categorical_coverage_ratio at
    theta_cat=50 is ~0.82, NOT small -- correcting an assumption made while
    planning the WDC study (that santos's ~5,000-row tables would show a
    much lower ratio than WDC's ~6-row tables). Real open-data tables
    apparently have plenty of genuinely low-cardinality columns (status
    codes, boroughs, categories) regardless of row count, so a high ratio
    on its own isn't evidence of anything broken.

    The actual claim this stat is for is narrower: theta_cat=50 is doing
    SOME real discriminative work on santos (excluding a nonzero fraction
    of columns as too high-cardinality) -- unlike WDC's ~6-row median
    tables, where nunique is upper-bounded by n_rows itself, so the filter
    is structurally unable to exclude anything, regardless of what the
    data looks like. The ratio alone can't distinguish "high because the
    data is genuinely categorical" from "high because the filter can't
    discriminate at this table size" -- RESULTS-wdc.md needs to report
    both the ratio AND the row-count context together, not the ratio
    alone, to make that argument honestly.
    """
    synopsis, datalake_files, _, _ = ctx_mod.load_shared_resources()
    sample = datalake_files[:30]
    stats = ctx_mod.column_coverage_stats(synopsis, sample)
    assert stats["n_tables"] == 30
    assert stats["n_columns_total"] >= stats["n_columns_categorical"] >= 0
    assert 0.0 <= stats["categorical_coverage_ratio"] <= 1.0
    # theta_cat excludes a nonzero fraction of columns on santos -- real
    # discriminative work, not a null filter (contrast with WDC, where it's
    # structurally unable to exclude almost anything at ~6 rows/table).
    assert stats["categorical_coverage_ratio"] < 1.0


@pytestmark_real
def test_real_retrieval_ablation_smoke():
    """2 queries x the 2x2 semantic/overlap grid -- exercises the real
    driver (index building + all 4 adapter combinations) without paying for
    the full 48-query run."""
    rows = run_retrieval_ablation(
        k=3, alpha=2, F_star=0.3, delta=0.2, top_n=50, task_limit=2, verbose=False,
    )
    assert len(rows) == 2 * 2 * 2  # 2 queries x 2 semantic x 2 overlap
    assert all(r.error is None for r in rows)
    config_ids = {r.config_id for r in rows}
    assert config_ids == {
        "semantic=hnsw_overlap=inverted_index", "semantic=hnsw_overlap=null",
        "semantic=exact_scan_overlap=inverted_index", "semantic=exact_scan_overlap=null",
    }


@pytestmark_real
def test_real_stage1_skip_ablation_smoke():
    """2 queries x 2 conditions against real santos data -- confirms
    run_query_skip_stage1 wires correctly against the real adapters (not
    just the stub context) and that the sum_U monotonicity property holds
    on real, non-uniform U values."""
    rows = run_stage1_skip_ablation(
        k=5, alpha=1, F_star=0.3, delta=0.15, top_n=50, task_limit=2, verbose=False,
    )
    assert len(rows) == 4  # 2 queries x 2 conditions
    assert all(r.error is None for r in rows)

    by_query = {}
    for r in rows:
        condition = r.config_id.split("__")[0]
        by_query.setdefault(r.q_table, {})[condition] = r
    for q_table, conditions in by_query.items():
        two, skip = conditions["two_stage"], conditions["skip_stage1"]
        if two.feasible and skip.feasible:
            assert skip.sum_U >= two.sum_U - 1e-9


# ---------------------------------------------------------------------------
# experiments/fraction_reachability.py -- brute-force attainable-F analysis
# ---------------------------------------------------------------------------


def test_reachability_enumerates_every_subset_and_brackets_correctly():
    """A hand-checkable instance: D of 4 tables, k=2, include_query=False.

    N/n pairs (1,10) (2,10) (3,10) (4,10) give per-pair F of
    (1+2)/20=.15, (1+3)/20=.20, (1+4)/20=.25, (2+3)/20=.25, (2+4)/20=.30,
    (3+4)/20=.35 -- 6 = C(4,2) subsets, F_min=.15, F_max=.35, and F*=0.30 is
    both attainable exactly (best_gap 0) and bracketed."""
    from experiments.fraction_reachability import analyse_query

    raw = [("t0", 1, 10), ("t1", 2, 10), ("t2", 3, 10), ("t3", 4, 10)]
    tables = [r[0] for r in raw]
    n_rows = {t: n for (t, _, n) in raw}
    n_rows["Q"] = 10
    dist = {(t, 0): {"m": N, "x": n - N} for (t, N, n) in raw}
    dist[("Q", 0)] = {"m": 0, "x": 10}
    ctx = RunnerContext(
        synopsis=_StubSynopsis(n_rows, dist),
        semantic=_StubSemantic([(t, 0, 1.0 - i * 1e-6) for i, t in enumerate(tables)]),
        overlap=_StubOverlap({(t, 0): 1 for t in tables}),
        unionability=_StubUnionability(value=1.0),
        query_vectors={"Q": np.zeros((1, 4))},
    )
    task = QueryTask(q_table="Q", attr=0, M={"m"}, F_star=0.30, delta=0.15, k=2,
                     alpha=1.0, include_query=False)
    row = analyse_query(task, ctx, "stub")

    assert row.skipped == ""
    assert row.n_subsets == 6                      # exactly C(4,2) -- nothing skipped or double-counted
    assert row.F_min == pytest.approx(0.15)
    assert row.F_max == pytest.approx(0.35)
    assert row.best_gap == pytest.approx(0.0)      # 0.30 is hit exactly
    assert row.F_closest == pytest.approx(0.30)
    assert row.tau_reachable is True               # tau = 0.15 <= F_max
    assert row.fstar_bracketed is True
    assert row.fstar_within_delta is True


def test_reachability_reports_unreachable_target_rather_than_failing():
    """Same shape, but F* far above anything attainable: the row must still be
    produced, with fstar_bracketed False and a real positive best_gap -- this
    is the case the module exists to distinguish from an optimizer failure."""
    from experiments.fraction_reachability import analyse_query

    raw = [("t0", 1, 10), ("t1", 2, 10), ("t2", 3, 10), ("t3", 4, 10)]
    tables = [r[0] for r in raw]
    n_rows = {t: n for (t, _, n) in raw}
    n_rows["Q"] = 10
    dist = {(t, 0): {"m": N, "x": n - N} for (t, N, n) in raw}
    dist[("Q", 0)] = {"m": 0, "x": 10}
    ctx = RunnerContext(
        synopsis=_StubSynopsis(n_rows, dist),
        semantic=_StubSemantic([(t, 0, 1.0 - i * 1e-6) for i, t in enumerate(tables)]),
        overlap=_StubOverlap({(t, 0): 1 for t in tables}),
        unionability=_StubUnionability(value=1.0),
        query_vectors={"Q": np.zeros((1, 4))},
    )
    task = QueryTask(q_table="Q", attr=0, M={"m"}, F_star=0.90, delta=0.05, k=2,
                     alpha=1.0, include_query=False)
    row = analyse_query(task, ctx, "stub")

    assert row.fstar_bracketed is False
    assert row.tau_reachable is False              # tau = 0.85 > F_max = 0.35
    assert row.best_gap == pytest.approx(0.55)     # 0.90 - 0.35
    assert row.fstar_within_delta is False


def test_reachability_skips_when_fewer_candidates_than_k():
    """|D| < k has no k-subset at all -- must be recorded as skipped with a
    reason, not silently dropped or reported as an unreachable target."""
    from experiments.fraction_reachability import analyse_query

    raw = [("t0", 1, 10), ("t1", 2, 10)]
    tables = [r[0] for r in raw]
    n_rows = {t: n for (t, _, n) in raw}
    n_rows["Q"] = 10
    dist = {(t, 0): {"m": N, "x": n - N} for (t, N, n) in raw}
    dist[("Q", 0)] = {"m": 0, "x": 10}
    ctx = RunnerContext(
        synopsis=_StubSynopsis(n_rows, dist),
        semantic=_StubSemantic([(t, 0, 1.0 - i * 1e-6) for i, t in enumerate(tables)]),
        overlap=_StubOverlap({(t, 0): 1 for t in tables}),
        unionability=_StubUnionability(value=1.0),
        query_vectors={"Q": np.zeros((1, 4))},
    )
    task = QueryTask(q_table="Q", attr=0, M={"m"}, F_star=0.3, delta=0.15, k=5,
                     alpha=1.0, include_query=False)
    row = analyse_query(task, ctx, "stub")

    assert row.skipped == "insufficient_candidates"
    assert row.best_gap is None and row.tau_reachable is None
