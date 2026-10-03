"""Tests for the release entry point (``main.py`` -> ``experiments/entry``).

* Configuration: interactive prompting (answers fed through a stream), non-interactive failure on a
  missing required parameter, value validation, and that the printed rerun command parses back to
  the same configuration.
* Prerequisites and the whole flow, on a synthetic dataset built in ``tmp_path``: input data that
  cannot be generated is reported as fatal; missing derived artifacts (synopsis, HNSW index) are
  built on request and the run then succeeds and reports results. Embeddings are written directly
  (random, with planted neighbours) -- generating them needs a trained Starmie checkpoint, which is
  exercised by hand, not here. Needs starmie_fair's ``TableMetadata`` (for the synopsis), so these
  skip when that checkout is absent, like the other ``test_dutsx_*`` real-data tests.
"""
import csv
import io
import json
import os
import pickle
import shlex

import numpy as np
import pytest

from experiments.entry import config, prereqs, report
from experiments.entry.app import EXIT_BUILT, EXIT_CONFIG, EXIT_MISSING, EXIT_OK, ExperimentApp
from experiments.entry.systems import QueryOutcome, RunOutput

STARMIE_OK = os.path.isfile(os.path.join(config.DEFAULT_STARMIE_ROOT, "TableMetadata.py"))
needs_starmie = pytest.mark.skipif(not STARMIE_OK, reason="starmie_fair checkout not available")


# -- configuration -------------------------------------------------------------------------

def _collect(argv, answers=None):
    args = config.build_parser().parse_args(argv)
    prompter = config.Prompter(answers is not None, stream_in=io.StringIO(answers or ""),
                               stream_out=io.StringIO())
    return config.collect(args, prompter, use_defaults=answers is None)


def test_interactive_prompts_fill_missing_core_and_method_params(tmp_path):
    # system (bad choice, then valid), dataset, three paths accepted/overridden, then method params
    answers = "\n".join([
        "nope", "duts",                 # invalid choice is re-asked
        "mydata",
        str(tmp_path / "data"),         # dataset_path
        "",                             # index_path: accept suggestion
        "",                             # embedding_path: accept suggestion
        "7", "", "", "", "x", "3", "", "",   # k=7, defaults..., alpha: bad float re-asked then 3
    ]) + "\n"
    v = _collect(["--starmie-root", str(tmp_path)], answers)
    assert v["system"] == "duts" and v["dataset"] == "mydata"
    assert v["dataset_path"] == str(tmp_path / "data")
    assert v["embedding_path"] == str(tmp_path / "data" / "vectors")
    assert v["index_path"].endswith(os.path.join("artifacts", "mydata", "index"))
    assert v["k"] == 7 and v["alpha"] == 3.0 and v["f_star"] == 0.4
    assert "n_columns" not in v       # baseline-only parameter, never asked for duts


def test_non_interactive_missing_core_param_is_an_error():
    with pytest.raises(config.ConfigError, match="--system"):
        _collect(["--dataset", "x"])


def test_method_defaults_used_without_prompting(tmp_path):
    v = _collect(["--system", "starmie_nl", "--dataset", "d", "--dataset-path", str(tmp_path),
                  "--index-path", str(tmp_path / "i"), "--embedding-path", str(tmp_path / "e")])
    assert v["n_columns"] == 1000 and v["workers"] == 1 and "alpha" not in v
    assert config.validate(v) == []


@pytest.mark.parametrize("override, message", [
    ({"alpha": 0.5}, "--alpha"),
    ({"f_star": 0.05, "delta": 0.1}, "tau"),
    ({"k": 0}, "--k"),
    ({"sigma": 1.0}, "--sigma"),
])
def test_validate_rejects_bad_values(tmp_path, override, message):
    v = _collect(["--system", "duts", "--dataset", "d", "--dataset-path", str(tmp_path),
                  "--index-path", str(tmp_path / "i"), "--embedding-path", str(tmp_path / "e")])
    v.update(override)
    assert any(message in e for e in config.validate(v))


def test_rerun_command_round_trips(tmp_path):
    argv = ["--system", "duts", "--dataset", "d", "--dataset-path", str(tmp_path / "a b"),
            "--index-path", str(tmp_path / "i"), "--embedding-path", str(tmp_path / "e"),
            "--k", "3", "--alpha", "2", "--limit-queries", "5"]
    v = _collect(argv)
    again = _collect(shlex.split(config.rerun_command(v))[2:])
    assert again == v


def test_summary_counts_infeasible_queries_as_zero():
    def q(name, feasible, prec, rec, n):
        return QueryOutcome(name, feasible, n, prec, rec, 0.5, 4, None, None, 0.1, None, "")
    out = RunOutput([q("a", True, 1.0, 0.5, 2), q("b", False, 0.5, 0.25, 2), q("c", False, None, None, 0)],
                    n_skipped=1, setup_s=1.0, details={})
    s = report.summarize(out, total_s=2.0)
    assert s["n_feasible"] == 1 and abs(s["pct_feasible"] - 100 / 3) < 1e-9
    assert abs(s["precision"] - 1 / 3) < 1e-12 and abs(s["recall"] - 0.5 / 3) < 1e-12
    assert s["precision_as_returned"] == 0.75 and s["n_returned_any"] == 2
    assert abs(s["avg_query_s"] - 0.1) < 1e-12


def test_environment_report_has_required_fields():
    env = report.collect_environment()
    for key in ("python", "platform", "packages", "milp_solver", "java", "cuda_torch", "gpus"):
        assert key in env
    assert env["packages"]["scipy"] and env["packages"]["numpy"]


# -- prerequisites and the full flow on a synthetic dataset --------------------------------

def _make_dataset(root, n_tables=12, dim=16, seed=0):
    """Query ``q.csv`` with protected column 0 (value ``a``). Datalake tables t0..t{n-1}: the even
    ones are unionable (groundtruth) and get an embedding close to the query's column 0; all have a
    categorical column 0 over {a, b} with varying share of ``a``."""
    rng = np.random.RandomState(seed)
    (root / "datalake").mkdir(parents=True)
    (root / "query").mkdir()
    base = rng.randn(dim).astype(np.float32)

    def write(path, n_a, n_b):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["grp", "num"])
            for i in range(n_a + n_b):
                w.writerow(["a" if i < n_a else "b", i])

    write(root / "query" / "q.csv", 3, 7)
    q_vecs = [("q.csv", np.stack([base, rng.randn(dim)]).astype(np.float32))]
    dl_vecs, gt = [], []
    for i in range(n_tables):
        name = "t{}.csv".format(i)
        write(root / "datalake" / name, 2 + i % 5, 6)
        col0 = base + (0.05 if i % 2 == 0 else 3.0) * rng.randn(dim)
        dl_vecs.append((name, np.stack([col0, rng.randn(dim)]).astype(np.float32)))
        if i % 2 == 0:
            gt.append(name)
    (root / "vectors").mkdir()
    with open(root / "vectors" / config.VEC_DATALAKE, "wb") as f:
        pickle.dump(dl_vecs, f)
    with open(root / "vectors" / config.VEC_QUERY, "wb") as f:
        pickle.dump(q_vecs, f)
    with open(root / "protected_attributes.csv", "w", newline="") as f:
        f.write("q_name,protected_attribute_id,protected_value\nq.csv,0,a\nmissing.csv,0,a\n")
    with open(root / "syn_benchmark_groundtruth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_table", "data_lake_table"])
        for t in gt:
            w.writerow(["q.csv", t])


def _argv(root, *extra):
    return ["--system", "duts", "--dataset", "syn", "--dataset-path", str(root),
            "--index-path", str(root / "index"), "--embedding-path", str(root / "vectors"),
            "--output-dir", str(root / "out"), "--k", "3", "--alpha", "2", "--top-n", "50",
            "--sigma", "0.5", "--defaults", "--non-interactive"] + list(extra)


@needs_starmie
def test_fatal_inputs_are_reported_not_built(tmp_path, capsys):
    _make_dataset(tmp_path)
    os.remove(tmp_path / "syn_benchmark_groundtruth.csv")
    assert ExperimentApp(_argv(tmp_path, "--build-missing")).run() == EXIT_MISSING
    out = capsys.readouterr().out
    assert "cannot be generated" in out and "groundtruth" in out
    assert not (tmp_path / "index").exists()       # nothing built when an input is missing


@needs_starmie
def test_missing_artifacts_need_confirmation_then_build_then_run(tmp_path, capsys):
    _make_dataset(tmp_path)
    # 1. non-interactive without --build-missing: reported, nothing built
    assert ExperimentApp(_argv(tmp_path)).run() == EXIT_MISSING
    assert not os.path.exists(tmp_path / "index" / config.METADATA_FILE)

    # 2. interactive: build params asked, confirmation given -> synopsis + index built, rerun asked
    answers = io.StringIO("16\n100\ny\n")
    app = ExperimentApp([a for a in _argv(tmp_path) if a != "--non-interactive"], stdin=answers,
                        interactive=True)
    assert app.run() == EXIT_BUILT
    out = capsys.readouterr().out
    assert "Rerun the experiment with" in out and "--hnsw-m 16" in out
    v = app.values
    assert os.path.isfile(v["metadata_path"])
    assert prereqs.find_index_m(v, None) == 16

    # 3. rerun: everything present -> results and environment reported, outputs written
    assert ExperimentApp(_argv(tmp_path)).run() == EXIT_OK
    out = capsys.readouterr().out
    assert "RESULTS" in out and "EXECUTION ENVIRONMENT" in out and "avg per query" in out
    with open(tmp_path / "out" / "syn_duts_k3_a2_summary.json") as f:
        summary = json.load(f)["summary"]
    assert summary["n_queries"] == 1 and summary["n_skipped"] == 1
    assert summary["n_feasible"] == 1 and summary["precision"] == 1.0


@needs_starmie
def test_stale_index_is_detected_and_rebuilt(tmp_path, capsys):
    _make_dataset(tmp_path)
    assert ExperimentApp(_argv(tmp_path, "--build-missing")).run() == EXIT_BUILT
    # add a datalake table (and its embedding + synopsis entry): the index no longer matches
    with open(tmp_path / "vectors" / config.VEC_DATALAKE, "rb") as f:
        vecs = pickle.load(f)
    vecs.append(("extra.csv", np.ones((2, 16), dtype=np.float32)))
    with open(tmp_path / "vectors" / config.VEC_DATALAKE, "wb") as f:
        pickle.dump(vecs, f)
    with open(tmp_path / "datalake" / "extra.csv", "w") as f:
        f.write("grp,num\na,1\nb,2\n")
    os.remove(tmp_path / "index" / config.METADATA_FILE)
    assert ExperimentApp(_argv(tmp_path, "--build-missing")).run() == EXIT_BUILT   # synopsis rebuilt
    capsys.readouterr()
    assert ExperimentApp(_argv(tmp_path, "--build-missing")).run() == EXIT_BUILT   # stale index rebuilt
    assert "built from different" in capsys.readouterr().out
    assert ExperimentApp(_argv(tmp_path)).run() == EXIT_OK


def test_unknown_system_is_a_config_error(tmp_path):
    with pytest.raises(SystemExit):
        ExperimentApp(["--system", "nope"])
    app = ExperimentApp(["--dataset", "x", "--non-interactive"])
    assert app.run() == EXIT_CONFIG
