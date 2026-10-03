"""Experiment configuration: every parameter the entry point knows about, where its default
comes from, which systems need it, and how to collect it (command line first, then an
interactive prompt).

A parameter is in one of three groups:

* ``core``    -- system, dataset and the three artifact paths. Always collected; when missing
                 from the command line the user is prompted (a derived suggestion is shown in
                 brackets). Without a terminal, a missing core parameter is an error.
* ``method``  -- per-system knobs (k, alpha, F*, ...). Prompted with their default shown unless
                 ``--defaults`` is given or there is no terminal, in which case the default is used.
* ``derived`` -- paths that follow from the core ones (protected-attribute CSV, groundtruth,
                 metadata store, Starmie root). Never prompted; derived and reported, and can be
                 overridden on the command line.
"""
import argparse
import os
import sys
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_STARMIE_ROOT = os.environ.get("STARMIE_FAIR_ROOT", "/u6/bkassaie/starmie_fair")

# Starmie's column-vector file names (cl_<side>_<augment>_<sample>_<order>_<run_id>.pkl).
VEC_DATALAKE = "cl_datalake_drop_col_tfidf_entity_column_0.pkl"
VEC_QUERY = "cl_query_drop_col_tfidf_entity_column_0.pkl"
METADATA_FILE = "metadata_combined.pkl"


class System(NamedTuple):
    name: str
    description: str
    starmie_approach: Optional[str]   # None for DUTS; the run_starmie_baselines approach otherwise


SYSTEMS: Dict[str, System] = {s.name: s for s in (
    System("duts", "DUTS: retrieval -> Stage 1 (Dinkelbach) -> unionability on the pool -> Stage 2 (ILP)", None),
    System("starmie", "Starmie top-k (unconstrained bipartite matching, no fairness constraint)", "starmie"),
    System("starmie_exhaustive", "Starmie + exhaustive swap fairification", "exhaustive"),
    System("starmie_nl", "Starmie + NL swap fairification (winnow/dominance pruning)", "nl"),
)}
BASELINES = tuple(n for n, s in SYSTEMS.items() if s.starmie_approach)

# Benchmarks laid out as <starmie_root>/data/<name>/{datalake,query,vectors,indexes}. Any other
# directory with the same layout works too; these only drive the suggested default path.
KNOWN_DATASETS = ("santos3", "tusSmall3", "tusLarge3", "santos", "santos2", "santos4")


class Param(NamedTuple):
    name: str                         # argparse dest; the flag is --name-with-dashes
    help: str
    group: str                        # core | method | derived
    type: Callable[[str], Any] = str
    default: Any = None               # a value, or a callable(cfg_dict) -> value
    systems: Optional[Sequence[str]] = None   # None = every system
    choices: Optional[Sequence[str]] = None
    is_path: bool = False

    @property
    def flag(self) -> str:
        return "--" + self.name.replace("_", "-")

    def applies_to(self, system: Optional[str]) -> bool:
        return self.systems is None or system is None or system in self.systems


def _default_dataset_path(c):
    if not c.get("dataset"):
        return None
    return os.path.join(c["starmie_root"], "data", c["dataset"])


def _default_embedding_path(c):
    return os.path.join(c["dataset_path"], "vectors") if c.get("dataset_path") else None


def _default_index_path(c):
    return os.path.join(REPO_ROOT, "artifacts", c["dataset"], "index") if c.get("dataset") else None


def _default_metadata_path(c):
    """The index directory's own copy if it has one, else the benchmark's prebuilt Starmie
    synopsis (``<dataset_path>/indexes``) if that exists, else the index directory (to be built)."""
    own = os.path.join(c["index_path"], METADATA_FILE)
    if os.path.isfile(own):
        return own
    prebuilt = os.path.join(c["dataset_path"], "indexes", METADATA_FILE)
    return prebuilt if os.path.isfile(prebuilt) else own


def _default_protected_csv(c):
    inside = os.path.join(c["dataset_path"], "protected_attributes.csv")
    if os.path.isfile(inside):
        return inside
    return os.path.join(os.path.dirname(os.path.normpath(c["dataset_path"])),
                        "protected_attributes_{}.csv".format(c["dataset"]))


def _default_groundtruth(c):
    for name in ("{}_benchmark_groundtruth.csv", "{}_small_benchmark_groundtruth.csv", "groundtruth.csv"):
        path = os.path.join(c["dataset_path"], name.format(c["dataset"]))
        if os.path.isfile(path):
            return path
    return os.path.join(c["dataset_path"], "{}_benchmark_groundtruth.csv".format(c["dataset"]))


def _default_output_dir(c):
    return os.path.join(REPO_ROOT, "experiments", "results", "main")


PARAMS: List[Param] = [
    # -- core ------------------------------------------------------------------------------
    Param("system", "system/method to run", "core", choices=tuple(SYSTEMS)),
    Param("dataset", "dataset (benchmark) name, e.g. " + ", ".join(KNOWN_DATASETS), "core"),
    Param("dataset_path", "dataset directory holding datalake/ and query/", "core",
          default=_default_dataset_path, is_path=True),
    Param("index_path", "index directory (DUTS HNSW index; metadata store if built here)", "core",
          default=_default_index_path, is_path=True),
    Param("embedding_path", "embedding directory holding the Starmie column-vector pickles", "core",
          default=_default_embedding_path, is_path=True),
    # -- method ----------------------------------------------------------------------------
    Param("k", "result size k", "method", int, 10),
    Param("f_star", "target proportion F*", "method", float, 0.4),
    Param("delta", "max deviation delta (tau = F* - delta)", "method", float, 0.1),
    Param("sigma", "column-similarity threshold sigma", "method", float, 0.6),
    Param("alpha", "pool expansion factor alpha (pool = alpha*k)", "method", float, 5.0, systems=("duts",)),
    Param("top_n", "ANN probe width top_n (sec. 7.1)", "method", int, 1000, systems=("duts",)),
    Param("theta_cat", "categorical domain-size threshold theta_cat", "method", int, 50, systems=("duts",)),
    Param("n_columns", "nearest columns retrieved per query column (Starmie's N)", "method", int, 1000,
          systems=BASELINES),
    Param("workers", "worker processes for the baseline (1 = in-process)", "method", int, 1,
          systems=BASELINES),
    # -- derived ---------------------------------------------------------------------------
    Param("starmie_root", "Starmie checkout (TableMetadata, HNSWSearcher_Fair, sdd/)", "derived",
          default=DEFAULT_STARMIE_ROOT, is_path=True),
    Param("metadata_path", "value-distribution synopsis (Starmie MetadataStore pickle)", "derived",
          default=_default_metadata_path, is_path=True),
    Param("protected_csv", "query list: q_name, protected_attribute_id, protected_value", "derived",
          default=_default_protected_csv, is_path=True),
    Param("groundtruth_csv", "table-union groundtruth (query_table, data_lake_table)", "derived",
          default=_default_groundtruth, is_path=True),
    Param("output_dir", "where per-query CSV and summary JSON are written", "derived",
          default=_default_output_dir, is_path=True),
]
PARAMS_BY_NAME = {p.name: p for p in PARAMS}


class ConfigError(Exception):
    """A parameter is missing or invalid; the message says which and how to fix it."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="Run DUTS or a Starmie baseline on one dataset, checking (and offering to build) "
                    "every prerequisite first. Missing parameters are prompted for interactively.",
        epilog="systems: " + "; ".join("{} = {}".format(s.name, s.description) for s in SYSTEMS.values()),
    )
    for p in PARAMS:
        kw = dict(dest=p.name, default=None,
                  help=p.help + (" [{}]".format(", ".join(p.systems)) if p.systems else ""))
        if p.choices:
            kw["choices"] = p.choices
        if p.type is not str:
            kw["type"] = p.type
        ap.add_argument(p.flag, **kw)
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="run only the first N queries (smoke test)")
    ap.add_argument("--defaults", action="store_true",
                    help="use defaults for method parameters instead of prompting")
    ap.add_argument("--build-missing", action="store_true",
                    help="build missing prerequisites without asking for confirmation")
    ap.add_argument("--non-interactive", action="store_true",
                    help="never prompt; fail on missing required parameters")
    # Prerequisite-construction parameters (collected only when something must be built).
    ap.add_argument("--hnsw-m", type=int, default=None, help="HNSW graph degree M for the DUTS index")
    ap.add_argument("--hnsw-ef-construction", type=int, default=None, help="HNSW ef_construction")
    ap.add_argument("--checkpoint", default=None,
                    help="trained Starmie model checkpoint (.pt) used to generate embeddings")
    ap.add_argument("--embed-batch-size", type=int, default=None, help="tables per inference batch")
    return ap


class Prompter(object):
    """Reads answers from the terminal. ``interactive=False`` never reads: ``ask`` returns the
    default, or raises ``ConfigError`` when there is none."""

    def __init__(self, interactive: bool, stream_in=None, stream_out=None):
        self.interactive = interactive
        self._in = stream_in or sys.stdin
        self._out = stream_out or sys.stdout

    def ask(self, label: str, default: Any = None, cast: Callable[[str], Any] = str,
            choices: Optional[Sequence[str]] = None, flag: Optional[str] = None) -> Any:
        if not self.interactive:
            if default is None:
                raise ConfigError("missing required parameter {}{}".format(
                    label, " (pass {})".format(flag) if flag else ""))
            return default
        hint = " ({})".format("/".join(choices)) if choices else ""
        suffix = " [{}]".format(default) if default is not None else ""
        while True:
            self._out.write("  {}{}{}: ".format(label, hint, suffix))
            self._out.flush()
            line = self._in.readline()
            if line == "":
                raise ConfigError("input closed while asking for {}".format(label))
            raw = line.strip()
            if not raw:
                if default is not None:
                    return default
                self._out.write("    a value is required\n")
                continue
            if choices and raw not in choices:
                self._out.write("    choose one of: {}\n".format(", ".join(choices)))
                continue
            try:
                return cast(raw)
            except (TypeError, ValueError):
                self._out.write("    not a valid {}\n".format(getattr(cast, "__name__", "value")))

    def confirm(self, question: str, default: bool = True) -> bool:
        if not self.interactive:
            return False
        suffix = " [Y/n] " if default else " [y/N] "
        self._out.write(question + suffix)
        self._out.flush()
        raw = self._in.readline().strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes")


def _resolve_default(p: Param, values: Dict[str, Any]) -> Any:
    if callable(p.default):
        try:
            return p.default(values)
        except (KeyError, TypeError):
            return None
    return p.default


def collect(args: argparse.Namespace, prompter: Prompter, use_defaults: bool) -> Dict[str, Any]:
    """Merge command-line values, prompts and derived defaults into one flat dict. Order
    matters: derived defaults (e.g. ``embedding_path``) read earlier values."""
    values: Dict[str, Any] = {}
    # starmie_root first: dataset_path's suggestion depends on it.
    values["starmie_root"] = args.starmie_root or DEFAULT_STARMIE_ROOT
    for p in PARAMS:
        if p.name == "starmie_root":
            continue
        given = getattr(args, p.name)
        if given is not None:
            values[p.name] = given
            continue
        if not p.applies_to(values.get("system")):
            continue
        default = _resolve_default(p, values)
        if p.group == "core":
            values[p.name] = prompter.ask(p.help, default, p.type, p.choices, p.flag)
        elif p.group == "method" and not use_defaults:
            values[p.name] = prompter.ask(p.help, default, p.type, p.choices, p.flag)
        else:
            values[p.name] = default
    for p in PARAMS:
        if p.is_path and values.get(p.name):
            values[p.name] = os.path.abspath(os.path.expanduser(values[p.name]))
    values["limit_queries"] = args.limit_queries
    return values


def validate(values: Dict[str, Any]) -> List[str]:
    """Value-level checks (paths are checked by ``prereqs``). Returns error messages."""
    errors = []
    system = values.get("system")
    if system not in SYSTEMS:
        errors.append("--system must be one of {}".format(", ".join(SYSTEMS)))
    if not values.get("dataset"):
        errors.append("--dataset is required")
    k = values.get("k")
    if k is None or k < 1:
        errors.append("--k must be >= 1")
    f_star, delta = values.get("f_star"), values.get("delta")
    if f_star is not None and not 0.0 <= f_star <= 1.0:
        errors.append("--f-star must be in [0, 1]")
    if delta is not None and delta < 0.0:
        errors.append("--delta must be >= 0")
    if f_star is not None and delta is not None and not 0.0 <= f_star - delta <= 1.0:
        errors.append("tau = F* - delta = {:.3f} must be in [0, 1]".format(f_star - delta))
    sigma = values.get("sigma")
    if sigma is not None and not -1.0 <= sigma < 1.0:
        errors.append("--sigma must be in [-1, 1)")
    if system == "duts":
        if values.get("alpha") is None or values["alpha"] < 1.0:
            errors.append("--alpha must be >= 1")
        if values.get("top_n") is None or values["top_n"] < 1:
            errors.append("--top-n must be >= 1")
        if values.get("theta_cat") is None or values["theta_cat"] < 1:
            errors.append("--theta-cat must be >= 1")
    if system in BASELINES:
        if values.get("n_columns") is None or values["n_columns"] < 1:
            errors.append("--n-columns must be >= 1")
        if values.get("workers") is None or values["workers"] < 1:
            errors.append("--workers must be >= 1")
    if values.get("limit_queries") is not None and values["limit_queries"] < 1:
        errors.append("--limit-queries must be >= 1")
    return errors


def rerun_command(values: Dict[str, Any]) -> str:
    """The exact command that reproduces this configuration non-interactively."""
    parts = ["python", "main.py"]
    for p in PARAMS:
        v = values.get(p.name)
        if v is None or not p.applies_to(values.get("system")):
            continue
        parts += [p.flag, _quote(str(v))]
    if values.get("limit_queries"):
        parts += ["--limit-queries", str(values["limit_queries"])]
    return " ".join(parts)


def _quote(s: str) -> str:
    return "'{}'".format(s) if any(ch in s for ch in " \t'\"$") else s
