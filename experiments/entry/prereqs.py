"""Prerequisite detection and construction for the entry point.

Every artifact a run needs is one ``Requirement``. ``check`` inspects them all without side
effects and returns the ones that are not satisfied. Unsatisfied requirements are either
*fatal* (input data the program cannot create: the dataset tables, query list, groundtruth,
the Starmie checkout) or *buildable* (derived artifacts: column embeddings, the
value-distribution synopsis, the DUTS HNSW index). ``build`` constructs the buildable ones in
dependency order -- embeddings and synopsis before the HNSW index, which is built from both.

The builders write only to the paths the user configured (embedding path, metadata path, index
path); nothing is ever written under the Starmie checkout itself unless a configured path points
there.
"""
import os
import pickle
import sys
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import numpy as np

from .config import BUNDLED_STARMIE, REPO_ROOT, VEC_DATALAKE, VEC_QUERY, Prompter

CHECKPOINT_NAMES = ("starmie_santos_model_drop_col_tfidf_entity_column_0.pt",   # name in the Zenodo record
                    "model_drop_col_tfidf_entity_column_0.pt")


class Requirement(NamedTuple):
    key: str
    label: str
    ok: bool
    detail: str             # what was checked / what is wrong
    fatal: bool             # True = the program cannot create it; the user must supply it
    how_to_fix: str = ""


# -- path helpers --------------------------------------------------------------------------

def datalake_dir(v: Dict[str, Any]) -> str:
    return os.path.join(v["dataset_path"], "datalake")


def query_dir(v: Dict[str, Any]) -> str:
    return os.path.join(v["dataset_path"], "query")


def datalake_vec_pkl(v: Dict[str, Any]) -> str:
    return os.path.join(v["embedding_path"], VEC_DATALAKE)


def query_vec_pkl(v: Dict[str, Any]) -> str:
    return os.path.join(v["embedding_path"], VEC_QUERY)


def hnsw_key(v: Dict[str, Any]) -> str:
    return "{}_duts".format(v["dataset"])


def _n_csv(d: str) -> int:
    try:
        return sum(1 for f in os.listdir(d) if f.endswith(".csv"))
    except OSError:
        return 0


# -- checks --------------------------------------------------------------------------------

def check(v: Dict[str, Any], hnsw_m: int = 32) -> List[Requirement]:
    """Every requirement for ``v['system']``, satisfied or not, in the order they are reported."""
    system = v["system"]
    reqs: List[Requirement] = []
    starmie_files = ["TableMetadata.py"]
    if system != "duts":
        starmie_files += ["HNSWSearcher_Fair.py", "bounds.py", "utility.py", "exhaustive_swap.py",
                          "nl_swap.py", "preference.py", "Custom_Heap.py"]
    missing = [f for f in starmie_files if not os.path.isfile(os.path.join(BUNDLED_STARMIE, f))]
    reqs.append(Requirement(
        "starmie_bundle", "bundled Starmie modules", not missing,
        BUNDLED_STARMIE if not missing else "{} lacks {}".format(BUNDLED_STARMIE, ", ".join(missing)),
        True, "the repository checkout is incomplete; re-clone it"))

    for key, label, d in (("datalake", "datalake tables", datalake_dir(v)),
                          ("query", "query tables", query_dir(v))):
        n = _n_csv(d)
        reqs.append(Requirement(key, label, n > 0,
                                "{} ({} CSV files)".format(d, n) if n else "{}: no CSV files".format(d),
                                True, "--dataset-path must contain datalake/ and query/ with CSV tables"))
    for key, label, flag in (("protected_csv", "query list (protected attributes)", "--protected-csv"),
                             ("groundtruth_csv", "groundtruth", "--groundtruth-csv")):
        path = v[key]
        reqs.append(Requirement(key, label, os.path.isfile(path), path, True,
                                "pass {} with the file's location".format(flag)))
    if os.path.isfile(v["protected_csv"]):
        bad = _protected_csv_problem(v["protected_csv"])
        if bad:
            reqs[-2] = reqs[-2]._replace(ok=False, detail="{}: {}".format(v["protected_csv"], bad))

    have_vecs = os.path.isfile(datalake_vec_pkl(v)) and os.path.isfile(query_vec_pkl(v))
    reqs.append(Requirement(
        "embeddings", "column embeddings", have_vecs,
        "{} + {}".format(datalake_vec_pkl(v), VEC_QUERY) if have_vecs else
        "missing in {} (need {} and {})".format(v["embedding_path"], VEC_DATALAKE, VEC_QUERY), False))

    have_meta = os.path.isfile(v["metadata_path"])
    reqs.append(Requirement(
        "metadata", "value-distribution synopsis (MetadataStore)", have_meta,
        v["metadata_path"] if have_meta else "missing: {}".format(v["metadata_path"]), False))

    if system == "duts":
        index_ok, detail = _duts_index_status(v, hnsw_m)
        reqs.append(Requirement("duts_index", "DUTS HNSW index", index_ok, detail, False))
    return reqs


def _protected_csv_problem(path: str) -> Optional[str]:
    import csv
    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    need = {"q_name", "protected_attribute_id", "protected_value"}
    lacking = need - set(h.strip() for h in header)
    return "missing column(s) {}".format(", ".join(sorted(lacking))) if lacking else None


def _duts_index_status(v: Dict[str, Any], hnsw_m: int):
    """Existence only. Whether the index still matches its inputs is verified when it is loaded
    (``systems.run_duts`` raises ``StaleIndexError``), so the synopsis is not loaded twice."""
    from experiments.semantic_cache import cache_paths
    bin_path, meta_path = cache_paths(hnsw_key(v), hnsw_m, v["sigma"], v["theta_cat"],
                                      v["index_path"], create=False)
    if os.path.isfile(bin_path) and os.path.isfile(meta_path):
        return True, bin_path
    return False, "missing: {}".format(bin_path)


def find_index_m(v: Dict[str, Any], requested: Optional[int]) -> int:
    """The HNSW degree M to use: ``requested`` if given, else that of the single index already
    in ``index_path`` for this (dataset, sigma, theta_cat), else 32."""
    if requested:
        return requested
    from experiments.semantic_cache import cache_paths
    found = []
    for m in range(2, 129):
        bin_path, _ = cache_paths(hnsw_key(v), m, v["sigma"], v["theta_cat"], v["index_path"], create=False)
        if os.path.isfile(bin_path):
            found.append(m)
    if 32 in found or not found:
        return 32
    return found[0]


class StaleIndexError(Exception):
    """The DUTS HNSW index exists but was built from different inputs."""


# -- shared loaders (also used by systems.py) ----------------------------------------------

def load_synopsis(v: Dict[str, Any]):
    from dutsx import registry
    return registry.build("synopsis", "metadata_store", pkl_path=v["metadata_path"],
                          theta_cat=v.get("theta_cat"), starmie_fair_root=BUNDLED_STARMIE)


def indexed_tables(v: Dict[str, Any], synopsis, dl_vecs) -> List[str]:
    """Datalake tables that have both an embedding and a synopsis entry, sorted."""
    return [f for f in sorted(os.listdir(datalake_dir(v)))
            if f in dl_vecs and synopsis.has_table(f)]


def semantic_vectors(v: Dict[str, Any], synopsis, dl_vecs):
    from experiments.context import categorical_only_vectors
    return categorical_only_vectors(dl_vecs, synopsis, indexed_tables(v, synopsis, dl_vecs))


# -- builders ------------------------------------------------------------------------------

class BuildParams(NamedTuple):
    hnsw_m: int
    hnsw_ef_construction: int
    checkpoint: Optional[str]
    embed_batch_size: int
    starmie_root: Optional[str] = None     # public Starmie checkout providing sdd/


def collect_build_params(missing: List[Requirement], v: Dict[str, Any], args, prompter: Prompter) -> BuildParams:
    """Ask only for what the missing artifacts need to be built."""
    keys = {r.key for r in missing}
    m = args.hnsw_m or 32
    efc = args.hnsw_ef_construction or 200
    if "duts_index" in keys and prompter.interactive:
        print("\nThe DUTS HNSW index needs:")
        if args.hnsw_m is None:
            m = prompter.ask("HNSW graph degree M", 32, int, flag="--hnsw-m")
        if args.hnsw_ef_construction is None:
            efc = prompter.ask("HNSW ef_construction", 200, int, flag="--hnsw-ef-construction")
    checkpoint, batch = args.checkpoint, args.embed_batch_size or 1024
    starmie_root = v.get("starmie_root")
    if "embeddings" in keys:
        if prompter.interactive:
            print("\nGenerating embeddings needs the public Starmie code (its sdd/ package, from "
                  "https://github.com/megagonlabs/starmie) and a trained Starmie model checkpoint:")
        if not (starmie_root and os.path.isdir(os.path.join(starmie_root, "sdd"))):
            starmie_root = prompter.ask("public Starmie checkout (contains sdd/)", starmie_root,
                                        flag="--starmie-root")
            starmie_root = os.path.abspath(os.path.expanduser(starmie_root))
            if not os.path.isdir(os.path.join(starmie_root, "sdd")):
                raise FileNotFoundError("no sdd/ package in {} (git clone "
                                        "https://github.com/megagonlabs/starmie)".format(starmie_root))
        guesses = [os.path.join(d, n) for d in (REPO_ROOT, v["data_root"]) for n in CHECKPOINT_NAMES]
        guesses.append(os.path.join(starmie_root, "results", "santos", CHECKPOINT_NAMES[1]))
        guess = next((g for g in guesses if os.path.isfile(g)), None)
        if checkpoint is None:
            checkpoint = prompter.ask("checkpoint (.pt)", guess, flag="--checkpoint")
        checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError("checkpoint not found: {}".format(checkpoint))
        if args.embed_batch_size is None:
            batch = prompter.ask("tables per inference batch", 1024, int, flag="--embed-batch-size")
    return BuildParams(m, efc, checkpoint, batch, starmie_root)


def build(missing: List[Requirement], v: Dict[str, Any], bp: BuildParams,
          log: Callable[[str], None] = print) -> List[str]:
    """Build every buildable missing requirement in dependency order. Returns what was written."""
    keys = {r.key for r in missing}
    written: List[str] = []
    if "embeddings" in keys:
        written += generate_embeddings(datalake_dir(v), query_dir(v), v["embedding_path"],
                                       bp.checkpoint, bp.starmie_root, bp.embed_batch_size, log)
    if "metadata" in keys:
        written.append(build_metadata(datalake_dir(v), query_dir(v), v["metadata_path"],
                                      BUNDLED_STARMIE, log))
    if "duts_index" in keys:
        written += build_duts_index(v, bp.hnsw_m, bp.hnsw_ef_construction, log)
    return written


def generate_embeddings(datalake: str, query: str, out_dir: str, checkpoint: str,
                        starmie_root: str, batch_size: int, log=print) -> List[str]:
    """Starmie column embeddings for every datalake and query table.

    Mirrors ``starmie/extractVectors.py`` (first 1000 rows per table, ``lineterminator='\\n'``,
    ``inference_on_tables``) and ``sdd.pretrain.load_checkpoint`` -- except that the tokenizing
    dataset is built over THIS dataset's datalake rather than the checkpoint's hard-coded
    ``data/<task>/datalake``, so no file under the Starmie checkout has to be staged or rewritten.
    """
    import pandas as pd
    if starmie_root not in sys.path:
        sys.path.insert(0, starmie_root)
    import torch
    from sdd.dataset import PretrainTableDataset
    from sdd.model import BarlowTwinsSimCLR
    from sdd.pretrain import inference_on_tables

    t0 = time.time()
    ckpt = torch.load(checkpoint, map_location=torch.device("cpu"))
    hp = ckpt["hp"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("loading checkpoint {} on {}".format(checkpoint, device))
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm).to(device)
    state = ckpt["model"]
    # Checkpoints saved under transformers < 4.31 carry the non-persistent ``position_ids`` buffer
    # (always arange(max_positions)); newer transformers rejects it as an unexpected key.
    expected = set(model.state_dict())
    for key in [k for k in state if k.endswith("embeddings.position_ids") and k not in expected]:
        del state[key]
    model.load_state_dict(state)
    dataset = PretrainTableDataset.from_hp(datalake, hp)

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for side, folder, name in (("datalake", datalake, VEC_DATALAKE), ("query", query, VEC_QUERY)):
        dfs = {}
        for fn in sorted(f for f in os.listdir(folder) if f.endswith(".csv")):
            try:
                df = pd.read_csv(os.path.join(folder, fn), lineterminator="\n")
            except Exception as e:  # noqa: BLE001 -- same skip-and-report policy as extractVectors.get_df
                log("  skipping {}: {}".format(fn, e))
                continue
            if df.empty or len(df.columns) == 0:
                log("  skipping empty {}".format(fn))
                continue
            dfs[fn] = df.head(1000)
        log("embedding {} {} tables ...".format(len(dfs), side))
        feats = inference_on_tables(list(dfs.values()), model, dataset, batch_size=batch_size)
        out = os.path.join(out_dir, name)
        with open(out, "wb") as f:
            pickle.dump([(fn, np.array(x)) for fn, x in zip(dfs, feats)], f)
        written.append(out)
        log("wrote {}".format(out))
    log("embeddings done in {:.0f}s".format(time.time() - t0))
    return written


def build_metadata(datalake: str, query: str, out_path: str, starmie_root: str, log=print) -> str:
    """One combined ``MetadataStore`` (datalake + query tables, full value histograms), the
    synopsis ``HNSWSearcher_Fair`` loads as ``metadata_combined.pkl``. Per-file failures are
    tolerated as in ``scripts/wdc_build_metadata_store.py``."""
    if starmie_root not in sys.path:
        sys.path.insert(0, starmie_root)
    from TableMetadata import MetadataStore
    from scripts.wdc_build_metadata_store import _process_csv_file_tolerant

    t0 = time.time()
    store = MetadataStore()
    failed = []
    for folder in (datalake, query):
        files = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
        log("reading {} tables from {} ...".format(len(files), folder))
        for fn in files:
            path = os.path.join(folder, fn)
            try:
                store._process_csv_file(path, fn)  # noqa: SLF001 -- per file, so one bad file can't abort
            except Exception:  # noqa: BLE001
                try:
                    _process_csv_file_tolerant(store, path, fn)
                except Exception as e:  # noqa: BLE001
                    failed.append((fn, str(e)))
    store._update_global_stats()  # noqa: SLF001
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    store.save(out_path)
    log("wrote {} ({} tables, {} unreadable) in {:.0f}s".format(
        out_path, len(store._metadata), len(failed), time.time() - t0))  # noqa: SLF001
    for fn, err in failed[:10]:
        log("  unreadable: {}: {}".format(fn, err))
    return out_path


def build_duts_index(v: Dict[str, Any], m: int, ef_construction: int, log=print) -> List[str]:
    from dutsx.adapters.unionability import load_starmie_vectors
    from experiments.semantic_cache import cache_paths, get_or_build

    t0 = time.time()
    synopsis = load_synopsis(v)
    dl_vecs = load_starmie_vectors(datalake_vec_pkl(v))
    sem = semantic_vectors(v, synopsis, dl_vecs)
    bin_path, meta_path = cache_paths(hnsw_key(v), m, v["sigma"], v["theta_cat"], v["index_path"])
    for stale in (bin_path, meta_path):   # a stale index is rebuilt, never silently reused
        if os.path.isfile(stale):
            os.remove(stale)
    log("building HNSW (M={}, ef_construction={}) over {} tables ...".format(m, ef_construction, len(sem)))
    get_or_build(hnsw_key(v), sem, v["sigma"], v["theta_cat"], m=m, verbose=False,
                 cache_dir=v["index_path"], ef_construction=ef_construction)
    log("wrote {} in {:.0f}s".format(bin_path, time.time() - t0))
    return [bin_path, meta_path]
