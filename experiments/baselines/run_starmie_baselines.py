"""Starmie-side baselines on the three fairified benchmarks, evaluated with the SAME metric code as the DUTS tables.

Approaches (all from /u6/bkassaie/starmie_fair, imported read-only):
  starmie    : HNSWSearcher_Fair.topk  -- N nearest columns per query column -> unconstrained bipartite matching -> top-K
  exhaustive : topk_fairified(algorithm='exhustive_swap')  -- constrained matching, swaps, no candidate filtering
  nl         : topk_fairified(algorithm='nl_swap')         -- same, candidates first pruned by the winnow (dominance) filter
Fixed: N=1000 (nearest columns per query column), sigma=0.6, F*=0.4, delta=0.1 (tau=0.3), K in {5,10,20}.
Per query and K one CSV row: returned tables, F (Starmie's own compute_F on the constrained alignment, query included),
feasible (K tables returned and F>=tau), precision/recall/ideal recall vs the fair ground truth, Starmie's own matching
score and the DUTS pinned-match U of the returned tables, plus algorithm-specific details (no timings).
Usage: python -m experiments.baselines.run_starmie_baselines <benchmark> <approach> [--limit n] [--workers w]
"""
import argparse
import contextlib
import csv
import io
import json
import multiprocessing as mp
import os
import sys
from typing import NamedTuple, Sequence

import numpy as np

STARMIE = "/u6/bkassaie/starmie_fair"
sys.path.insert(0, STARMIE)
RESULTS = "/u6/bkassaie/DUTS/experiments/results"
KS = (5, 10, 20)
N, SIGMA, F_STAR, DELTA = 1000, 0.6, 0.4, 0.1

_S = {}


class Settings(NamedTuple):
    """Everything ``_init``/``_one`` need. ``settings_for`` gives the values this script has always
    used; ``experiments/entry`` (``main.py``) builds one from user-supplied paths instead."""
    bench: str
    approach: str
    starmie_root: str
    datalake_vec_pkl: str
    query_vec_pkl: str
    query_dir: str
    datalake_dir: str
    metadata_dir: str          # must contain metadata_combined.pkl
    groundtruth_csv: str
    scratch_dir: str           # HNSWSearcher_Fair writes its index here (one subdirectory per worker)
    ks: Sequence[int] = KS
    n: int = N
    sigma: float = SIGMA
    f_star: float = F_STAR
    delta: float = DELTA


def settings_for(bench: str, approach: str) -> Settings:
    from experiments.context import paths_for
    from experiments.fair3.eval_fair3 import gt_path
    p = paths_for(bench)
    root = os.path.dirname(p.query_dir)
    return Settings(bench=bench, approach=approach, starmie_root=STARMIE,
                    datalake_vec_pkl=p.datalake_vec_pkl, query_vec_pkl=p.query_vec_pkl,
                    query_dir=p.query_dir, datalake_dir=p.datalake_dir,
                    metadata_dir=os.path.join(root, "indexes"), groundtruth_csv=gt_path(bench, p),
                    scratch_dir=os.path.join(RESULTS, "_baseline_scratch"))


def _patch_starmie():
    import utility
    import TableMetadata
    from experiments.fair3.eval_fair3 import expand_M
    # never load the raw datalake into RAM (only needed to BUILD metadata, which we load from disk)
    orig_read = utility.Utility.read_csv_files_to_dict
    utility.Utility.read_csv_files_to_dict = staticmethod(lambda folder: {})
    # never rewrite the benchmark's metadata pickle
    TableMetadata.MetadataStore.save = lambda self, path: None
    # numeric spelling variants of the protected value ("4" <-> "4.0"), as M in the DUTS runs
    orig_cnt = TableMetadata.TableMetadata.get_category_count

    def cnt(self, col, value):
        return sum(orig_cnt(self, col, v) for v in sorted(expand_M(str(value))[0]))
    TableMetadata.TableMetadata.get_category_count = cnt
    return orig_read


def _init(st: Settings):
    if st.starmie_root not in sys.path:
        sys.path.insert(0, st.starmie_root)
    os.chdir(st.starmie_root)
    _patch_starmie()
    from HNSWSearcher_Fair import HNSWSearcher_Fair
    from dutsx import registry
    from dutsx.adapters.unionability import load_starmie_vectors
    from experiments.groundtruth import load_groundtruth
    import pickle
    scratch = os.path.join(st.scratch_dir, "%s_%d" % (st.bench, os.getpid()))
    os.makedirs(scratch, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        s = HNSWSearcher_Fair(
            st.datalake_vec_pkl, os.path.join(scratch, "hnsw.bin"), st.query_dir, st.datalake_dir, 1.0,
            random_seed=42, load_metadata=True, metadata_dir=st.metadata_dir,
            delta=st.delta, target_fairness=st.f_star)
    q_vecs = load_starmie_vectors(st.query_vec_pkl)
    dl_vecs = load_starmie_vectors(st.datalake_vec_pkl)
    _S.update(
        searcher=s, approach=st.approach, bench=st.bench, truth=load_groundtruth(st.groundtruth_csv),
        tvec={t[0]: t[1] for t in s.tables},
        queries={q[0]: q for q in pickle.load(open(st.query_vec_pkl, "rb"))},
        scorer=registry.build("unionability", "pinned_match", query_vectors=q_vecs, candidate_vectors=dl_vecs,
                              threshold=st.sigma),
        ks=tuple(st.ks), n=st.n, sigma=st.sigma, f_star=st.f_star, delta=st.delta,
    )


def _one(args):
    from bounds import verify_constrained
    from experiments.groundtruth import precision_recall
    from utility import fairness_delta_exceeds
    q_name, p_id, value = args
    s, approach, truth = _S["searcher"], _S["approach"], _S["truth"]
    N, SIGMA, F_STAR, DELTA = _S["n"], _S["sigma"], _S["f_star"], _S["delta"]
    q = _S["queries"][q_name]
    gt = truth.get(q_name, frozenset())
    rows = []
    for K in _S["ks"]:
        extras = {}
        with contextlib.redirect_stdout(io.StringIO()):
            if approach == "starmie":
                res, n_cand = s.topk("cl", q, K, N=N, threshold=SIGMA)
            else:
                alg = "exhustive_swap" if approach == "exhaustive" else "nl_swap"
                qres, n_cand = s.topk_fairified("cl", q, K=K, N=N, threshold=SIGMA, p_id=p_id,
                                                protected_value=value, algorithm=alg)
                res = qres["sorted_results"] if isinstance(qres, dict) else qres
                if isinstance(qres, dict):
                    for key in ("needs_fairification", "success", "swaps_made", "initial_fairness", "initial_delta",
                                "dominated_removed_count", "dominance_candidate_pool_size", "candidates_after_winnow"):
                        if key in qres:
                            extras[key] = qres[key]
            names = [r[1] for r in res]
            mmap = {t: verify_constrained(q[1], _S["tvec"][t], SIGMA, p_id)[1] for t in names}
            delta_, fres = s.compute_Delta(names, q[0], p_id, value, include_query=True, matched_columns_map=mmap,
                                           target_fairness=F_STAR)
        F = fres["protected_proportion"] if fres else None
        feasible = len(names) == K and F is not None and not fairness_delta_exceeds(delta_, DELTA)
        pr, rc, hits = precision_recall(names, gt)
        # DUTS pinned-match U of the returned tables, pinned on the column aligned with the protected column
        # (fallback: the table column most similar to the query's protected column)
        qv = np.asarray(q[1][p_id], dtype=np.float32)
        u_sum = 0.0
        for t in names:
            col = next((c for r_, c in mmap[t] if r_ == p_id), None)
            if col is None:
                tv = np.asarray(_S["tvec"][t], dtype=np.float32)
                col = int(np.argmax(tv @ qv / (np.linalg.norm(tv, axis=1) * np.linalg.norm(qv) + 1e-12)))
            u_sum += _S["scorer"].score(q_name, t, (p_id, col))
        rows.append(dict(
            bench=_S["bench"], approach=approach, q_table=q_name, k=K, gt_size=len(gt), n_candidates=n_cand,
            n_returned=len(names), feasible=bool(feasible), F=F, prec=pr, rec=rc,
            ideal_recall=(min(K, len(gt)) / len(gt)) if gt else 0.0, starmie_sum_score=sum(r[0] for r in res),
            duts_sum_U=u_sum, tables=";".join(names), extras=json.dumps(extras)))
    return rows


def run(bench, approach, limit=None, workers=1):
    import csv as _csv
    import resource
    import time
    t0 = time.time()
    prot = os.path.join(STARMIE, "data", "protected_attributes_%s.csv" % bench)
    qs = []
    with open(prot, newline="") as f:
        for r in _csv.DictReader(f):
            qs.append((r["q_name"].strip(), int(r["protected_attribute_id"]), r["protected_value"].strip()))
    import pickle
    from experiments.context import paths_for
    have = {q[0] for q in pickle.load(open(paths_for(bench).query_vec_pkl, "rb"))}
    qs = [q for q in qs if q[0] in have]
    if limit:
        qs = qs[:limit]
    print("[%s/%s] %d queries, %d workers" % (bench, approach, len(qs), workers), flush=True)
    rows = []
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(settings_for(bench, approach),)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, qs), 1):
            rows.extend(r)
            if i % 10 == 0 or i == len(qs):
                print("  [%s/%s] %d/%d queries" % (bench, approach, i, len(qs)), flush=True)
    out = os.path.join(RESULTS, "baselines_%s_%s%s.csv" % (bench, approach, "_pilot" if limit else ""))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[%s/%s] wrote %s (%d rows); elapsed %.0fs; peak child rss %.1fGB" % (bench, approach, out, len(rows), time.time() - t0,
          resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("approach", choices=["starmie", "exhaustive", "nl"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    run(a.bench, a.approach, a.limit, a.workers)
