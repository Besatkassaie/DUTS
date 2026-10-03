"""Are the Starmie-side failures (k=10) a candidate limit, or did the swap miss an existing solution?
Per query, on ONE searcher build: candidates = the N=1000-nearest-column tables (what all three approaches see);
F_max = best F over sets of EXACTLY k tables (unaligned candidates pad when fewer than k align) drawn from the candidates whose protected column aligns (Starmie's own
F rule: a table counts only if aligned), computed exactly by Dinkelbach at each size; then pure Starmie, exhaustive swap and
nested-loop swap are run on the same searcher and checked with the same feasibility rule as the tables.
Usage: python -m experiments.baselines.diag_candidate_limit <bench> [workers]"""
import contextlib, csv, io, multiprocessing as mp, os, sys
from experiments.baselines import run_starmie_baselines as R
from duts import stage1_dinkelbach
from duts.stats import F as Fr, aggregate
from duts.types import CandidateStats

K, TAU = 10, R.F_STAR - R.DELTA


def _diag(args):
    from bounds import verify_constrained
    from utility import fairness_delta_exceeds
    q_name, p_id, value = args
    s, q = R._S["searcher"], R._S["queries"][q_name]
    with contextlib.redirect_stdout(io.StringIO()):
        cands = s._find_candidates(list(q[1]), R.N)
    cs = []
    for t in cands:
        _, mc = verify_constrained(q[1], t[1], R.SIGMA, p_id)
        col = next((c for r_, c in mc if r_ == p_id), None)
        meta = s.metadata_store.get_metadata(t[0])
        if col is None or meta is None:
            continue
        cname = meta.column_names[col]
        if not meta.is_categorical(cname):
            continue
        cs.append(CandidateStats(table=t[0], N=meta.get_category_count(cname, value), n=meta.num_records, U=0.0))
    qm = s.metadata_store.get_metadata(q_name)
    N_Q, n_Q = qm.get_category_count(qm.column_names[p_id], value), qm.num_records
    # exactly k tables: if fewer than k candidates align, the rest are unaligned candidates that add no rows
    if len(cands) < K:
        fmax = 0.0
    elif len(cs) >= K:
        sel = stage1_dinkelbach.solve(cs, K, N_Q, n_Q, True).selected
        fmax = Fr(*aggregate(sel, N_Q, n_Q, True))
    else:
        fmax = max(Fr(*aggregate(stage1_dinkelbach.solve(cs, m, N_Q, n_Q, True).selected, N_Q, n_Q, True))
                   for m in range(1, len(cs) + 1)) if cs else Fr(0, 1) if False else (N_Q / n_Q if n_Q else 0.0)
    out = dict(q_table=q_name, n_cand=len(cands), n_aligned=len(cs), F_max=fmax)
    for name, alg in (("starmie", None), ("exhaustive", "exhustive_swap"), ("nl", "nl_swap")):
        with contextlib.redirect_stdout(io.StringIO()):
            if alg is None:
                res, _ = s.topk("cl", q, K, N=R.N, threshold=R.SIGMA)
            else:
                qr, _ = s.topk_fairified("cl", q, K=K, N=R.N, threshold=R.SIGMA, p_id=p_id, protected_value=value, algorithm=alg)
                res = qr["sorted_results"] if isinstance(qr, dict) else qr
            names = [r[1] for r in res]
            mmap = {t: verify_constrained(q[1], R._S["tvec"][t], R.SIGMA, p_id)[1] for t in names}
            d_, fres = s.compute_Delta(names, q[0], p_id, value, include_query=True, matched_columns_map=mmap, target_fairness=R.F_STAR)
        out[name] = bool(len(names) == K and fres and not fairness_delta_exceeds(d_, R.DELTA))
    return out


if __name__ == "__main__":
    bench = sys.argv[1]; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    import pickle
    from experiments.context import paths_for
    have = {x[0] for x in pickle.load(open(paths_for(bench).query_vec_pkl, "rb"))}
    qs = []
    with open(os.path.join(R.STARMIE, "data", "protected_attributes_%s.csv" % bench), newline="") as f:
        for r in csv.DictReader(f):
            if r["q_name"].strip() in have:
                qs.append((r["q_name"].strip(), int(r["protected_attribute_id"]), r["protected_value"].strip()))
    with mp.get_context("fork").Pool(workers, initializer=R._init, initargs=(bench, "exhaustive")) as pool:
        rows = pool.map(_diag, qs, chunksize=1)
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(R.RESULTS, "baselines_candidate_limit_%s.csv" % bench), index=False)
    print("[%s] queries %d" % (bench, len(d)), flush=True)
    for m in ("starmie", "exhaustive", "nl"):
        inf = d[~d[m]]
        cand_lim = int((inf.F_max < TAU - 1e-9).sum()); missed = int((inf.F_max >= TAU - 1e-9).sum())
        print("[%s] %-10s k=10 infeasible %3d: no solution among candidates %3d | solution existed but not found %3d"
              % (bench, m, len(inf), cand_lim, missed), flush=True)
