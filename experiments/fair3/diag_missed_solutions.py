"""For every infeasible query (alpha=5,k=10 by default): is a feasible k-subset of D even available?
F_max^k = exact max over k-subsets of D of F (Dinkelbach at cardinality k over ALL of D, query included).
  |D|<k            -> retrieval returned too few tables
  F_max^k <  tau   -> no solution exists in D (retrieval limit)
  F_max^k >= tau   -> a solution existed in D but DUTS did not return it (pool/Stage-1 induced)"""
import csv, os, sys
import pandas as pd
from duts import stage1_dinkelbach, stage2_ilp
from duts.types import CandidateStats
from duts.stats import F as Fr, aggregate, drop_empty
from dutsx.runner import load_queries_from_csv, retrieve_unscored_candidates
from .eval_fair3 import F_STAR, DELTA, build_ctx, expand_M, RESULTS

bench, alpha, k = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
TAU = F_STAR - DELTA
ctx, p = build_ctx(bench)
tasks, _ = load_queries_from_csv(p.protected_csv, p.query_dir, k=k, alpha=alpha, F_star=F_STAR, delta=DELTA, top_n=1000)
rows = []
for t in tasks:
    (v,) = t.M
    task = t._replace(M=expand_M(v)[0], top_n=1000)
    D, N_Q, n_Q, _ = retrieve_unscored_candidates(task, ctx)
    D = drop_empty(D)
    if len(D) < k:
        rows.append((bench, t.q_table, len(D), None, "D<k"))
        continue
    pool = int(alpha * k)
    P_un = D if pool >= len(D) else stage1_dinkelbach.solve(D, pool, N_Q, n_Q, True).selected
    P = [CandidateStats(table=c.table, N=c.N, n=c.n, U=0.0) for c in P_un]   # feasibility does not depend on U
    if stage2_ilp.solve(P, k, TAU, N_Q, n_Q, True).feasible:
        continue                                                            # DUTS found a solution on THIS D
    best = stage1_dinkelbach.solve(D, k, N_Q, n_Q, True).selected
    fmax = Fr(*aggregate(best, N_Q, n_Q, True))
    rows.append((bench, t.q_table, len(D), fmax, "no_solution_in_D" if fmax < TAU - 1e-9 else "MISSED"))
out = os.path.join(RESULTS, "fair3_missed_%s_a%d_k%d.csv" % (bench, int(alpha), k))
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["bench", "q_table", "n_D", "F_max_k", "class"]); w.writerows(rows)
d = pd.DataFrame(rows, columns=["bench", "q", "n_D", "fmax", "cls"])
print("[%s a=%g k=%d] infeasible %d: %s" % (bench, alpha, k, len(d), d.cls.value_counts().to_dict()), flush=True)
