"""Why are some queries infeasible? For each query: how many of ITS OWN fair copies (ratio>=0.4 for its value)
are in the retrieved candidate set D, versus how many exist in the datalake."""
import csv, os, sys
from duts.stats import drop_empty
from dutsx.runner import load_queries_from_csv, retrieve_unscored_candidates
from .eval_fair3 import F_STAR, DELTA, build_ctx, expand_M, RESULTS

bench = sys.argv[1]
ctx, p = build_ctx(bench)
tasks, _ = load_queries_from_csv(p.protected_csv, p.query_dir, k=5, alpha=10.0, F_star=F_STAR, delta=DELTA, top_n=100)
out = []
for top_n in (1000,):
    for base in tasks:
        (v,) = base.M
        task = base._replace(M=expand_M(v)[0], top_n=top_n)
        D, N_Q, n_Q, tel = retrieve_unscored_candidates(task, ctx)
        D = drop_empty(D)
        stem = task.q_table[:-4]
        own = [c for c in D if c.table.endswith("__" + stem + ".csv")]
        n_own_total = sum(1 for t in os.listdir(p.datalake_dir) if t.endswith("__" + stem + ".csv"))
        out.append((bench, top_n, task.q_table, len(D), len(own), n_own_total,
                    sum(c.N for c in own) / max(1, sum(c.n for c in own)),
                    sum(c.N for c in D) / max(1, sum(c.n for c in D))))
with open(os.path.join(RESULTS, "fair3_diag_own_copies_{}.csv".format(bench)), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["bench", "top_n", "q_table", "n_D", "own_in_D", "own_total", "ratio_own_in_D", "ratio_D"])
    w.writerows(out)
print("done", len(out))
