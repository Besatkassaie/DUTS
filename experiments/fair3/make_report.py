"""Aggregate experiments/results/fair3_eval_*.csv into md tables and beamer frames.
Usage: python -m experiments.fair3.make_report  -> writes experiments/results/fair3_tables.md and fair3_frames.tex"""
import glob
import os

import pandas as pd

R = "/u6/bkassaie/DUTS/experiments/results"
BENCH = ["santos3", "tusSmall3", "tusLarge3"]
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "fair3_eval_*_n1000.csv")))])  # top_n=1000 only
df["prec_R"] = pd.to_numeric(df.prec_R)
_own = {}
for _b in BENCH:
    _m = pd.read_csv("/u6/bkassaie/starmie_fair/data/%s/%s_query_mapping.csv" % (_b, _b))
    _own.update({(_b, q): int(n) for q, n in zip(_m.fair_query_table, _m.n_own_copy)})
df["n_own"] = [_own[(b, q)] for b, q in zip(df.bench, df.q_table)]


def cell(g):
    n = len(g)
    feas = g[g.feasible]
    elig = g[g.n_own >= g.k]
    m = lambda s: s.dropna().mean() if s.notna().any() else float("nan")
    z = lambda s: s.fillna(0.0).mean()
    return dict(
        n=n, feas=len(feas), elig=len(elig), elig_feas=int(elig.feasible.sum()), D=g.n_D.mean(), na=int((g.stage1 == "na").sum()),
        clamp=int((g.stage1 == "clamped").sum()), s1ran=int((g.stage1 == "ran").sum()),
        dec=int((g.stage2 == "lp_decisive").sum()), milp=int(g.milp_ran.sum()),
        prec_R=z(g.prec_R), rec_R=z(g.rec_R), ideal=g.ideal_recall.mean(),   # all queries, failed = 0
        sumU=z(g.sum_U), F_R=z(g.F_R),                                        # all queries, failed = 0
        unf_prec=m(feas.unf_prec), unf_rec=m(feas.unf_rec), unf_U=m(feas.unf_sum_U), unf_F=m(feas.unf_F),
        prec_D=z(g.prec_D), rec_D=z(g.rec_D), prec_P=z(g.prec_P), rec_P=z(g.rec_P),   # all queries; empty/missing set = 0
    )


rows = {}
for (b, n, a, k), g in df.groupby(["bench", "top_n", "alpha", "k"]):
    rows[(n, b, a, k)] = cell(g)


def fmt(x, d=3):
    return "--" if x != x else ("{:." + str(d) + "f}").format(x)


TABLES = {
    "quality": (["bench", "α", "k", "feasible", "precision@R", "recall@R", "ideal recall", "mean ΣU", "mean F_R"],
                lambda r: ["{}/{}".format(r["feas"], r["n"]), fmt(r["prec_R"]), fmt(r["rec_R"], 4), fmt(r["ideal"], 4),
                           fmt(r["sumU"], 1), fmt(r["F_R"])]),
    "stages": (["bench", "α", "k", "mean |D|", "|D|<k (S1+S2 bypassed)", "S1 bypassed (P=D)", "S1 ran",
                "LP decisive (MILP bypassed)", "MILP ran", "feasible", "queries with ≥k own copies", "feasible among them"],
               lambda r: [fmt(r["D"], 1), str(r["na"]), str(r["clamp"]), str(r["s1ran"]), str(r["dec"]),
                          str(r["milp"]), "{}/{}".format(r["feas"], r["n"]), str(r["elig"]),
                          "{}/{}".format(r["elig_feas"], r["elig"])]),
    "dpr": (["bench", "α", "k", "prec@D", "rec@D", "prec@P", "rec@P", "prec@R", "rec@R"],
            lambda r: [fmt(r["prec_D"]), fmt(r["rec_D"], 4), fmt(r["prec_P"]), fmt(r["rec_P"], 4), fmt(r["prec_R"]),
                       fmt(r["rec_R"], 4)]),
}
TITLES = {"quality": "Quality of the fair result R (all queries; a failed query counts 0 for precision, recall, ΣU and F_R)",
          "stages": "Stage bypasses, LP pre-check and feasibility (counts of queries)",
          "dpr": "Precision and recall at D, P and R (all queries; an empty or missing set counts 0)"}

md, tex = [], []
for n in (1000,):
    for key in ("quality", "stages", "dpr"):
        head, fn = TABLES[key]
        md.append("\n**top_n = {} -- {}**\n".format(n, TITLES[key]))
        md.append("| " + " | ".join(head) + " |")
        md.append("|" + "|".join(["---"] * 3 + ["---:"] * (len(head) - 3)) + "|")
        tex_rows = []
        for b in BENCH:
            for a in (10.0, 5.0):
                for k in (5, 10, 20):
                    r = rows[(n, b, a, k)]
                    vals = fn(r)
                    md.append("| " + " | ".join([b, str(int(a)), str(k)] + vals) + " |")
                    tex_rows.append(" & ".join(["\\texttt{%s}" % b.replace("_", "\\_"), str(int(a)), str(k)]
                                               + [v.replace("--", "--") for v in vals]))
        cols = "lrr" + "r" * (len(head) - 3)
        th = " & ".join(h.replace("α", "$\\alpha$").replace("Σ", "$\\Sigma$").replace("|D|", "$|D|$")
                        .replace("<", "$<$").replace("@", "@").replace("_R", "$_R$").replace("→", "$\\to$")
                        .replace("(S1+S2 bypassed)", "(bypass S1,S2)") for h in head)
        tex.append((n, key, cols, th, tex_rows))

open(os.path.join(R, "fair3_tables.md"), "w").write("\n".join(md) + "\n")
with open(os.path.join(R, "fair3_frames.tex"), "w") as f:
    for n, key, cols, th, trs in tex:
        f.write("%% ---- top_n=%d %s\n" % (n, key))
        f.write("\\begin{frame}{Fairified benchmarks: %s}{top\\_n$=%d$, $F^*{=}0.4$, $\\delta{=}0.1$, "
                "$\\alpha\\in\\{10,5\\}\\times k\\in\\{5,10,20\\}$}\n" % (TITLES[key].split(" (")[0].replace("&", "\\&"), n))
        f.write("\\tiny\n\\begin{center}\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n" % (cols, th))
        f.write(" \\\\\n".join(trs) + " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{center}\n@@NOTE_%s_%d@@\n\\end{frame}\n\n" % (key, n))
print("ok", len(md), "md lines")
