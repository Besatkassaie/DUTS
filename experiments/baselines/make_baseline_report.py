"""Aggregate experiments/results/baselines_<bench>_<approach>.csv into per-approach md tables (written to
experiments/results/baselines_tables.md) and beamer table frames (baselines_frames.tex).
Means over FEASIBLE queries for precision / recall / ideal recall / sum U / F_R (the same convention as the DUTS tables);
'(all)' columns average over every query with a non-empty result."""
import json
import os

import pandas as pd

R = "/u6/bkassaie/DUTS/experiments/results"
BENCH = ["santos3", "tusSmall3", "tusLarge3"]
APPR = {"starmie": "Pure Starmie (no fairification)", "exhaustive": "Exhaustive swap (no filtering)",
        "nl": "Nested-loop swap (winnow filtering)"}


def load(a):
    d = pd.concat([pd.read_csv(os.path.join(R, "baselines_%s_%s.csv" % (b, a))) for b in BENCH])
    ex = d.extras.map(lambda s: json.loads(s) if isinstance(s, str) else {})
    for key in ("needs_fairification", "success", "swaps_made", "initial_fairness", "dominated_removed_count",
                "dominance_candidate_pool_size", "candidates_after_winnow"):
        d[key] = [e.get(key) for e in ex]
    return d


def f(x, dg=3):
    return "--" if x != x or x is None else ("{:." + str(dg) + "f}").format(x)


def main_rows(d, a):
    """Fair methods (exhaustive, nl): precision, recall, sum U and F_R over ALL queries, an infeasible query counts 0. Pure Starmie: everything over ALL queries as returned, regardless of tau (reached-tau count shown)."""
    out = []
    for b in BENCH:
        for k in (5, 10, 20):
            g = d[(d.bench == b) & (d.k == k)]
            fe = g[g.feasible]
            if a == "starmie":
                p, r, u, fv = g.prec.fillna(0).mean(), g.rec.mean(), g.duts_sum_U.mean(), g.F.mean()
            else:
                p, r = g.prec.where(g.feasible, 0).fillna(0).mean(), g.rec.where(g.feasible, 0).mean()
                u, fv = g.duts_sum_U.where(g.feasible, 0).mean(), g.F.where(g.feasible, 0).mean()
            out.append([b, str(k), "%d/%d" % (len(fe), len(g)), f(p), f(r, 4), f(g.ideal_recall.mean(), 4), f(u, 1), f(fv)])
    head = ["bench", "k", "reached τ" if a == "starmie" else "feasible", "precision", "recall", "ideal recall",
            "mean ΣU", "mean F" if a == "starmie" else "mean F_R"]
    return head, out


def detail_rows(d, a):
    out = []
    for b in BENCH:
        for k in (5, 10, 20):
            g = d[(d.bench == b) & (d.k == k)]
            n = len(g)
            row = [b, str(k), f(g.n_candidates.mean(), 1), "%d" % (g.n_returned < k).sum()]
            if a != "starmie":
                need = g[g.needs_fairification == True]  # noqa: E712
                ok = need[need.success == True]  # noqa: E712
                row += ["%d/%d" % (len(need), n), "%d/%d" % (len(ok), len(need)), f(need.swaps_made.mean(), 1),
                        f(need.initial_fairness.mean())]
                if a == "nl":
                    pool = g.dominance_candidate_pool_size.sum()
                    rem = g.dominated_removed_count.sum()
                    row += ["%d/%d" % (rem, pool), f(rem / pool if pool else float("nan")),
                            f(g.candidates_after_winnow.mean(), 1)]
            out.append(row)
    if a == "starmie":
        head = ["bench", "k", "mean candidates scored", "queries with < k returned"]
    elif a == "exhaustive":
        head = ["bench", "k", "mean candidates scored", "queries with < k returned", "needed fairification",
                "swap succeeded (of needed)", "mean swaps (of needed)", "mean F before swap (of needed)"]
    else:
        head = ["bench", "k", "mean candidates scored", "queries with < k returned", "needed fairification",
                "swap succeeded (of needed)", "mean swaps (of needed)", "mean F before swap (of needed)",
                "dominated removed / pool", "removed share", "mean candidates after winnow"]
    return head, out


md, tex = [], {}
for a, title in APPR.items():
    d = load(a)
    for kind, (head, rows) in (("main", main_rows(d, a)), ("detail", detail_rows(d, a))):
        md.append("\n**%s -- %s**\n" % (title, "results" if kind == "main" else "algorithm details"))
        md.append("| " + " | ".join(head) + " |")
        md.append("|" + "|".join(["---"] * 2 + ["---:"] * (len(head) - 2)) + "|")
        for r in rows:
            md.append("| " + " | ".join(r) + " |")
        tex[(a, kind)] = (head, rows)
open(os.path.join(R, "baselines_tables.md"), "w").write("\n".join(md) + "\n")

TH = {"α": "$\\alpha$", "Σ": "$\\Sigma$", "τ": "$\\tau$", "<": "$<$", "F_R": "$F_R$", "F (all)": "$F$ (all)", "|D|": "$|D|$"}


def th(h):
    for a_, b_ in TH.items():
        h = h.replace(a_, b_)
    return h.replace("F before swap", "$F$ before swap").replace("mean F", "mean $F$") if "$F" not in h else h


with open(os.path.join(R, "baselines_frames.tex"), "w") as fh:
    for (a, kind), (head, rows) in tex.items():
        cols = "l" + "r" * (len(head) - 1)
        sub = "$N{=}1000$, $\\sigma{=}0.6$, $F^*{=}0.4$, $\\delta{=}0.1$" if kind == "main" else "algorithm-specific measures, $N{=}1000$, $\\sigma{=}0.6$"
        fh.write("%% ---- %s %s\n\\begin{frame}{%s: %s}{%s}\n\\scriptsize\n\\begin{center}\n\\setlength{\\tabcolsep}{2.5pt}\n"
                 "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%%\n\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n"
                 % (a, kind, APPR[a], "results" if kind == "main" else "algorithm details", sub, cols,
                    " & ".join(th(h) for h in head)))
        fh.write(" \\\\\n".join(" & ".join(["\\texttt{%s}" % r[0]] + r[1:]) for r in rows))
        fh.write(" \\\\\n\\bottomrule\n\\end{tabular}}\n\\end{center}\n@@NOTE_%s_%s@@\n\\end{frame}\n\n" % (a, kind))
print("ok")
