"""Build the LaTeX table + md section for the Stage-1 benefit experiment."""
import os
import pandas as pd

REPO = "/u6/bkassaie/DUTS"
RES = os.path.join(REPO, "experiments/results")
DATASETS = [("santosLarge", "Santos-Large"), ("tier_10k", "WDC-10K"),
            ("tier_100k", "WDC-100K"), ("tier_1m", "WDC-1M")]


def stats():
    out = []
    for key, label in DATASETS:
        d = pd.read_csv(os.path.join(RES, "stage1_benefit_{}.csv".format(key)))
        m = d.mean(numeric_only=True)
        d["sv"] = (d.skip_total_s - d.two_total_s) / d.skip_total_s * 100
        out.append(dict(
            label=label, n=len(d), n_D=m.n_D, scored=m.two_n_scored,
            s1=m.two_stage1_s * 1000, sc=m.two_scoring_s * 1000, st2=m.two_stage2_s * 1000,
            tot=m.two_total_s * 1000, ksc=m.skip_scoring_s * 1000,
            kst2=m.skip_stage2_s * 1000, ktot=m.skip_total_s * 1000,
            saved=(m.skip_total_s - m.two_total_s) / m.skip_total_s * 100,
            saved_med=d.sv.median(), x=m.skip_total_s / m.two_total_s,
            feas_two=int(d.two_feasible.sum()), feas_skip=int(d.skip_feasible.sum()),
            u_two=m.two_sum_U, u_skip=m.skip_sum_U))
    return out


TABLE = r"""\begin{tabular}{lrr rrrr rrr rr}
\toprule
& & & \multicolumn{4}{c}{Two-stage (scores $\alpha k{=}30$)} & \multicolumn{3}{c}{Stage 1 skipped (scores $|D|$)} & \multicolumn{2}{c}{Saving} \\
\cmidrule(lr){4-7}\cmidrule(lr){8-10}\cmidrule(lr){11-12}
Dataset & $n$ & $|D|$ & S1 & score & S2 & \textbf{total} & score & S2 & \textbf{total} & \% & $\times$ \\
\midrule
__BODY__\bottomrule
\end{tabular}"""

ROW = ("%s & %d & %d & %.2f & %.2f & %.2f & \\textbf{%.1f} & %.1f & %.1f & "
       "\\textbf{%.1f} & \\textbf{%.1f} & %.1f \\\\\n")


FRAMES = r"""
\begin{frame}{Benefit of Stage 1: post-retrieval pipeline time}{top\_n$=30000$, $k{=}10$, $\alpha{=}3$ (pool $\alpha k{=}30$) -- mean ms per query, timed from just before Stage 1 to the end of the pipeline}
\footnotesize
\begin{center}
\setlength{\tabcolsep}{3pt}
__TBL__
\end{center}
\vspace{3pt}
\footnotesize
Same retrieved $D$ in both conditions, so they differ only in whether Stage 1 runs. \highlight{Two-stage} = Stage 1 (Dinkelbach) $\to$ score $U$ on the 30-table pool $\to$ Stage 2 over the pool. \highlight{Stage 1 skipped} = score $U$ on all of $D$ $\to$ Stage 2 over $D$ at cardinality $k$. Saving $=(T_{\text{skip}}-T_{\text{two}})/T_{\text{skip}}$.

\vspace{1pt}
{\footnotesize\color{Muted} Retrieval, the overlap intersection and the $N_i/n_i$ lookup are outside the timed window (identical in both). $F^\star/\delta$: $0.122/0.07$ on Santos-Large, $0.20/0.10$ on the WDC tiers. HNSW indexes are cached to disk, so $|D|$ is reproducible across runs. One WDC-1M query has $|D|{=}2<k$ and is excluded. Details: \texttt{stage1-benefit-report.md}.}
\end{frame}

\begin{frame}{What the Stage 1 table shows}{In plain terms}
\footnotesize
\begin{itemize}
  \item \highlight{The bigger the data lake, the more Stage 1 saves}: __SAVED_LIST__ (__XLO__$\times$ to __XHI__$\times$). Stage 1 caps unionability computations at $\alpha k = 30$ no matter how many candidates retrieval returns, so the larger $|D|$ grows, the more work it removes.
  \item \highlight{Almost all of the saving is unionability scoring} -- __SC_SL__\,ms $\to$ __SC_SL2__\,ms on Santos-Large, __SC_1M__\,ms $\to$ __SC_1M2__\,ms on WDC-1M. That is 30 bipartite matchings instead of $|D|$ of them: the $\alpha k$-not-$|\mathcal{T}|$ claim, measured.
  \item \highlight{At WDC-1M, skipping Stage 1 also makes Stage 2 expensive}: its ILP runs over $\sim$__ND_1M__ candidates instead of 30, costing __ST_1M__\,ms against __ST_1M2__\,ms. Both the scoring and the solving inflate, not just the scoring.
  \item \highlight{Stage 1 is nearly free}: __S1LO__--__S1HI__\,ms, against savings of hundreds of ms.
  \item \highlight{Feasibility is unaffected} -- __FEAS__, identical in both conditions. The distribution-aware pre-filter never cost a feasible answer here.
  \item \highlight{But the speedup is not free in quality}: the pool reaches __ULO__--__UHI__\% of the $\Sigma U$ that scoring every candidate achieves. Stage 1 can exclude a high-$U$ table that Stage 2 would otherwise have chosen.
\end{itemize}

\vspace{1pt}
{\footnotesize\color{Muted} Absolute ms vary with machine load between runs (the same cached $|D|$ gave 181.8 and 238.9\,ms for WDC-10K's skip condition); the \emph{ratio} is the stable quantity. $|D|$ is right-skewed on WDC-1M (mean __ND_1M__, median $\approx$3{,}557, max 16{,}715), so the median per-query saving there is __MED_1M__\% against a mean of __MEAN_1M__\%. $top\_n{=}30000$ is the deepest retrieval setting and so the largest $|D|$; the saving narrows at smaller $top\_n$.}
\end{frame}
"""


def write_frames(S, tex):
    f = FRAMES.replace("__TBL__", tex)
    rep = {
        "__SAVED_LIST__": ", ".join("%.1f\%% on %s" % (s["saved"], s["label"]) for s in S),
        "__XLO__": "%.1f" % min(s["x"] for s in S), "__XHI__": "%.1f" % max(s["x"] for s in S),
        "__SC_SL__": "%.1f" % S[0]["ksc"], "__SC_SL2__": "%.1f" % S[0]["sc"],
        "__SC_1M__": "%.1f" % S[3]["ksc"], "__SC_1M2__": "%.1f" % S[3]["sc"],
        "__ND_1M__": "{:,}".format(round(S[3]["n_D"])).replace(",", "{,}"),
        "__ST_1M__": "%.1f" % S[3]["kst2"], "__ST_1M2__": "%.1f" % S[3]["st2"],
        "__S1LO__": "%.2f" % min(s["s1"] for s in S), "__S1HI__": "%.1f" % max(s["s1"] for s in S),
        "__FEAS__": ", ".join("%d/%d" % (s["feas_two"], s["n"]) for s in S),
        "__ULO__": "%.0f" % min(s["u_two"] / s["u_skip"] * 100 for s in S),
        "__UHI__": "%.0f" % max(s["u_two"] / s["u_skip"] * 100 for s in S),
        "__MED_1M__": "%.1f" % S[3]["saved_med"], "__MEAN_1M__": "%.1f" % S[3]["saved"],
    }
    for a, b in rep.items():
        f = f.replace(a, b)
    with open(os.path.join(RES, "stage1_benefit_frames.tex"), "w") as fh:
        fh.write(f)
    return f


def main():
    S = stats()
    body = "".join(ROW % (s["label"], s["n"], round(s["n_D"]), s["s1"], s["sc"], s["st2"],
                          s["tot"], s["ksc"], s["kst2"], s["ktot"], s["saved"], s["x"])
                   for s in S)
    tex = TABLE.replace('__BODY__', body)
    with open(os.path.join(RES, "stage1_benefit_table.tex"), "w") as f:
        f.write(tex + "\n")

    md = ["# Stage 1 benefit: post-retrieval pipeline time\n",
          "`top_n=30000`, `k=10`, `alpha=3` (pool `alpha*k=30`). "
          "`F*/delta` = 0.122/0.07 on Santos-Large, 0.20/0.10 on the WDC tiers.\n",
          "**Timed window**: starts immediately before Stage 1, ends when the pipeline returns `R`. "
          "Retrieval, the overlap intersection and the `N_i`/`n_i` synopsis lookup are all outside it. "
          "Each query is retrieved once and the identical `D` is handed to both conditions, so they "
          "differ in nothing but whether Stage 1 runs. All times are means over the cohort, in ms.\n",
          "- `two_stage`: Stage 1 (Dinkelbach) -> score `U` on the 30-table pool -> Stage 2 over the pool",
          "- `skip_stage1`: score `U` on all of `D` -> Stage 2 over `D`, cardinality `k`",
          "\nSaving = `(T_skip - T_two)/T_skip`; the `x` column is `T_skip/T_two`.\n",
          "| Dataset | n | \\|D\\| | S1 | score | S2 | **two-stage total** | score | S2 | "
          "**skip total** | saved % | x |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in S:
        md.append("| %s | %d | %d | %.2f | %.2f | %.2f | **%.1f** | %.1f | %.1f | **%.1f** | "
                  "**%.1f** | %.1f |" % (s["label"], s["n"], round(s["n_D"]), s["s1"], s["sc"],
                                         s["st2"], s["tot"], s["ksc"], s["kst2"], s["ktot"],
                                         s["saved"], s["x"]))
    md.append("\n## Findings\n")
    md.append("- **The saving grows with the corpus**: %s. Stage 1 caps the number of unionability "
              "computations at `alpha*k=30` regardless of how many candidates retrieval returns, "
              "so the larger `|D|` is, the more work it removes." %
              ", ".join("%.1f%% on %s" % (s["saved"], s["label"]) for s in S))
    md.append("- **Almost all of it is unionability scoring.** Scoring falls from %s, because "
              "`two_stage` runs 30 bipartite matchings and `skip_stage1` runs `|D|` of them. This is "
              "the paper's `alpha*k`-not-`|T|` efficiency claim measured directly." %
              "; ".join("%.1f to %.2f ms on %s" % (s["ksc"], s["sc"], s["label"]) for s in S))
    md.append("- **At WDC-1M, Stage 2 becomes expensive too.** Its ILP runs over ~%d candidates "
              "instead of 30, costing %.1f ms against %.2f ms -- so skipping Stage 1 inflates both "
              "the scoring and the solving, not just the scoring." %
              (round(S[3]["n_D"]), S[3]["kst2"], S[3]["st2"]))
    md.append("- **Stage 1 pays for itself many times over**: it costs %s, against savings of "
              "hundreds of ms." % ", ".join("%.2f ms on %s" % (s["s1"], s["label"]) for s in S))
    md.append("- **Feasibility is unaffected**: %s -- identical in both conditions on every dataset. "
              "The distribution-aware pre-filter never cost a feasible answer here." %
              ", ".join("%d/%d on %s" % (s["feas_two"], s["n"], s["label"]) for s in S))
    md.append("\n## The quality trade-off\n")
    md.append("`skip_stage1` optimizes over a superset of `two_stage`'s candidates under the "
              "identical constraint, so its `sum_U` can only be greater than or equal. Mean `sum_U`:")
    md.append("\n| Dataset | two-stage | skip Stage 1 | retained |")
    md.append("|---|---:|---:|---:|")
    for s in S:
        md.append("| %s | %.1f | %.1f | %.0f%% |" % (s["label"], s["u_two"], s["u_skip"],
                                                     s["u_two"] / s["u_skip"] * 100))
    md.append("\nSo the speedup is not free: the pool retains %d-%d%% of the unionability that "
              "scoring every candidate would achieve, while cutting post-retrieval time by "
              "%.0f-%.0f%%." % (min(s["u_two"] / s["u_skip"] * 100 for s in S),
                                max(s["u_two"] / s["u_skip"] * 100 for s in S),
                                min(s["saved"] for s in S), max(s["saved"] for s in S)))
    md.append("\n## Caveats\n")
    md.append("1. `top_n=30000` is the deepest retrieval setting, hence the largest `|D|`; the saving "
              "narrows at smaller `top_n`, where `|D|` is smaller.")
    md.append("2. `|D|` is right-skewed on WDC-1M (mean %d, median ~3{,}557, max 16{,}715), so the "
              "median per-query saving (%.1f%%) is reported alongside the mean (%.1f%%)." %
              (round(S[3]["n_D"]), S[3]["saved_med"], S[3]["saved"]))
    md.append("3. One WDC-1M query has `|D|=2 < k`, returning an empty result in both conditions "
              "with no work done; it is excluded (n=66).")
    md.append("4. One run per cell, ms-scale values carry process noise; the monotone trend across "
              "corpus sizes is the stable finding.")
    md.append("\nScript: `experiments/stage1_benefit.py` (sbatch). Per-query CSVs: "
              "`results/stage1_benefit_<dataset>.csv`.")
    with open(os.path.join(REPO, "experiments/stage1-benefit-report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    write_frames(S, tex)
    print(tex)
    print("\nwrote stage1_benefit_table.tex + stage1-benefit-report.md")


if __name__ == "__main__":
    main()
