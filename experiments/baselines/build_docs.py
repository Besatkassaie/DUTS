"""Rebuild experiments/baselines-report.md and the beamer section 'Starmie baselines' (run make_baseline_report first).
Metric convention: precision / recall / ideal recall over ALL queries. Fair methods (exhaustive, nl): an infeasible query
counts 0 for precision and recall; sum U and F_R over feasible queries. Pure Starmie: everything over all queries as returned,
regardless of tau, with the reached-tau count beside it."""
import re

RES = "/u6/bkassaie/DUTS/experiments/results"
tabs = open(RES + "/baselines_tables.md").read()
blocks = [b.strip("\n") for b in re.split(r"\n(?=\*\*)", tabs.strip("\n"))]
T = dict(zip(["s_main", "s_det", "e_main", "e_det", "n_main", "n_det"], blocks))
fr = {}
for blk in [x for x in re.split(r"(?=% ---- )", open(RES + "/baselines_frames.tex").read()) if x.strip()]:
    m = re.match(r"% ---- (\w+) (\w+)", blk)
    fr[(m.group(1), m.group(2))] = blk

DETAIL_HEAD = {6: "bench & $k$ & cand. & $<k$ ret. & needed & succeeded & swaps & $F$ before \\\\",
               11: "bench & $k$ & cand. & $<k$ ret. & needed & succeeded & swaps & $F$ before & removed / pool & share & left \\\\",
               4: "bench & $k$ & cand. & $<k$ ret. \\\\"}
for key in list(fr):
    if key[1] == "detail":
        lines = fr[key].split("\n")
        for i, l in enumerate(lines):
            if l.startswith("bench & k & mean candidates scored"):
                n = l.count("&") + 1
                new = {4: DETAIL_HEAD[4], 8: DETAIL_HEAD[6], 11: DETAIL_HEAD[11]}[n]
                lines[i] = new
        fr[key] = "\n".join(lines)

NOTE_S = "\\vspace{2pt}\n{\\footnotesize\\color{Muted} Pure Starmie is scored on the list it returns, \\emph{regardless of $\\tau$}: precision, recall, ideal recall, $\\Sigma U$ and $F$ are means over \\textbf{all} queries. \\emph{reached $\\tau$}: queries whose unfair top-$k$ has $F\\ge\\tau{=}0.3$.}"
NOTE_F = "\\vspace{2pt}\n{\\footnotesize\\color{Muted} precision, recall, ideal recall: means over \\textbf{all} queries; a query that is not feasible (exactly $k$ tables with $F\\ge\\tau{=}0.3$) counts 0 for precision, recall, $\\Sigma U$ and $F_R$. %s}"
notes = {
    ("starmie", "main"): NOTE_S,
    ("exhaustive", "main"): NOTE_F % "When the exhaustive swap cannot reach $\\tau$ it returns nothing.",
    ("nl", "main"): NOTE_F % "The nested-loop swap returns $k$ tables even when it failed; that unfair list is discarded (counts 0).",
    ("starmie", "detail"): "\\vspace{2pt}\n{\\footnotesize\\color{Muted} \\emph{cand.}: mean tables verified after the $N{=}1000$ nearest-column lookup, before taking the top-$k$. \\emph{$<k$ ret.}: queries with fewer than $k$ tables returned. No swap is done, so there is no swap information; the reached-$\\tau$ count and $F$ are in the results table.}",
    ("exhaustive", "detail"): "\\vspace{2pt}\n{\\footnotesize\\color{Muted} \\emph{cand.}: mean candidates scored. \\emph{$<k$ ret.}: queries with fewer than $k$ tables returned (swap failed, empty list). \\emph{needed}: the initial top-$k$ had $F<\\tau$, so swaps ran. \\emph{succeeded}: swaps brought $F$ to $\\ge\\tau$ (out of those that needed it). \\emph{swaps}: mean swaps made, over the queries that needed fairification (failed ones included). \\emph{$F$ before}: mean $F$ of the initial top-$k$ of those queries.}",
    ("nl", "detail"): "\\vspace{2pt}\n{\\footnotesize\\color{Muted} Columns as for the exhaustive swap, plus the winnow filter: \\emph{removed / pool}: candidates dropped as dominated out of those considered, summed over queries; \\emph{share}: their ratio; \\emph{left}: mean candidates left for swapping. The nested-loop swap returns its last (possibly unfair) state on failure, so it never returns fewer than $k$.}",
}
for k, v in notes.items():
    fr[k] = fr[k].replace("@@NOTE_%s_%s@@" % k, v)


def frame(t, s, b):
    return "\\begin{frame}{%s}{%s}\n\\small\n%s\n\\end{frame}\n\n" % (t, s, b)


setup = r'''\section{Starmie baselines -- pure Starmie, exhaustive swap, nested-loop swap}

''' + frame("Starmie baselines: what was run", "same fair queries, datalake and ground truth as the DUTS tables -- no timings reported", r'''\begin{itemize}\setlength\itemsep{4pt}
\item \highlight{Pure Starmie:} the $N$ nearest columns of every query column give the candidates; each is scored by bipartite column matching; the top-$k$ is returned. No protected attribute is used.
\item \highlight{Exhaustive swap (no filtering):} candidates are scored with the protected column \emph{pinned} in the matching; if the initial top-$k$ has $F<\tau$, violating tables are swapped out for the best replacement, trying \emph{every} candidate.
\item \highlight{Nested-loop swap (winnow filtering):} the same swap, but the candidates are first pruned by a dominance filter (a candidate is dropped if another has more protected rows, fewer other rows and at least as good unionability).
\item Fixed for all three: $N{=}1000$, $\sigma{=}0.6$, $F^*{=}0.4$, $\delta{=}0.1$ ($\tau{=}0.3$), $k\in\{5,10,20\}$, on santos3 (48 queries), tusSmall3 (92), tusLarge3 (142).
\item \textbf{Metrics, recomputed by our code as for DUTS.} Precision, recall and ideal recall are means over \textbf{all} queries. For the two swap methods an infeasible query (not exactly $k$ tables with $F\ge\tau$) counts 0 for precision, recall, $\Sigma U$ and $F_R$. \textbf{Pure Starmie is scored on what it returns, regardless of $\tau$}, with the number of queries that reached $\tau$ shown beside it; its precision, recall, $\Sigma U$ and $F$ are over all queries.
\end{itemize}
{\footnotesize\color{Muted} The baselines are Starmie's own code (\texttt{HNSWSearcher\_Fair}), run unmodified except for the three changes listed on the next slide. Because pure Starmie is not zeroed when it misses $\tau$, its precision and recall count unfair lists and are not directly comparable with the swap methods'.}''') + frame("Starmie baselines: defaults and changes", "taken from the Starmie code, not chosen by us", r'''\begin{itemize}\setlength\itemsep{3pt}\footnotesize
\item \highlight{Index:} HNSW, cosine, $M{=}32$, $ef_{construction}{=}100$, $ef{=}10$ (a query with $N{=}1000$ uses $\max(ef,N)$); tables shuffled with seed 42 at scale 1.0.
\item \highlight{Matching:} an edge counts if cosine $>\sigma$. We use $\sigma{=}0.6$, as DUTS; Starmie's own default for these benchmarks is $0.7$.
\item \highlight{Dominance rules} of the nested-loop filter: the bundled \texttt{preference\_config.json} (protected count, non-protected count, unionability score), unchanged.
\item \highlight{Swap algorithms:} default removal and replacement order; no parameters were set.
\item \highlight{$F$:} Starmie's \texttt{compute\_F} over the benchmark's \texttt{metadata\_combined.pkl}: any column counts as categorical (no $\theta_{cat}{=}50$ limit); a table adds rows only if its protected column aligns in the constrained matching; the query is included. Pure Starmie is scored after the fact with the same alignment.
\item \highlight{Our three changes:} (1) the protected value is matched with its numeric spelling variants (\texttt{4}/\texttt{4.0}), as in the DUTS runs; (2) the raw datalake is not loaded into memory and the metadata pickle is not rewritten; (3) the index is written to a scratch folder.
\item \highlight{Metrics:} recomputed from the returned lists (Starmie's own script averages precision only over queries with $|gt|\ge k$ and uses an uncapped ideal recall, so it is not comparable).
\item \highlight{$\Sigma U$:} Starmie's matching score of the returned tables; it equals the DUTS pinned-match score in all but 20 of 2{,}538 rows.
\end{itemize}''')

setup = setup.replace("\\small\n\\begin{itemize}\\setlength\\itemsep{4pt}\n\\item \\highlight{Pure Starmie:}", "\\footnotesize\n\\begin{itemize}\\setlength\\itemsep{3pt}\n\\item \\highlight{Pure Starmie:}", 1)

F_S = frame("What the pure Starmie table shows", "findings in plain words -- all queries, as returned, regardless of $\\tau$", r'''\begin{itemize}\setlength\itemsep{4pt}
\item \highlight{The unfair top-$k$ rarely meets the fairness target.} It has $F\ge0.3$ for only santos3 $21/12/9$ of 48, tusSmall3 $7/5/1$ of 92, tusLarge3 $22/13/8$ of 142 queries ($k{=}5/10/20$).
\item \highlight{It gets worse as $k$ grows.} Mean $F$ over all queries: santos3 $0.34/0.27/0.21$, tusSmall3 $0.16/0.13/0.09$, tusLarge3 $0.18/0.14/0.11$; $\Sigma U$ grows with $k$ ($60/116/220$ on santos3).
\item \highlight{Precision over all queries:} $0.89/0.83/0.75$ (santos3), $0.90/0.89/0.84$ (tusSmall3), $0.79/0.77/0.71$ (tusLarge3).
\item \highlight{Recall vs ideal recall:} santos3 $0.28/0.42/0.65$ vs $0.30/0.47/0.73$; tusSmall3 $0.010/0.019/0.036$ vs $0.011/0.021/0.042$; tusLarge3 $0.010/0.019/0.035$ vs $0.012/0.025/0.050$.
\end{itemize}
\vspace{2pt}
{\footnotesize\color{Muted} These are the accuracy of Starmie's own (mostly unfair) lists; the swap methods are zeroed when they fail, so their numbers are lower by construction.}''')
F_E = frame("What the exhaustive-swap table shows", "findings in plain words -- $N{=}1000$, $\\sigma{=}0.6$", r'''\begin{itemize}\setlength\itemsep{4pt}
\item \highlight{Swapping makes many more queries feasible} than pure Starmie reaches: santos3 $47/42/31$ of 48, tusSmall3 $66/43/24$ of 92, tusLarge3 $108/77/49$ of 142 -- but the number drops quickly as $k$ grows.
\item \highlight{Precision and recall over all queries (failed $=0$)} fall with $k$ for that reason: precision $0.88/0.70/0.44$ (santos3), $0.68/0.43/0.23$ (tusSmall3), $0.63/0.40/0.21$ (tusLarge3); recall $0.25/0.31/0.33$, $0.0085/0.0135/0.0202$, $0.0086/0.0129/0.0166$ against ideal recall $0.30/0.47/0.73$, $0.011/0.021/0.042$, $0.012/0.025/0.050$.
\item \highlight{Almost every query needs the swap:} the initial top-$k$ violates the target for $28/36/39$ of 48 (santos3), $85/87/90$ of 92 (tusSmall3), $122/129/134$ of 142 (tusLarge3); it succeeds for, e.g., $27/28$, $30/36$, $22/39$ on santos3.
\item \highlight{When it fails it returns nothing} ($1/6/17$ empty results on santos3, $26/49/68$ on tusSmall3, $34/65/93$ on tusLarge3) and \highlight{it stops as soon as the target is met:} among the feasible queries mean $F_R$ is only just above $0.3$ ($0.31$--$0.41$); the table's $\Sigma U$ and $F_R$ include the failed queries as 0.
\end{itemize}''')
F_N = frame("What the nested-loop-swap table shows", "findings in plain words -- $N{=}1000$, $\\sigma{=}0.6$", r'''\begin{itemize}\setlength\itemsep{4pt}
\item \highlight{Fewer feasible queries than the exhaustive swap} in every row except santos3 at $k{=}5$: santos3 $47/35/20$ of 48, tusSmall3 $52/33/12$ of 92, tusLarge3 $101/59/39$ of 142.
\item \highlight{Precision and recall over all queries (failed $=0$)} are correspondingly lower: precision $0.85/0.61/0.32$ (santos3), $0.53/0.32/0.11$ (tusSmall3), $0.57/0.28/0.15$ (tusLarge3); recall $0.23/0.25/0.23$, $0.0076/0.0116/0.0095$, $0.0083/0.0109/0.0134$.
\item \highlight{The winnow filter removes 89--96\% of the swap candidates} (e.g.\ tusSmall3, $k{=}5$: 10{,}828 of 11{,}306), leaving on average only 2--6 candidates per query -- fewer chances to find a replacement that reaches $\tau$.
\item \highlight{It always returns $k$ tables, even when the swap failed;} those unfair lists are discarded here (counted 0, also for $\Sigma U$ and $F_R$). Among the feasible queries mean $F_R$ is $0.31$--$0.41$.
\end{itemize}''')
sect = setup
for a, F_ in (("starmie", F_S), ("exhaustive", F_E), ("nl", F_N)):
    sect += fr[(a, "main")] + F_ + fr[(a, "detail")]
p = "/u6/bkassaie/DUTS/experiments/beamer/duts_presentation.tex"
t = open(p).read()
a = t.index("\\section{Starmie baselines")
b = t.index("\\section{santosLarge")
t = t[:a] + sect + t[b:]
open(p, "w").write(t)

md = """# Starmie baselines on the three fairified benchmarks

**2026-09-21** · Starmie code: `/u6/bkassaie/starmie_fair` (`HNSWSearcher_Fair`, imported unmodified except for the changes listed in §1) ·
driver `experiments/baselines/run_starmie_baselines.py`, tables `make_baseline_report.py` / `build_docs.py` · raw rows
`experiments/results/baselines_<benchmark>_<approach>.csv` (2,538 rows: one per query and k).
Same fair queries (santos3 48, tusSmall3 92, tusLarge3 142), fair datalake and fair ground truth as `fair3-report.md`.
Fixed for all three approaches: **N = 1000** nearest columns per query column, **σ = 0.6**, **F\\* = 0.4, δ = 0.1 (τ = 0.3)**,
**k ∈ {5, 10, 20}**. No timings are reported.

## 1. What was run, and the defaults used

* **Pure Starmie** (`HNSWSearcher_Fair.topk`): the N nearest columns of every query column give the candidate tables; each is
  scored by unconstrained bipartite column matching; the top-k is returned. No protected attribute is used.
* **Exhaustive swap** (`topk_fairified(algorithm="exhustive_swap")`, no filtering): candidates are scored with the protected column
  pinned in the matching (`verify_constrained`); if the initial top-k has F < τ, violating tables are swapped out for the best
  replacement, trying every candidate.
* **Nested-loop swap** (`nl_swap`, winnow filtering): the same, but the candidates are first pruned by a dominance filter.

**Defaults taken from the Starmie code (not chosen by us).**
* Index: HNSW, cosine, M = 32, ef_construction = 100, ef = 10 (a query with N = 1000 uses max(ef, N)); tables shuffled with seed 42 at scale 1.0.
* Matching: an edge counts if cosine > σ. We use σ = 0.6 (as DUTS); Starmie's own default for these benchmarks is 0.7.
* Dominance rules of the nested-loop filter: the bundled `preference_config.json` (protected count, non-protected count, unionability score), unchanged.
* Swap algorithms: default removal and replacement order; no parameters were set.
* F: Starmie's `compute_F` over the benchmark's `metadata_combined.pkl` — any column counts as categorical (no θ_cat = 50 limit);
  a table adds rows only if its protected column aligns in the constrained matching; the query is included. Pure Starmie is scored
  after the fact with the same alignment (its ranking uses the unconstrained matching).
* Feasible = exactly k tables returned and F ≥ τ (Starmie's rounding of F and δ).

**Our three changes to how the code is run.** (1) The protected value is matched with its numeric spelling variants (`4`/`4.0`),
as in the DUTS runs. (2) The raw datalake is not loaded into memory and the metadata pickle is not rewritten. (3) The HNSW index
is written to a scratch folder.

**Metrics** are recomputed from the returned lists with the same code as the DUTS tables (Starmie's own `calcMetrics_new`
averages precision only over queries with |gt| ≥ k and uses an uncapped ideal recall, so it is not comparable).
* **Precision, recall and ideal recall are means over all queries.** For the two swap methods an infeasible query counts 0 for
  precision, recall, ΣU and F_R (the exhaustive swap then returns nothing; the nested-loop swap's unfair list is discarded).
* **Pure Starmie is scored on what it returns, regardless of τ:** precision, recall, ideal recall, ΣU and F are means over all
  queries (nothing is zeroed); the number of queries that reached τ is shown beside them.
* ΣU is Starmie's matching score of the returned tables; it equals the DUTS pinned-match score in all but 20 of 2,538 rows.

## 2. Pure Starmie (no fairification)

@S_MAIN

**What it shows.** *The unfair top-k rarely meets the fairness target:* it has F ≥ 0.3 for only santos3 21/12/9 of 48, tusSmall3 7/5/1 of 92 and tusLarge3 22/13/8 of 142 queries (k = 5/10/20). *It gets worse as k grows:* mean F over all queries is 0.34/0.27/0.21 (santos3), 0.16/0.13/0.09 (tusSmall3), 0.18/0.14/0.11 (tusLarge3). *Precision over all queries:* 0.89/0.83/0.75, 0.90/0.89/0.84 and 0.79/0.77/0.71. *Recall vs ideal recall:* santos3 0.28/0.42/0.65 vs 0.30/0.47/0.73; tusSmall3 0.010/0.019/0.036 vs 0.011/0.021/0.042; tusLarge3 0.010/0.019/0.035 vs 0.012/0.025/0.050. These are the accuracy of Starmie's own, mostly unfair, lists; the swap methods are zeroed when they fail, so their numbers are lower by construction.

@S_DET

*cand.* is the mean number of tables verified after the N-nearest-column lookup, before taking the top-k; no swap is done, so there is no swap information.

## 3. Exhaustive swap (no filtering)

@E_MAIN

**What it shows.** *Swapping makes many more queries feasible* than pure Starmie reaches — santos3 47/42/31 of 48, tusSmall3 66/43/24 of 92, tusLarge3 108/77/49 of 142 — but the number falls quickly as k grows. *Precision and recall over all queries (failed = 0) fall with k for that reason:* precision 0.88/0.70/0.44 (santos3), 0.68/0.43/0.23 (tusSmall3), 0.63/0.40/0.21 (tusLarge3); recall 0.25/0.31/0.33, 0.0085/0.0135/0.0202, 0.0086/0.0129/0.0166 against ideal recall 0.30/0.47/0.73, 0.011/0.021/0.042, 0.012/0.025/0.050. *Almost every query needs the swap:* the initial top-k violates the target for 28/36/39 of 48 (santos3), 85/87/90 of 92 (tusSmall3), 122/129/134 of 142 (tusLarge3); it succeeds for e.g. 27/28, 30/36, 22/39 on santos3 and 59/85, 38/87, 22/90 on tusSmall3. *When it fails it returns nothing* (1/6/17 empty results on santos3, 26/49/68 on tusSmall3, 34/65/93 on tusLarge3), and *it stops as soon as the target is met:* among the feasible queries mean F_R is only just above 0.3 (0.31–0.41); the table's ΣU and F_R include the failed queries as 0.

@E_DET

*needed*: the initial top-k had F < τ so swaps ran; *succeeded*: swaps brought F to ≥ τ (of those that needed it); *swaps*: mean swaps over the queries that needed fairification (failed ones included); *queries with < k returned*: the swap failed and the empty list was returned.

## 4. Nested-loop swap (winnow filtering)

@N_MAIN

**What it shows.** *Fewer feasible queries than the exhaustive swap* in every row except santos3 at k = 5 — santos3 47/35/20 of 48, tusSmall3 52/33/12 of 92, tusLarge3 101/59/39 of 142. *Precision and recall over all queries (failed = 0) are correspondingly lower:* precision 0.85/0.61/0.32 (santos3), 0.53/0.32/0.11 (tusSmall3), 0.57/0.28/0.15 (tusLarge3); recall 0.23/0.25/0.23, 0.0076/0.0116/0.0095, 0.0083/0.0109/0.0134. *The winnow filter removes 89–96% of the swap candidates* (e.g. tusSmall3, k=5: 10,828 of 11,306), leaving on average only 2–6 candidates per query — fewer chances to find a replacement that reaches τ. *It always returns k tables, even when the swap failed;* those unfair lists are discarded here (counted 0). Among the feasible queries mean F_R is 0.31–0.41.

@N_DET

Columns as for the exhaustive swap, plus the winnow filter: *removed / pool* = candidates dropped as dominated out of those considered, summed over queries; *share* = their ratio; *left* = mean candidates left for swapping. The nested-loop swap returns its last (possibly unfair) state on failure, so it never returns fewer than k.

## 5. Caveats

* Precision and recall count a failed query as 0 for the swap methods but not for pure Starmie (scored regardless of τ), so pure Starmie's numbers count unfair lists and are not directly comparable with the swap methods'.
* The three approaches differ in what they return on failure (exhaustive: nothing; nested-loop: its last unfair state; pure Starmie: the unfair top-k).
* N (nearest columns per query column) is not the same quantity as DUTS's `top_n` (nearest columns to the protected column only), so candidate-set sizes differ (mean candidates scored: 106 / 145 / 125 for santos3 / tusSmall3 / tusLarge3).
* Ground truth counts every version of a table (original and each fair copy) as unionable; precision and recall are by table name.
"""
for k, v in T.items():
    md = md.replace("@" + k.upper(), v)
open("/u6/bkassaie/DUTS/experiments/baselines-report.md", "w").write(md)
print("built")
