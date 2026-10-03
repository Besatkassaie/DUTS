"""Rebuild experiments/fair3-report.md and the beamer section 'Fairified benchmarks' from
experiments/results/fair3_tables.md / fair3_frames.tex (run make_report first).
Metric convention: precision, recall (and precision/recall at D, P) are means over ALL queries, a failed query counts 0;
ideal recall over all queries (identical for every method); sum U and F_R over feasible queries only."""
import re

RES = "/u6/bkassaie/DUTS/experiments/results"
tabs = open(RES + "/fair3_tables.md").read()
Q, S, P = [b.strip("\n") for b in re.split(r"\n(?=\*\*top_n = 1000)", tabs.strip("\n"))]
frames = re.sub(r"@@NOTE_\w+_\d+@@\n", "", open(RES + "/fair3_frames.tex").read())
fq, fs, fd = [x for x in re.split(r"(?=% ---- top_n=)", frames) if x.strip()]
shrink = lambda x: x.replace("\\begin{tabular}", "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}}", 1)
fq, fd = shrink(fq), shrink(fd)
# the stage table has its own shortened header and no shrinking
fs = re.sub(r"bench & \$\\alpha\$ & k & mean.*?\\\\\n", "bench & $\\\\alpha$ & $k$ & $|D|$ & $|D|{<}k$ & S1 skip & S1 ran & LP dec. & MILP ran & feasible & $\\\\ge k$ own & feas.\\\\ among \\\\\\\\\n", fs, count=1, flags=re.S)
for name in ("Quality of the fair result R", "Stage bypasses", "Precision and recall at D"):
    pass
fq = fq.replace("\\tiny", "\\scriptsize", 1); fs = fs.replace("\\tiny", "\\scriptsize", 1); fd = fd.replace("\\tiny", "\\scriptsize", 1)


def frame(title, sub, body):
    return "\\begin{frame}{%s}{%s}\n\\small\n%s\n\\end{frame}\n\n" % (title, sub, body)


E1 = frame("What the quality table shows", "findings in plain words -- $\\alpha{=}10$, $top\\_n{=}1000$", r'''\begin{itemize}\setlength\itemsep{4pt}
\item \highlight{Most queries get a fair answer.} Feasible queries ($k{=}5/10/20$): santos3 $42/39/26$ of 48, tusSmall3 $89/84/65$ of 92, tusLarge3 $133/122/93$ of 142. Feasible means $k$ tables whose union with the query has at least $30\%$ rows with the protected value.
\item \highlight{Precision and recall are means over all queries; a failed query counts 0.} precision@R: santos3 $0.88/0.81/0.52$, tusSmall3 $0.93/0.89/0.68$, tusLarge3 $0.79/0.72/0.53$. They fall with $k$ mainly because more queries fail.
\item \highlight{Recall stays below the ideal recall} (the best possible with $k$ tables, the same for every method): santos3 $0.18/0.30/0.34$ vs $0.30/0.47/0.73$, tusSmall3 $0.010/0.020/0.035$ vs $0.011/0.021/0.042$, tusLarge3 $0.010/0.019/0.029$ vs $0.012/0.025/0.050$. The gap is widest where many queries fail (santos3, $k{=}20$: 26 of 48 feasible).
\item \highlight{The fairness target is met} where an answer exists: among the feasible queries $F_R\ge0.31$ ($\tau{=}0.30$), about $0.44$ on santos3 and $0.31$--$0.41$ on TUS. In the table $\Sigma U$ and $F_R$ are means over \textbf{all} queries with a failed query counting 0, so they are lower where queries fail.
\end{itemize}
\vspace{2pt}
{\footnotesize\color{Muted} Example: tusSmall3, $\alpha{=}10$, $k{=}5$: 89 of 92 feasible, precision $0.933$, recall $0.0100$ vs ideal $0.0105$.}''')

E2 = frame("What the stage table shows", "how each query ends, and which stages were skipped -- $\\alpha{=}10$, $top\\_n{=}1000$", r'''\begin{itemize}\setlength\itemsep{4pt}
\item Every query ends one of three ways: \highlight{$|D|<k$} (too few candidates -- Stage 1 and 2 skipped), \highlight{LP decisive} (the LP proves no $k$ tables reach $F\ge0.3$ -- MILP skipped) or \highlight{MILP ran} (feasible; the MILP never failed after a feasible LP).
\item \highlight{santos3:} the failures are mostly too few candidates: $|D|{<}k$ for 4 / 7 / 20 of the 6 / 9 / 22 infeasible queries at $k{=}5/10/20$. Own copies are a sufficient, not a necessary, condition: only 3 queries have $\ge20$ own fair copies, yet 26 are feasible at $k{=}20$, because the kept original tables and other queries' copies fill the slots. Among queries with $\ge k$ own copies: $39/39$, $35/36$, $3/3$.
\item \highlight{TUS:} every query has at least $k$ own copies for $k\le10$, yet $3$--$20$ queries per row are infeasible -- retrieval did not bring enough of them into $D$ (next slides).
\item \highlight{Stage 1 is bypassed} ($P{=}D$) for most santos3 queries and for most TUS queries at $k{=}20$ (77/92, 119/142); at $k{=}5$ it runs for most TUS queries (81/92, 102/142). The LP pre-check is decisive for $2$--$34$ queries per row, each skipping a MILP call.
\end{itemize}
\vspace{2pt}
{\footnotesize\color{Muted} Columns: $|D|$ mean candidates; $|D|{<}k$ both stages skipped; S1 skip Stage 1 bypassed ($P{=}D$); S1 ran; LP dec.\ LP pre-check decisive (MILP skipped); MILP ran; $\ge k$ own queries with at least $k$ own fair copies; feas.\ among feasible among those.}
\vspace{2pt}
{\footnotesize\color{Muted} Example: tusSmall3, $\alpha{=}10$, $k{=}20$: $92 = 2\ (|D|{<}k) + 25\ (\text{LP infeasible}) + 65\ (\text{feasible})$. The optimizer cannot fix these failures: it only chooses inside $D$.}''')

E3 = frame("What the precision/recall table shows", "the same measures at $D$, $P$ and $R$, all queries -- $\\alpha{=}10$, $top\\_n{=}1000$", r'''\begin{itemize}\setlength\itemsep{4pt}
\item $D$: tables retrieved (before any selection). $P$: the pool of $\alpha k$ tables chosen by Stage 1. $R$: the final $k$ tables. An empty or missing set (no pool because $|D|{<}k$, or no result because the query failed) counts 0.
\item \highlight{The retrieved set is mostly unionable:} precision@D $0.90$ (santos3), $0.93$ (tusSmall3), $0.75$ (tusLarge3); recall@D $0.82/0.26/0.16$.
\item \highlight{Precision at $P$ and $R$ stays close to $D$ where queries succeed,} and drops at large $k$ because queries fail: santos3 precision@R $0.88/0.81/0.52$ (only 26 of 48 queries have a result at $k{=}20$).
\item \highlight{Recall falls from $D$ to $R$ because the sets shrink} ($D$ holds $\sim$25 / 120 / 90 tables, $R$ holds $k$) and because failed queries count 0.
\end{itemize}
\vspace{2pt}
{\footnotesize\color{Muted} Example: tusSmall3, $\alpha{=}10$, $k{=}5$: precision $0.934\to0.923\to0.933$ and recall $0.259\to0.095\to0.0100$ at $D\to P\to R$. Precision/recall at $D$ do not depend on $k$ or $\alpha$.}''')

setup = r'''\section{Fairified benchmarks -- santos3, tusSmall3, tusLarge3}

\begin{frame}{Fairified benchmarks: setup and what is reported}{same pipeline, same metrics on all three -- $F^*{=}0.4$, $\delta{=}0.1$ ($\tau{=}0.3$), $top\_n{=}1000$}
\small
\begin{center}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrr}
\toprule
 & santos3 & tusSmall3 & tusLarge3 \\
\midrule
fair queries & 48 & 92 & 142 \\
datalake tables (originals + fair copies) & 999 & 9,178 & 18,309 \\
mean ground truth tables / query & 28.1 & 852.7 & 626.7 \\
own fair copies / query (min / mean) & 1 / 11.0 & 16 / 85.9 & 27 / 100.0 \\
mean $|D|$ & 25.5 & 122.5 & 90.5 \\
\bottomrule
\end{tabular}
\end{center}
\vspace{4pt}
\begin{itemize}\setlength\itemsep{2pt}
\item Grid: $k\in\{5,10,20\}\times\alpha\in\{10,5\}$; real pinned-match $U$ ($\sigma{=}0.6$).
\item \textbf{Precision, recall, $\Sigma U$ and $F_R$ are means over all queries; a failed query (empty result) counts 0.} \textbf{Ideal recall} is over all queries and is identical for every method.
\item Also reported: feasible queries, \textbf{bypassed stages} and LP pre-check decisions.
\end{itemize}
{\footnotesize\color{Muted} All three are built the same way: one fair copy per (query, table) pair, ratio $\ge0.4$ for the query's value, matched by column name. A table that cannot be augmented for a query is removed from \emph{that query's} ground truth (and from the datalake if it fails for every query), so every ground-truth table has a fair version for its query. The original table and each fair copy are separate unionable tables (both count as hits in precision/recall; the original may not carry enough of the value to reach $\tau$). Ground truth includes every copy of a unionable table. Protected value matched with numeric spelling variants (\texttt{4}/\texttt{4.0}). Details: \texttt{fair3-report.md}.}
\end{frame}

'''
why = r'''\begin{frame}{Why some queries stay infeasible: own fair copies in $D$}{$top\_n{=}1000$ -- each query has its own fair copies (ratio $\ge0.4$ for its value) in the datalake}
\small
\begin{center}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}
\toprule
benchmark & mean $|D|$ & own copies in $D$ & own copies in datalake & ratio over $D$ & ratio, own copies in $D$ \\
\midrule
\texttt{santos3} & 25.5 & 9.9 & 11.0 & 0.372 & 0.485 \\
\texttt{tusSmall3} & 122.4 & 17.1 & 85.9 & 0.152 & 0.396 \\
\texttt{tusLarge3} & 83.8 & 13.6 & 100.0 & 0.173 & 0.386 \\
\bottomrule
\end{tabular}
\end{center}
\vspace{4pt}
\begin{itemize}\setlength\itemsep{3pt}
\item \highlight{santos3:} retrieval brings almost all of the query's own copies into $D$ (9.9 of 11.0) and the ratio over $D$ ($0.37$) is above $\tau{=}0.3$; the failures come mostly from queries with fewer than $k$ candidates.
\item \highlight{TUS:} only $\sim$17 of $\sim$86 (tusSmall3) and $\sim$14 of 100 (tusLarge3) own copies reach $D$; the ratio over all of $D$ ($0.15$--$0.17$) is below $\tau$. Copies of one table have nearly identical vectors, so the $top\_n$ window fills with originals and with copies made for \emph{other} queries, where the query's value is rare. 9 of 142 tusLarge3 queries get none.
\item Retrieval is not fairness-aware and the optimizer only sees $D$: these failures are a retrieval limit, and the MILP never failed.
\end{itemize}
{\footnotesize\color{Muted} Separate run of the same code (own HNSW build), so $|D|$ differs slightly from the tables. Details: \texttt{fair3-report.md}.}
\end{frame}

'''
sect = setup + fq + E1 + fs + E2 + fd + E3 + why
p = "/u6/bkassaie/DUTS/experiments/beamer/duts_presentation.tex"
t = open(p).read()
a = t.index("\\section{Fairified benchmarks")
b = t.index("\\section{Starmie baselines") if "\\section{Starmie baselines" in t else t.index("\\section{santosLarge")
t = t[:a] + sect + t[b:]
open(p, "w").write(t)

md = """# DUTS on the three fairified benchmarks — santos3, tusSmall3, tusLarge3

**2026-09-21** · conda `TableUnionNew`, py 3.8.5 · real pinned-match unionability (σ=0.6) · seed 42.
`F* = 0.4`, `δ = 0.1` (`τ = 0.3`), `include_query = True`. Grid: `k ∈ {5, 10, 20}` × `α ∈ {10, 5}`, `top_n = 1000`.
Code: `experiments/fair3/eval_fair3.py` (runner), `make_report.py` / `build_docs.py` (tables and text), `diag_own_copies.py` (§4);
raw rows: `experiments/results/fair3_eval_<benchmark>_n1000.csv` (1,692 rows).
This replaces all earlier santos3 / fair3 results (old files: `experiments/results/_stale_fair3_pre_prune/`).

| | santos3 | tusSmall3 | tusLarge3 |
|---|---:|---:|---:|
| Fair queries evaluated | 48 | 92 | 142 |
| Datalake tables (originals + fair copies) | 999 (471 + 528) | 9,178 (1,272 + 7,906) | 18,309 (4,102 + 14,207) |
| Mean ground-truth tables per query | 28.1 | 852.7 | 626.7 |
| Own fair copies per query (min / mean) | 1 / 11.0 | 16 / 85.9 | 27 / 100.0 |
| Queries with ≥ 5 / ≥ 10 / ≥ 20 own copies | 39 / 36 / 3 | 92 / 92 / 85 | 142 / 142 / 142 |
| Mean `\\|D\\|` | 25.5 | 122.5 | 90.5 |
| Queries whose protected value has a numeric spelling variant (M expanded) | 10/48 | 10/92 | 10/142 |

## 0. How the benchmarks are built, and what is measured

**Construction (identical for all three).** For every ground-truth pair (Q, D) an independent copy of D is made and
rebalanced for Q only: protected-value tuples of Q are appended, projected onto D's schema, until the ratio of Q's value
is ≥ 0.4 (+ up to 25% random overshoot). The protected column is matched **by name** only. If D has no same-named
column, D cannot be fairified for Q, so D is removed from **Q's** ground truth (the original and every copy of D);
D stays in the datalake for queries where it can be augmented, and is deleted from the datalake only if it fails for
every query it is unionable with. Consequently every table in a query's ground truth has a fair version for that
query. The ground truth of a query holds the original tables and every copy of them (a copy made for another query is
still unionable with this one).

**Pipeline per query and `(k, α)` cell:** `retrieval → Stage 1 → score P → Stage 2`, composed as `dutsx.runner.run_query`
does, except that retrieval is run once per query and reused across cells. Retrieval is `D = D_sem ∩ D_ovl` (§7.3); the
synopsis is `MetadataStoreSynopsis` over each benchmark's `metadata_combined.pkl`, `θ_cat = 50`.

* **feasible** = Stage 2 returned `k` tables with `F ≥ τ`.
* **precision / recall (at D, P, R)**: tables of the set that are in the query's ground truth / the set size, and / `|gt|`.
  **Means over ALL queries of the benchmark; a query that failed (no result) counts 0.** At D and P an empty or missing set
  (e.g. no pool because `|D| < k`) also counts 0. **Ideal recall@k** = `min(k, |gt|) / |gt|`, the ceiling for `k` returned
  tables, averaged over all queries; it is therefore identical for every method.
* **ΣU and F_R** are also means over all queries, and a failed query (empty result) counts 0 for them too (the mean over the feasible queries alone is the table value times n / feasible, e.g. F_R ≈ 0.44 on santos3).
* **Bypassed stages** (counted per row, out of the queries of that benchmark): `|D| < k` (`insufficient_candidates`:
  Stage 1 and Stage 2 both skipped); **S1 bypassed (P=D)** when `α·k ≥ |D|` (Stage 1 skipped, Stage 2 still runs);
  **LP decisive** = the Stage 2 LP pre-check proved infeasibility so the MILP was skipped; **MILP ran** = the LP was
  feasible. A feasible LP followed by an infeasible MILP occurred **0** times.
* **Queries with ≥ k own copies**: after pruning, a feasible answer is guaranteed to exist from the query's own fair
  copies alone if it has at least `k` of them (sufficient, not necessary: the original tables, which stay in the datalake,
  and copies made for other queries can also fill the `k` slots, but they carry the query's value only at its natural
  share); the last two columns of the stage table separate this case.
* **Versions of one table (by design).** The original table and each fair copy are separate, unionable tables. More than
  one of them can appear in the same result `R` (e.g. an original and its copy for this query); each counts as a table
  and as a ground-truth hit in precision and recall. The original may not carry enough of the protected value to help
  reach `τ`, which is why the fairness target is reached through the copies.
* **M expansion.** The protected value comes from the query table, where a numeric column may read `4`, while datalake
  columns with missing values store `4.0`; `M` is `{v}` plus its numeric spelling variants (10 queries per benchmark).
  Its own effect was not isolated.

## 1. Results

@Q

**What it shows (α=10).** *Most queries get a fair answer:* feasible (k = 5/10/20) santos3 42/39/26 of 48, tusSmall3 89/84/65 of 92, tusLarge3 133/122/93 of 142. *Precision, recall, ΣU and F_R are means over all queries, a failed query counting 0:* precision@R is 0.88/0.81/0.52 (santos3), 0.93/0.89/0.68 (tusSmall3), 0.79/0.72/0.53 (tusLarge3) and falls with k mainly because more queries fail. *Recall stays below the ideal recall* (the ceiling, the same for every method): 0.18/0.30/0.34 vs 0.30/0.47/0.73 (santos3), 0.010/0.020/0.035 vs 0.011/0.021/0.042 (tusSmall3), 0.010/0.019/0.029 vs 0.012/0.025/0.050 (tusLarge3); the gap is widest where many queries fail (santos3 k=20: 26 of 48 feasible). *The fairness target is met where an answer exists:* among the feasible queries F_R ≥ 0.31 (τ = 0.30); the table's ΣU and F_R are means over all queries with a failed query counting 0. Example: tusSmall3, α=10, k=5: 89/92 feasible, precision 0.933, recall 0.0100 vs ideal 0.0105.

@S

**What it shows (α=10).** *santos3:* the failures are mostly too few candidates — `|D| < k` for 4 / 7 / 20 of the 6 / 9 / 22 infeasible queries at k = 5 / 10 / 20, the rest are LP-proven infeasible. Having ≥ k own copies is sufficient but not necessary: only 3 of 48 queries have ≥ 20 own copies, yet 26 are feasible at k = 20, because the original tables (kept in the datalake) and copies made for other queries can fill the k slots. Among queries with ≥ k own copies, feasible are 39/39 (k=5), 35/36 (k=10) and 3/3 (k=20). *TUS:* every query has ≥ k own copies for k ≤ 10, yet 3–20 queries per row are infeasible because retrieval did not bring enough of them into D (§4). *Stage 1* is bypassed for most santos3 queries and for most TUS queries at k=20 (77/92, 119/142); at k=5 it runs for most TUS queries (81/92, 102/142). The LP pre-check is decisive for 2–34 queries per row, each skipping a MILP call. Example: tusSmall3, α=10, k=20: 92 = 2 (|D|<k) + 25 (LP infeasible) + 65 (feasible). The optimizer cannot fix these failures: Stages 1 and 2 choose only inside D, and α only widens the pool inside D.

@P

**What it shows (α=10).** D = tables retrieved, P = the pool of α·k tables from Stage 1, R = the final k tables; an empty or missing set counts 0. *The retrieved set is mostly unionable:* precision@D 0.90 (santos3), 0.93 (tusSmall3), 0.75 (tusLarge3); recall@D 0.82 / 0.26 / 0.16. *Precision at P and R stays close to D where queries succeed and drops at large k because queries fail* (santos3 precision@R 0.88/0.81/0.52; only 26 of 48 queries have a result at k=20). *Recall falls from D to R because the sets shrink* (D holds ~25 / 120 / 90 tables, R holds k) and because failed queries count 0. Example: tusSmall3, α=10, k=5: precision 0.934 → 0.923 → 0.933 and recall 0.259 → 0.095 → 0.0100 at D → P → R. Precision and recall at D do not depend on k or α.

## 2. Caveats

* Precision, recall, ΣU and F_R count a failed query as 0, so they mix coverage and accuracy; the feasible count is shown beside them.
* Precision and recall are at table-name level; every copy of a unionable table counts as a hit.
* santos3's vectors were extracted with the same model as tusSmall3/tusLarge3; against the stored santos vectors of the 470 shared original tables the median per-table minimum cosine is 0.998 (9 tables below 0.99), the difference coming from a newer `transformers` version.
* 21 fair copies of tusSmall3/tusLarge3 (single-column tables of whitespace-only cells) lose blank rows when pandas re-reads them; their ratios are unaffected. 6 santos3 copies differ from their originals only by float printing precision.
* §4 comes from a separate run of the same code (its own HNSW build), so its `|D|` differs slightly from §1. HNSW builds are not bit-reproducible across rebuilds; each benchmark's index was built once and reused for all cells.

## 3. Findings

1. **Feasibility is high but not 100%.** On santos3 the infeasible queries mostly have fewer than k candidates in `D`; on TUS a feasible answer exists for nearly every query but retrieval returns too few of the qualifying copies (§4).
2. **Recall is bounded by k and by failed queries** (see the quality table).
3. **Stage bypasses are common:** Stage 1 is skipped whenever `α·k ≥ |D|`; the LP pre-check is decisive for 2–34 queries per row; the MILP never failed after a feasible LP.

## 4. Why some queries stay infeasible: own fair copies in `D`

| benchmark | mean `\\|D\\|` | own copies in `D` (mean) | own copies in the datalake (mean) | queries with 0 own copies in `D` | ratio of the query value over all of `D` | ratio over the own copies in `D` |
|---|---:|---:|---:|---:|---:|---:|
| santos3 | 25.5 | 9.9 | 11.0 | 0/48 | 0.372 | 0.485 |
| tusSmall3 | 122.4 | 17.1 | 85.9 | 0/92 | 0.152 | 0.396 |
| tusLarge3 | 83.8 | 13.6 | 100.0 | 9/142 | 0.173 | 0.386 |

On **santos3** retrieval brings almost all of a query's own copies into `D`, and the ratio over `D` (0.37) is above `τ = 0.3`.
On **TUS** only ~17 of ~86 (tusSmall3) and ~14 of 100 (tusLarge3) own copies reach `D`: copies of one table have nearly
identical vectors (Starmie reads only the first 1000 rows) and the `top_n` window fills with originals and with copies made
for *other* queries, where the query's value is rare. The ratio over all of `D` (0.15–0.17) is below `τ`, so feasibility
needs the few own copies to be picked. Raising `top_n` above 1000, or making retrieval prefer tables whose protected-value
share is high, are the levers; neither was applied here.
""".replace("@Q", Q).replace("@S", S).replace("@P", P)
open("/u6/bkassaie/DUTS/experiments/fair3-report.md", "w").write(md)
print("built")
