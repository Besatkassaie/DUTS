# DUTS on the three fairified benchmarks — santos3, tusSmall3, tusLarge3

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
| Mean `\|D\|` | 25.5 | 122.5 | 90.5 |
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

**top_n = 1000 -- Quality of the fair result R (all queries; a failed query counts 0 for precision, recall, ΣU and F_R)**

| bench | α | k | feasible | precision@R | recall@R | ideal recall | mean ΣU | mean F_R |
|---|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 10 | 5 | 42/48 | 0.875 | 0.1769 | 0.2984 | 52.8 | 0.388 |
| santos3 | 10 | 10 | 39/48 | 0.810 | 0.3005 | 0.4736 | 96.0 | 0.355 |
| santos3 | 10 | 20 | 26/48 | 0.520 | 0.3352 | 0.7299 | 111.7 | 0.239 |
| santos3 | 5 | 5 | 42/48 | 0.846 | 0.1723 | 0.2984 | 51.2 | 0.405 |
| santos3 | 5 | 10 | 39/48 | 0.810 | 0.3005 | 0.4736 | 95.9 | 0.360 |
| santos3 | 5 | 20 | 26/48 | 0.520 | 0.3352 | 0.7299 | 111.7 | 0.239 |
| tusSmall3 | 10 | 5 | 89/92 | 0.933 | 0.0100 | 0.0105 | 46.3 | 0.364 |
| tusSmall3 | 10 | 10 | 84/92 | 0.890 | 0.0197 | 0.0211 | 85.6 | 0.297 |
| tusSmall3 | 10 | 20 | 65/92 | 0.678 | 0.0347 | 0.0421 | 114.1 | 0.221 |
| tusSmall3 | 5 | 5 | 89/92 | 0.920 | 0.0098 | 0.0105 | 46.0 | 0.393 |
| tusSmall3 | 5 | 10 | 84/92 | 0.864 | 0.0192 | 0.0211 | 85.1 | 0.338 |
| tusSmall3 | 5 | 20 | 65/92 | 0.665 | 0.0339 | 0.0421 | 113.6 | 0.232 |
| tusLarge3 | 10 | 5 | 133/142 | 0.790 | 0.0101 | 0.0125 | 50.3 | 0.348 |
| tusLarge3 | 10 | 10 | 122/142 | 0.715 | 0.0190 | 0.0249 | 88.9 | 0.288 |
| tusLarge3 | 10 | 20 | 93/142 | 0.526 | 0.0291 | 0.0498 | 121.5 | 0.208 |
| tusLarge3 | 5 | 5 | 133/142 | 0.782 | 0.0099 | 0.0125 | 49.8 | 0.375 |
| tusLarge3 | 5 | 10 | 122/142 | 0.704 | 0.0184 | 0.0249 | 88.2 | 0.308 |
| tusLarge3 | 5 | 20 | 93/142 | 0.518 | 0.0283 | 0.0498 | 120.5 | 0.218 |

**What it shows (α=10).** *Most queries get a fair answer:* feasible (k = 5/10/20) santos3 42/39/26 of 48, tusSmall3 89/84/65 of 92, tusLarge3 133/122/93 of 142. *Precision, recall, ΣU and F_R are means over all queries, a failed query counting 0:* precision@R is 0.88/0.81/0.52 (santos3), 0.93/0.89/0.68 (tusSmall3), 0.79/0.72/0.53 (tusLarge3) and falls with k mainly because more queries fail. *Recall stays below the ideal recall* (the ceiling, the same for every method): 0.18/0.30/0.34 vs 0.30/0.47/0.73 (santos3), 0.010/0.020/0.035 vs 0.011/0.021/0.042 (tusSmall3), 0.010/0.019/0.029 vs 0.012/0.025/0.050 (tusLarge3); the gap is widest where many queries fail (santos3 k=20: 26 of 48 feasible). *The fairness target is met where an answer exists:* among the feasible queries F_R ≥ 0.31 (τ = 0.30); the table's ΣU and F_R are means over all queries with a failed query counting 0. Example: tusSmall3, α=10, k=5: 89/92 feasible, precision 0.933, recall 0.0100 vs ideal 0.0105.

**top_n = 1000 -- Stage bypasses, LP pre-check and feasibility (counts of queries)**

| bench | α | k | mean |D| | |D|<k (S1+S2 bypassed) | S1 bypassed (P=D) | S1 ran | LP decisive (MILP bypassed) | MILP ran | feasible | queries with ≥k own copies | feasible among them |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| santos3 | 10 | 5 | 25.5 | 4 | 40 | 4 | 2 | 42 | 42/48 | 39 | 39/39 |
| santos3 | 10 | 10 | 25.5 | 7 | 41 | 0 | 2 | 39 | 39/48 | 36 | 35/36 |
| santos3 | 10 | 20 | 25.5 | 20 | 28 | 0 | 2 | 26 | 26/48 | 3 | 3/3 |
| santos3 | 5 | 5 | 25.5 | 4 | 27 | 17 | 2 | 42 | 42/48 | 39 | 39/39 |
| santos3 | 5 | 10 | 25.5 | 7 | 37 | 4 | 2 | 39 | 39/48 | 36 | 35/36 |
| santos3 | 5 | 20 | 25.5 | 20 | 28 | 0 | 2 | 26 | 26/48 | 3 | 3/3 |
| tusSmall3 | 10 | 5 | 122.5 | 0 | 11 | 81 | 3 | 89 | 89/92 | 92 | 89/92 |
| tusSmall3 | 10 | 10 | 122.5 | 1 | 42 | 49 | 7 | 84 | 84/92 | 92 | 84/92 |
| tusSmall3 | 10 | 20 | 122.5 | 2 | 77 | 13 | 25 | 65 | 65/92 | 85 | 58/85 |
| tusSmall3 | 5 | 5 | 122.5 | 0 | 3 | 89 | 3 | 89 | 89/92 | 92 | 89/92 |
| tusSmall3 | 5 | 10 | 122.5 | 1 | 10 | 81 | 7 | 84 | 84/92 | 92 | 84/92 |
| tusSmall3 | 5 | 20 | 122.5 | 2 | 41 | 49 | 25 | 65 | 65/92 | 85 | 58/85 |
| tusLarge3 | 10 | 5 | 90.5 | 6 | 34 | 102 | 3 | 133 | 133/142 | 142 | 133/142 |
| tusLarge3 | 10 | 10 | 90.5 | 10 | 76 | 56 | 10 | 122 | 122/142 | 142 | 122/142 |
| tusLarge3 | 10 | 20 | 90.5 | 15 | 119 | 8 | 34 | 93 | 93/142 | 142 | 93/142 |
| tusLarge3 | 5 | 5 | 90.5 | 6 | 10 | 126 | 3 | 133 | 133/142 | 142 | 133/142 |
| tusLarge3 | 5 | 10 | 90.5 | 10 | 30 | 102 | 10 | 122 | 122/142 | 142 | 122/142 |
| tusLarge3 | 5 | 20 | 90.5 | 15 | 71 | 56 | 34 | 93 | 93/142 | 142 | 93/142 |

**What it shows (α=10).** *santos3:* the failures are mostly too few candidates — `|D| < k` for 4 / 7 / 20 of the 6 / 9 / 22 infeasible queries at k = 5 / 10 / 20, the rest are LP-proven infeasible. Having ≥ k own copies is sufficient but not necessary: only 3 of 48 queries have ≥ 20 own copies, yet 26 are feasible at k = 20, because the original tables (kept in the datalake) and copies made for other queries can fill the k slots. Among queries with ≥ k own copies, feasible are 39/39 (k=5), 35/36 (k=10) and 3/3 (k=20). *TUS:* every query has ≥ k own copies for k ≤ 10, yet 3–20 queries per row are infeasible because retrieval did not bring enough of them into D (§4). *Stage 1* is bypassed for most santos3 queries and for most TUS queries at k=20 (77/92, 119/142); at k=5 it runs for most TUS queries (81/92, 102/142). The LP pre-check is decisive for 2–34 queries per row, each skipping a MILP call. Example: tusSmall3, α=10, k=20: 92 = 2 (|D|<k) + 25 (LP infeasible) + 65 (feasible). The optimizer cannot fix these failures: Stages 1 and 2 choose only inside D, and α only widens the pool inside D.

**top_n = 1000 -- Precision and recall at D, P and R (all queries; an empty or missing set counts 0)**

| bench | α | k | prec@D | rec@D | prec@P | rec@P | prec@R | rec@R |
|---|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 10 | 5 | 0.899 | 0.8216 | 0.813 | 0.7266 | 0.875 | 0.1769 |
| santos3 | 10 | 10 | 0.899 | 0.8216 | 0.769 | 0.6910 | 0.810 | 0.3005 |
| santos3 | 10 | 20 | 0.899 | 0.8216 | 0.516 | 0.5233 | 0.520 | 0.3352 |
| santos3 | 5 | 5 | 0.899 | 0.8216 | 0.799 | 0.6100 | 0.846 | 0.1723 |
| santos3 | 5 | 10 | 0.899 | 0.8216 | 0.766 | 0.6724 | 0.810 | 0.3005 |
| santos3 | 5 | 20 | 0.899 | 0.8216 | 0.516 | 0.5233 | 0.520 | 0.3352 |
| tusSmall3 | 10 | 5 | 0.934 | 0.2585 | 0.923 | 0.0953 | 0.933 | 0.0100 |
| tusSmall3 | 10 | 10 | 0.934 | 0.2585 | 0.916 | 0.1729 | 0.890 | 0.0197 |
| tusSmall3 | 10 | 20 | 0.934 | 0.2585 | 0.912 | 0.2352 | 0.678 | 0.0347 |
| tusSmall3 | 5 | 5 | 0.934 | 0.2585 | 0.928 | 0.0487 | 0.920 | 0.0098 |
| tusSmall3 | 5 | 10 | 0.934 | 0.2585 | 0.913 | 0.0952 | 0.864 | 0.0192 |
| tusSmall3 | 5 | 20 | 0.934 | 0.2585 | 0.905 | 0.1728 | 0.665 | 0.0339 |
| tusLarge3 | 10 | 5 | 0.746 | 0.1570 | 0.723 | 0.0767 | 0.790 | 0.0101 |
| tusLarge3 | 10 | 10 | 0.746 | 0.1570 | 0.694 | 0.1226 | 0.715 | 0.0190 |
| tusLarge3 | 10 | 20 | 0.746 | 0.1570 | 0.657 | 0.1456 | 0.526 | 0.0291 |
| tusLarge3 | 5 | 5 | 0.746 | 0.1570 | 0.744 | 0.0444 | 0.782 | 0.0099 |
| tusLarge3 | 5 | 10 | 0.746 | 0.1570 | 0.696 | 0.0764 | 0.704 | 0.0184 |
| tusLarge3 | 5 | 20 | 0.746 | 0.1570 | 0.659 | 0.1211 | 0.518 | 0.0283 |

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

| benchmark | mean `\|D\|` | own copies in `D` (mean) | own copies in the datalake (mean) | queries with 0 own copies in `D` | ratio of the query value over all of `D` | ratio over the own copies in `D` |
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
