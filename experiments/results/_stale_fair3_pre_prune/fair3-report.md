# DUTS on the three fairified benchmarks — santos3, tusSmall3, tusLarge3

**2026-09-20** · conda `TableUnionNew`, py 3.8.5 · real pinned-match unionability (σ=0.6) · seed 42.
`F* = 0.4`, `δ = 0.1` (`τ = 0.3`), `include_query = True`. Grid: `k ∈ {5, 10, 20}` × `α ∈ {10, 5}`, `top_n = 1000`.
Code: `experiments/fair3/eval_fair3.py` (runner), `make_report.py` (tables), `diag_own_copies.py` (§4);
raw rows: `experiments/results/fair3_eval_<benchmark>_n1000.csv` (1,692 rows).

| | santos3 | tusSmall3 | tusLarge3 |
|---|---:|---:|---:|
| Fair queries evaluated | 48 | 92 | 142 |
| Datalake tables | 931 (550 orig + 381 fair) | 9,336 (1,430 + 7,906 copies) | 19,101 (4,894 + 14,207 copies) |
| Mean ground-truth size per query | 25.5 | 1,248.5 | 914.9 |
| Mean `\|D\|` (`top_n=1000`) | 21.6 | 122.5 | 96.7 |
| Queries whose protected value has a numeric spelling variant (M expanded) | 10/48 | 10/92 | 10/142 |

## 0. What is measured, and how

Per query and per `(k, α)` cell the pipeline is `retrieval → Stage 1 → score P → Stage 2`, composed exactly as
`dutsx.runner.run_query` does, except retrieval is timed and run once per `(query, top_n)` and reused (it does not
depend on `k, α`). Retrieval is `D = D_sem ∩ D_ovl` (§7.3); the synopsis is `MetadataStoreSynopsis` over each
benchmark's `metadata_combined.pkl` with `θ_cat = 50` (the same adapter for all three benchmarks, so the three are
comparable; the earlier santos3 report used `CsvSynopsis`).

* **precision@X / recall@X** (X = D, P, R): tables in X that are in the fair ground truth of the query / `|gt|`.
  **Ideal recall@k** = `min(k, |gt|) / |gt|`, the ceiling any method can reach at that k; it is reported next to
  recall, not divided into it. Ground truth is by table name; every copy of a unionable table counts.
* **feasible** = Stage 2 returned `k` tables with `F ≥ τ`. Precision, recall, ideal recall, ΣU and `F_R` are means
  **over the feasible queries of that cell** (infeasible queries have no R).
* **Bypassed stages** (all counted per cell, out of the queries of that benchmark):
  * `|D| < k` — `insufficient_candidates`: **both** Stage 1 and Stage 2 are bypassed.
  * **S1 bypassed (P=D)** — `α·k ≥ |D|` (C5 clamp): Stage 1 is skipped and `P = D`; Stage 2 still runs.
  * **LP decisive (MILP bypassed)** — the Stage 2 LP pre-check proved the instance infeasible, so the MILP was skipped
    (the query is then infeasible). **MILP ran** — the LP was feasible and the MILP solved it.
  * `MILP-infeasible` (LP feasible, MILP infeasible) was **0** in every cell of every benchmark.
* **M expansion.** The protected value comes from the query table, where a numeric column may read `4`, while
  datalake columns with missing values store the same value as `4.0`; `M` is therefore `{v}` plus its numeric
  spelling variants. This touches 10 queries per benchmark. Its own effect was not isolated (no run without it).

## 1. Results

**top_n = 1000 -- Quality of the fair result R (means over feasible queries)**

| bench | α | k | feasible | precision@R | recall@R | ideal recall | mean ΣU | mean F_R |
|---|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 10 | 5 | 42/48 | 0.995 | 0.2053 | 0.2061 | 58.7 | 0.450 |
| santos3 | 10 | 10 | 33/48 | 0.988 | 0.3798 | 0.3836 | 108.9 | 0.451 |
| santos3 | 10 | 20 | 22/48 | 0.952 | 0.7126 | 0.7453 | 199.5 | 0.467 |
| santos3 | 5 | 5 | 42/48 | 0.967 | 0.2007 | 0.2061 | 57.3 | 0.471 |
| santos3 | 5 | 10 | 33/48 | 0.988 | 0.3798 | 0.3836 | 108.8 | 0.460 |
| santos3 | 5 | 20 | 22/48 | 0.952 | 0.7126 | 0.7453 | 199.5 | 0.467 |
| tusSmall3 | 10 | 5 | 90/92 | 0.998 | 0.0079 | 0.0079 | 48.4 | 0.374 |
| tusSmall3 | 10 | 10 | 85/92 | 0.996 | 0.0163 | 0.0165 | 92.6 | 0.326 |
| tusSmall3 | 10 | 20 | 63/92 | 0.996 | 0.0393 | 0.0397 | 158.1 | 0.314 |
| tusSmall3 | 5 | 5 | 90/92 | 0.989 | 0.0077 | 0.0079 | 48.0 | 0.406 |
| tusSmall3 | 5 | 10 | 85/92 | 0.993 | 0.0162 | 0.0165 | 92.0 | 0.369 |
| tusSmall3 | 5 | 20 | 63/92 | 0.990 | 0.0388 | 0.0397 | 157.5 | 0.331 |
| tusLarge3 | 10 | 5 | 135/142 | 0.874 | 0.0075 | 0.0087 | 52.4 | 0.369 |
| tusLarge3 | 10 | 10 | 126/142 | 0.882 | 0.0155 | 0.0179 | 103.0 | 0.338 |
| tusLarge3 | 10 | 20 | 96/142 | 0.826 | 0.0301 | 0.0376 | 186.4 | 0.322 |
| tusLarge3 | 5 | 5 | 135/142 | 0.871 | 0.0074 | 0.0087 | 51.7 | 0.405 |
| tusLarge3 | 5 | 10 | 126/142 | 0.861 | 0.0149 | 0.0179 | 102.3 | 0.360 |
| tusLarge3 | 5 | 20 | 96/142 | 0.820 | 0.0295 | 0.0376 | 184.8 | 0.339 |

**How to read it.** *feasible x/n*: for x of the n queries the pipeline returned k tables whose union with the query has at least 30% of its rows carrying the protected value (F\*=0.4, δ=0.1). *precision@R*: share of the k returned tables that are truly unionable. *recall@R*: returned tables in the ground truth / |gt|; *ideal recall* = min(k,|gt|)/|gt| is the best possible when only k tables are returned. ΣU: total unionability of the k tables; F_R: share of rows with the protected value in result + query (must be ≥ 0.3). All values are means over the feasible queries of that row. Example: tusSmall3, α=10, k=5: 90/92 feasible, precision 0.998, recall 0.0079 = ideal recall 0.0079; recall is tiny only because each query has ~1,250 unionable tables and 5 are returned.

**top_n = 1000 -- Stage bypasses, LP pre-check and feasibility (counts of queries)**

| bench | α | k | mean |D| | |D|<k (S1+S2 bypassed) | S1 bypassed (P=D) | S1 ran | LP decisive (MILP bypassed) | MILP ran | feasible |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| santos3 | 10 | 5 | 21.6 | 3 | 44 | 1 | 3 | 42 | 42/48 |
| santos3 | 10 | 10 | 21.6 | 6 | 42 | 0 | 9 | 33 | 33/48 |
| santos3 | 10 | 20 | 21.6 | 21 | 27 | 0 | 5 | 22 | 22/48 |
| santos3 | 5 | 5 | 21.6 | 3 | 32 | 13 | 3 | 42 | 42/48 |
| santos3 | 5 | 10 | 21.6 | 6 | 41 | 1 | 9 | 33 | 33/48 |
| santos3 | 5 | 20 | 21.6 | 21 | 27 | 0 | 5 | 22 | 22/48 |
| tusSmall3 | 10 | 5 | 122.5 | 0 | 10 | 82 | 2 | 90 | 90/92 |
| tusSmall3 | 10 | 10 | 122.5 | 0 | 44 | 48 | 7 | 85 | 85/92 |
| tusSmall3 | 10 | 20 | 122.5 | 2 | 76 | 14 | 27 | 63 | 63/92 |
| tusSmall3 | 5 | 5 | 122.5 | 0 | 3 | 89 | 2 | 90 | 90/92 |
| tusSmall3 | 5 | 10 | 122.5 | 0 | 10 | 82 | 7 | 85 | 85/92 |
| tusSmall3 | 5 | 20 | 122.5 | 2 | 42 | 48 | 27 | 63 | 63/92 |
| tusLarge3 | 10 | 5 | 96.7 | 4 | 34 | 104 | 3 | 135 | 135/142 |
| tusLarge3 | 10 | 10 | 96.7 | 6 | 80 | 56 | 10 | 126 | 126/142 |
| tusLarge3 | 10 | 20 | 96.7 | 10 | 120 | 12 | 36 | 96 | 96/142 |
| tusLarge3 | 5 | 5 | 96.7 | 4 | 10 | 128 | 3 | 135 | 135/142 |
| tusLarge3 | 5 | 10 | 96.7 | 6 | 32 | 104 | 10 | 126 | 126/142 |
| tusLarge3 | 5 | 20 | 96.7 | 10 | 76 | 56 | 36 | 96 | 96/142 |

**How to read it.** Every query ends in exactly one of three outcomes: *|D| < k* (retrieval returned fewer than k tables; Stage 1 and Stage 2 both skipped, infeasible), *LP decisive* (the Stage 2 LP pre-check proves no k tables reach F ≥ 0.3; MILP skipped, infeasible) or *MILP ran* (feasible; the MILP never failed after a feasible LP). Separately, *S1 bypassed* means α·k ≥ |D|, so the pool is all of D and Stage 1 is not needed (Stage 2 still runs); *S1 ran* means it picked α·k tables out of D. Example: tusSmall3, α=10, k=20: 92 = 2 (|D|<k) + 27 (LP infeasible) + 63 (feasible); of the 90 that reach Stage 1, it was bypassed for 76 and ran for 14. The optimizer cannot fix these failures: Stages 1 and 2 only choose within D, and α only widens the pool inside D.

**top_n = 1000 -- Precision and recall at D, P and R**

| bench | α | k | prec@D | rec@D | prec@P | rec@P | prec@R | rec@R |
|---|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 10 | 5 | 0.932 | 0.7272 | 0.926 | 0.7581 | 0.995 | 0.2053 |
| santos3 | 10 | 10 | 0.932 | 0.7272 | 0.922 | 0.7899 | 0.988 | 0.3798 |
| santos3 | 10 | 20 | 0.932 | 0.7272 | 0.914 | 0.9195 | 0.952 | 0.7126 |
| santos3 | 5 | 5 | 0.932 | 0.7272 | 0.913 | 0.6824 | 0.967 | 0.2007 |
| santos3 | 5 | 10 | 0.932 | 0.7272 | 0.921 | 0.7808 | 0.988 | 0.3798 |
| santos3 | 5 | 20 | 0.932 | 0.7272 | 0.914 | 0.9195 | 0.952 | 0.7126 |
| tusSmall3 | 10 | 5 | 0.990 | 0.2037 | 0.991 | 0.0749 | 0.998 | 0.0079 |
| tusSmall3 | 10 | 10 | 0.990 | 0.2037 | 0.990 | 0.1373 | 0.996 | 0.0163 |
| tusSmall3 | 10 | 20 | 0.990 | 0.2037 | 0.990 | 0.1912 | 0.996 | 0.0393 |
| tusSmall3 | 5 | 5 | 0.990 | 0.2037 | 0.989 | 0.0378 | 0.989 | 0.0077 |
| tusSmall3 | 5 | 10 | 0.990 | 0.2037 | 0.991 | 0.0749 | 0.993 | 0.0162 |
| tusSmall3 | 5 | 20 | 0.990 | 0.2037 | 0.990 | 0.1402 | 0.990 | 0.0388 |
| tusLarge3 | 10 | 5 | 0.812 | 0.1234 | 0.804 | 0.0580 | 0.874 | 0.0075 |
| tusLarge3 | 10 | 10 | 0.812 | 0.1234 | 0.804 | 0.0963 | 0.882 | 0.0155 |
| tusLarge3 | 10 | 20 | 0.812 | 0.1234 | 0.798 | 0.1191 | 0.826 | 0.0301 |
| tusLarge3 | 5 | 5 | 0.812 | 0.1234 | 0.824 | 0.0329 | 0.871 | 0.0074 |
| tusLarge3 | 5 | 10 | 0.812 | 0.1234 | 0.804 | 0.0587 | 0.861 | 0.0149 |
| tusLarge3 | 5 | 20 | 0.812 | 0.1234 | 0.798 | 0.0982 | 0.820 | 0.0295 |

**How to read it.** D = tables retrieved, P = the pool of α·k tables from Stage 1, R = the final k tables. Precision stays high at every step (almost everything retrieved is truly unionable). Recall falls from D to R only because the sets shrink (D ≈ 100 tables, P = α·k, R = k, against a much larger ground truth); precision and recall at D do not depend on k or α. Example: tusSmall3, α=10, k=5: precision 0.990 → 0.991 → 0.998 and recall 0.204 → 0.075 → 0.0079 at D → P → R.

## 2. Findings

1. **Feasible queries** (α=10, k = 5/10/20): santos3 42/33/22 of 48; tusSmall3 90/85/63 of 92; tusLarge3 135/126/96 of 142. Not 100%, although a feasible answer exists in every TUS datalake (each query has 16–145 (tusSmall3) or 27–248 (tusLarge3) fair copies with ratio ≥ 0.4 for its value): retrieval returns too few of them (§4). The MILP never failed; failures are `|D| < k` or an LP-proven infeasibility.
2. **Stage bypasses.** Stage 1 is bypassed (`P=D`) for most santos3 queries (`|D|` ≈ 22 is below `α·k`); on the TUS benchmarks it runs for most queries at k=5 (82/92 and 104/142) and is bypassed for most at k=20 (76/92 and 120/142). The LP pre-check is decisive for 2–36 queries per row (α=10; tusSmall3: 2, 7, 27 and tusLarge3: 3, 10, 36 at k = 5/10/20), each of which skips a MILP call.
3. **Precision is high, recall is bounded by k.** `precision@R` is 0.95–1.00 on santos3, 0.99–1.00 on tusSmall3 and 0.82–0.88 on tusLarge3 (its `precision@D` is only 0.81). `recall@R` is within 5% of ideal recall on santos3 and tusSmall3 (e.g. tusSmall3 k=20: 0.0393 vs 0.0397), up to ~20% below on tusLarge3 (0.0301 vs 0.0376), where precision is also lower. On the TUS benchmarks recall is small in absolute terms because the ground truth holds ~900–1,250 tables per query.

## 3. Caveats

* Means are over feasible queries, so rows with few feasible queries (e.g. santos3 k=20: 22 of 48) are noisier.
* santos3 shares fair tables between queries (one fair copy per table, built by rebalancing for each query in turn): 384 of the 841 tables in its ground truth are unionable with two queries (max 2; 181 of its 381 fair tables). Only the last query processed for a shared table is guaranteed a ratio ≥ 0.4, so santos3 is built differently from tusSmall3/tusLarge3 (one copy per query–table pair).
* Precision and recall are at table-name level and count every copy of a unionable table as a hit (see the
  ground-truth construction in `starmie_fair/data/tus*3/`).
* Ideal recall is averaged over the same feasible queries as recall, so the two are comparable cell by cell.
* §4 comes from a separate run of the same code (its own HNSW build), so its `|D|` differs slightly from §1 (e.g. 18.1 vs 18.0).
* HNSW builds are not bit-reproducible across rebuilds (multi-threaded insertion); each benchmark's index was built
  once and reused for both `top_n` values and all cells.

## 4. Why not every query is feasible: own fair copies in `D`

For each query the datalake holds ~86 (tusSmall3) or ~100 (tusLarge3) fair copies built for it (ratio ≥ 0.4 for its
value). `D` (at `top_n=1000`) holds only ~17 of them, because copies of one table have nearly identical vectors and the `top_n`
window is filled with originals and with copies made for *other* queries, where the query's value is rare:

| benchmark | mean `\|D\|` | own copies in `D` (mean) | queries with 0 own copies in `D` | ratio of the query value over all of `D` | ratio over the own copies in `D` |
|---|---:|---:|---:|---:|---:|
| tusSmall3 | 122.4 | 17.1 | 0/92 | 0.151 | 0.395 |
| tusLarge3 | 94.8 | 15.7 | 4/142 | 0.179 | 0.380 |

`τ = 0.3` is above the ratio over all of `D` (0.15–0.18), so feasibility needs the solver to pick the few own copies;
it fails when too few are in `D`. Raising `top_n` above 1000, or making retrieval prefer tables whose protected-value
share is high, are the levers; neither was applied here.
