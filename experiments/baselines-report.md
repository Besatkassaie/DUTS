# Starmie baselines on the three fairified benchmarks

**2026-09-21** · Starmie code: `/u6/bkassaie/starmie_fair` (`HNSWSearcher_Fair`, imported unmodified except for the changes listed in §1) ·
driver `experiments/baselines/run_starmie_baselines.py`, tables `make_baseline_report.py` / `build_docs.py` · raw rows
`experiments/results/baselines_<benchmark>_<approach>.csv` (2,538 rows: one per query and k).
Same fair queries (santos3 48, tusSmall3 92, tusLarge3 142), fair datalake and fair ground truth as `fair3-report.md`.
Fixed for all three approaches: **N = 1000** nearest columns per query column, **σ = 0.6**, **F\* = 0.4, δ = 0.1 (τ = 0.3)**,
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

**Pure Starmie (no fairification) -- results**

| bench | k | reached τ | precision | recall | ideal recall | mean ΣU | mean F |
|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 5 | 21/48 | 0.888 | 0.2772 | 0.2984 | 60.0 | 0.337 |
| santos3 | 10 | 12/48 | 0.829 | 0.4216 | 0.4736 | 115.7 | 0.269 |
| santos3 | 20 | 9/48 | 0.752 | 0.6516 | 0.7299 | 220.0 | 0.206 |
| tusSmall3 | 5 | 7/92 | 0.900 | 0.0098 | 0.0105 | 50.2 | 0.161 |
| tusSmall3 | 10 | 5/92 | 0.886 | 0.0195 | 0.0211 | 100.1 | 0.126 |
| tusSmall3 | 20 | 1/92 | 0.843 | 0.0358 | 0.0421 | 198.5 | 0.094 |
| tusLarge3 | 5 | 22/142 | 0.792 | 0.0099 | 0.0125 | 55.9 | 0.183 |
| tusLarge3 | 10 | 13/142 | 0.769 | 0.0191 | 0.0249 | 111.0 | 0.143 |
| tusLarge3 | 20 | 8/142 | 0.713 | 0.0348 | 0.0498 | 219.4 | 0.108 |

**What it shows.** *The unfair top-k rarely meets the fairness target:* it has F ≥ 0.3 for only santos3 21/12/9 of 48, tusSmall3 7/5/1 of 92 and tusLarge3 22/13/8 of 142 queries (k = 5/10/20). *It gets worse as k grows:* mean F over all queries is 0.34/0.27/0.21 (santos3), 0.16/0.13/0.09 (tusSmall3), 0.18/0.14/0.11 (tusLarge3). *Precision over all queries:* 0.89/0.83/0.75, 0.90/0.89/0.84 and 0.79/0.77/0.71. *Recall vs ideal recall:* santos3 0.28/0.42/0.65 vs 0.30/0.47/0.73; tusSmall3 0.010/0.019/0.036 vs 0.011/0.021/0.042; tusLarge3 0.010/0.019/0.035 vs 0.012/0.025/0.050. These are the accuracy of Starmie's own, mostly unfair, lists; the swap methods are zeroed when they fail, so their numbers are lower by construction.

**Pure Starmie (no fairification) -- algorithm details**

| bench | k | mean candidates scored | queries with < k returned |
|---|---|---:|---:|
| santos3 | 5 | 105.8 | 0 |
| santos3 | 10 | 105.8 | 0 |
| santos3 | 20 | 105.8 | 0 |
| tusSmall3 | 5 | 144.4 | 0 |
| tusSmall3 | 10 | 144.4 | 0 |
| tusSmall3 | 20 | 144.4 | 0 |
| tusLarge3 | 5 | 126.5 | 0 |
| tusLarge3 | 10 | 126.5 | 0 |
| tusLarge3 | 20 | 126.5 | 0 |

*cand.* is the mean number of tables verified after the N-nearest-column lookup, before taking the top-k; no swap is done, so there is no swap information.

## 3. Exhaustive swap (no filtering)

**Exhaustive swap (no filtering) -- results**

| bench | k | feasible | precision | recall | ideal recall | mean ΣU | mean F_R |
|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 5 | 47/48 | 0.875 | 0.2502 | 0.2984 | 55.7 | 0.397 |
| santos3 | 10 | 42/48 | 0.704 | 0.3113 | 0.4736 | 93.9 | 0.338 |
| santos3 | 20 | 31/48 | 0.441 | 0.3340 | 0.7299 | 106.4 | 0.242 |
| tusSmall3 | 5 | 66/92 | 0.676 | 0.0085 | 0.0105 | 29.6 | 0.239 |
| tusSmall3 | 10 | 43/92 | 0.429 | 0.0135 | 0.0211 | 32.8 | 0.151 |
| tusSmall3 | 20 | 24/92 | 0.227 | 0.0202 | 0.0421 | 29.4 | 0.081 |
| tusLarge3 | 5 | 108/142 | 0.628 | 0.0086 | 0.0125 | 37.4 | 0.264 |
| tusLarge3 | 10 | 77/142 | 0.399 | 0.0129 | 0.0249 | 44.8 | 0.181 |
| tusLarge3 | 20 | 49/142 | 0.205 | 0.0166 | 0.0498 | 44.7 | 0.111 |

**What it shows.** *Swapping makes many more queries feasible* than pure Starmie reaches — santos3 47/42/31 of 48, tusSmall3 66/43/24 of 92, tusLarge3 108/77/49 of 142 — but the number falls quickly as k grows. *Precision and recall over all queries (failed = 0) fall with k for that reason:* precision 0.88/0.70/0.44 (santos3), 0.68/0.43/0.23 (tusSmall3), 0.63/0.40/0.21 (tusLarge3); recall 0.25/0.31/0.33, 0.0085/0.0135/0.0202, 0.0086/0.0129/0.0166 against ideal recall 0.30/0.47/0.73, 0.011/0.021/0.042, 0.012/0.025/0.050. *Almost every query needs the swap:* the initial top-k violates the target for 28/36/39 of 48 (santos3), 85/87/90 of 92 (tusSmall3), 122/129/134 of 142 (tusLarge3); it succeeds for e.g. 27/28, 30/36, 22/39 on santos3 and 59/85, 38/87, 22/90 on tusSmall3. *When it fails it returns nothing* (1/6/17 empty results on santos3, 26/49/68 on tusSmall3, 34/65/93 on tusLarge3), and *it stops as soon as the target is met:* among the feasible queries mean F_R is only just above 0.3 (0.31–0.41); the table's ΣU and F_R include the failed queries as 0.

**Exhaustive swap (no filtering) -- algorithm details**

| bench | k | mean candidates scored | queries with < k returned | needed fairification | swap succeeded (of needed) | mean swaps (of needed) | mean F before swap (of needed) |
|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 5 | 106.2 | 1 | 28/48 | 27/28 | 1.6 | 0.221 |
| santos3 | 10 | 106.2 | 6 | 36/48 | 30/36 | 3.4 | 0.170 |
| santos3 | 20 | 106.2 | 17 | 39/48 | 22/39 | 5.3 | 0.131 |
| tusSmall3 | 5 | 144.9 | 26 | 85/92 | 59/85 | 2.4 | 0.144 |
| tusSmall3 | 10 | 144.9 | 49 | 87/92 | 38/87 | 3.6 | 0.115 |
| tusSmall3 | 20 | 144.9 | 68 | 90/92 | 22/90 | 3.7 | 0.089 |
| tusLarge3 | 5 | 125.5 | 34 | 122/142 | 88/122 | 2.2 | 0.150 |
| tusLarge3 | 10 | 125.5 | 65 | 129/142 | 64/129 | 3.4 | 0.120 |
| tusLarge3 | 20 | 125.5 | 93 | 134/142 | 41/134 | 3.6 | 0.093 |

*needed*: the initial top-k had F < τ so swaps ran; *succeeded*: swaps brought F to ≥ τ (of those that needed it); *swaps*: mean swaps over the queries that needed fairification (failed ones included); *queries with < k returned*: the swap failed and the empty list was returned.

## 4. Nested-loop swap (winnow filtering)

**Nested-loop swap (winnow filtering) -- results**

| bench | k | feasible | precision | recall | ideal recall | mean ΣU | mean F_R |
|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 5 | 47/48 | 0.846 | 0.2277 | 0.2984 | 55.2 | 0.392 |
| santos3 | 10 | 35/48 | 0.612 | 0.2495 | 0.4736 | 83.3 | 0.292 |
| santos3 | 20 | 20/48 | 0.317 | 0.2348 | 0.7299 | 68.9 | 0.170 |
| tusSmall3 | 5 | 52/92 | 0.528 | 0.0076 | 0.0105 | 20.1 | 0.189 |
| tusSmall3 | 10 | 33/92 | 0.320 | 0.0116 | 0.0211 | 22.3 | 0.114 |
| tusSmall3 | 20 | 12/92 | 0.105 | 0.0095 | 0.0421 | 13.7 | 0.040 |
| tusLarge3 | 5 | 101/142 | 0.569 | 0.0083 | 0.0125 | 32.1 | 0.245 |
| tusLarge3 | 10 | 59/142 | 0.284 | 0.0109 | 0.0249 | 28.7 | 0.141 |
| tusLarge3 | 20 | 39/142 | 0.150 | 0.0134 | 0.0498 | 32.2 | 0.090 |

**What it shows.** *Fewer feasible queries than the exhaustive swap* in every row except santos3 at k = 5 — santos3 47/35/20 of 48, tusSmall3 52/33/12 of 92, tusLarge3 101/59/39 of 142. *Precision and recall over all queries (failed = 0) are correspondingly lower:* precision 0.85/0.61/0.32 (santos3), 0.53/0.32/0.11 (tusSmall3), 0.57/0.28/0.15 (tusLarge3); recall 0.23/0.25/0.23, 0.0076/0.0116/0.0095, 0.0083/0.0109/0.0134. *The winnow filter removes 89–96% of the swap candidates* (e.g. tusSmall3, k=5: 10,828 of 11,306), leaving on average only 2–6 candidates per query — fewer chances to find a replacement that reaches τ. *It always returns k tables, even when the swap failed;* those unfair lists are discarded here (counted 0). Among the feasible queries mean F_R is 0.31–0.41.

**Nested-loop swap (winnow filtering) -- algorithm details**

| bench | k | mean candidates scored | queries with < k returned | needed fairification | swap succeeded (of needed) | mean swaps (of needed) | mean F before swap (of needed) | dominated removed / pool | removed share | mean candidates after winnow |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| santos3 | 5 | 106.1 | 0 | 28/48 | 27/28 | 1.8 | 0.221 | 1876/2102 | 0.892 | 6.2 |
| santos3 | 10 | 106.1 | 0 | 36/48 | 23/36 | 3.2 | 0.170 | 2406/2673 | 0.900 | 4.2 |
| santos3 | 20 | 106.1 | 0 | 39/48 | 11/39 | 3.2 | 0.131 | 2330/2593 | 0.899 | 3.5 |
| tusSmall3 | 5 | 144.7 | 0 | 85/92 | 45/85 | 2.2 | 0.145 | 10828/11306 | 0.958 | 3.4 |
| tusSmall3 | 10 | 144.7 | 0 | 87/92 | 28/87 | 2.8 | 0.114 | 10709/11184 | 0.958 | 2.7 |
| tusSmall3 | 20 | 144.7 | 0 | 90/92 | 10/90 | 3.0 | 0.088 | 10619/11077 | 0.959 | 2.1 |
| tusLarge3 | 5 | 125.1 | 0 | 121/142 | 80/121 | 2.2 | 0.149 | 12042/13031 | 0.924 | 6.0 |
| tusLarge3 | 10 | 125.1 | 0 | 128/142 | 45/128 | 3.2 | 0.122 | 12371/13443 | 0.920 | 5.2 |
| tusLarge3 | 20 | 125.1 | 0 | 134/142 | 31/134 | 3.3 | 0.094 | 12585/13694 | 0.919 | 5.0 |

Columns as for the exhaustive swap, plus the winnow filter: *removed / pool* = candidates dropped as dominated out of those considered, summed over queries; *share* = their ratio; *left* = mean candidates left for swapping. The nested-loop swap returns its last (possibly unfair) state on failure, so it never returns fewer than k.

## 5. Caveats

* Precision and recall count a failed query as 0 for the swap methods but not for pure Starmie (scored regardless of τ), so pure Starmie's numbers count unfair lists and are not directly comparable with the swap methods'.
* The three approaches differ in what they return on failure (exhaustive: nothing; nested-loop: its last unfair state; pure Starmie: the unfair top-k).
* N (nearest columns per query column) is not the same quantity as DUTS's `top_n` (nearest columns to the protected column only), so candidate-set sizes differ (mean candidates scored: 106 / 145 / 125 for santos3 / tusSmall3 / tusLarge3).
* Ground truth counts every version of a table (original and each fair copy) as unionable; precision and recall are by table name.
