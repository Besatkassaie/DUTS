# Stage 1 benefit: post-retrieval pipeline time

`top_n=30000`, `k=10`, `alpha=3` (pool `alpha*k=30`). `F*/delta` = 0.122/0.07 on Santos-Large, 0.20/0.10 on the WDC tiers.

**Timed window**: starts immediately before Stage 1, ends when the pipeline returns `R`. Retrieval, the overlap intersection and the `N_i`/`n_i` synopsis lookup are all outside it. Each query is retrieved once and the identical `D` is handed to both conditions, so they differ in nothing but whether Stage 1 runs. All times are means over the cohort, in ms.

- `two_stage`: Stage 1 (Dinkelbach) -> score `U` on the 30-table pool -> Stage 2 over the pool
- `skip_stage1`: score `U` on all of `D` -> Stage 2 over `D`, cardinality `k`

Saving = `(T_skip - T_two)/T_skip`; the `x` column is `T_skip/T_two`.

| Dataset | n | \|D\| | S1 | score | S2 | **two-stage total** | score | S2 | **skip total** | saved % | x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Santos-Large | 46 | 259 | 0.29 | 39.68 | 3.18 | **43.2** | 386.5 | 4.0 | **390.5** | **88.9** | 9.0 |
| WDC-10K | 67 | 881 | 1.33 | 9.21 | 2.10 | **12.6** | 227.9 | 10.5 | **238.4** | **94.7** | 18.9 |
| WDC-100K | 67 | 2518 | 3.49 | 7.06 | 2.84 | **13.4** | 433.0 | 36.9 | **470.0** | **97.1** | 35.1 |
| WDC-1M | 66 | 5211 | 8.82 | 6.46 | 4.85 | **20.1** | 667.3 | 525.8 | **1193.1** | **98.3** | 59.3 |

## Findings

- **The saving grows with the corpus**: 88.9% on Santos-Large, 94.7% on WDC-10K, 97.1% on WDC-100K, 98.3% on WDC-1M. Stage 1 caps the number of unionability computations at `alpha*k=30` regardless of how many candidates retrieval returns, so the larger `|D|` is, the more work it removes.
- **Almost all of it is unionability scoring.** Scoring falls from 386.5 to 39.68 ms on Santos-Large; 227.9 to 9.21 ms on WDC-10K; 433.0 to 7.06 ms on WDC-100K; 667.3 to 6.46 ms on WDC-1M, because `two_stage` runs 30 bipartite matchings and `skip_stage1` runs `|D|` of them. This is the paper's `alpha*k`-not-`|T|` efficiency claim measured directly.
- **At WDC-1M, Stage 2 becomes expensive too.** Its ILP runs over ~5211 candidates instead of 30, costing 525.8 ms against 4.85 ms -- so skipping Stage 1 inflates both the scoring and the solving, not just the scoring.
- **Stage 1 pays for itself many times over**: it costs 0.29 ms on Santos-Large, 1.33 ms on WDC-10K, 3.49 ms on WDC-100K, 8.82 ms on WDC-1M, against savings of hundreds of ms.
- **Feasibility is unaffected**: 46/46 on Santos-Large, 67/67 on WDC-10K, 67/67 on WDC-100K, 64/66 on WDC-1M -- identical in both conditions on every dataset. The distribution-aware pre-filter never cost a feasible answer here.

## The quality trade-off

`skip_stage1` optimizes over a superset of `two_stage`'s candidates under the identical constraint, so its `sum_U` can only be greater than or equal. Mean `sum_U`:

| Dataset | two-stage | skip Stage 1 | retained |
|---|---:|---:|---:|
| Santos-Large | 121.5 | 135.9 | 89% |
| WDC-10K | 44.8 | 55.7 | 80% |
| WDC-100K | 49.0 | 57.6 | 85% |
| WDC-1M | 51.6 | 56.8 | 91% |

So the speedup is not free: the pool retains 80-90% of the unionability that scoring every candidate would achieve, while cutting post-retrieval time by 89-98%.

## Caveats

1. `top_n=30000` is the deepest retrieval setting, hence the largest `|D|`; the saving narrows at smaller `top_n`, where `|D|` is smaller.
2. `|D|` is right-skewed on WDC-1M (mean 5211, median ~3{,}557, max 16{,}715), so the median per-query saving (97.7%) is reported alongside the mean (98.3%).
3. One WDC-1M query has `|D|=2 < k`, returning an empty result in both conditions with no work done; it is excluded (n=66).
4. One run per cell. Absolute ms vary with machine load between runs: the same cached `|D|` gave
   181.8 and 238.9 ms for WDC-10K's skip condition on two runs, so the **ratio**, not the raw
   millisecond value, is the stable quantity. The monotone trend across corpus sizes is the finding.
5. **HNSW indexes are now cached to disk** (`experiments/semantic_cache.py`), so `|D|` is reproducible
   across runs. Before caching it was not: `hnswlib.add_items` is multi-threaded, and WDC-100K's mean
   `|D|` came out as 2,369 / 2,695 / 2,699 / 2,518 across four builds (~14% spread). That tier is
   sensitive because six duplicate cohort queries sharing `(attr=4, value=2)` flip between `|D|=234`
   and `|D|=3102` depending on the graph. santosLarge and WDC-10K varied by at most +/-1 table per
   query; WDC-1M was already deterministic via its own M=16 cache. Verified after the fix: WDC-100K
   build-then-cache vs load-from-cache agree on 67/67 queries exactly.

Script: `experiments/stage1_benefit.py` (sbatch). Per-query CSVs: `results/stage1_benefit_<dataset>.csv`.
