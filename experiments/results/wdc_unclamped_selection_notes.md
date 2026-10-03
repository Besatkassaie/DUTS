# WDC unclamped shared cohort -- selection notes

Fixes the methodology gap vs. santosLarge: the original `wdc_tau_sweep_shared.py` grid (`k in {10..50} x alpha in {10..60}`, max pool 3000) never required `|D| >= max(alpha*k)`, so 87.1% of its (query, cell) pairs were silently clamped at `top_n=5000` (even the best query, |D|=563, had 20/30 cells clamped). This mirrors santosLarge's `|D| >= max(alpha*k)` cohort criterion instead.

## Candidate pool

210 candidates (60 reproduced from the original seed=42 draw + 150 new, same continued shuffle over `tier_10k`'s table universe), `top_n=5000` fixed throughout, `(attr, value)` chosen by the existing maxD proxy-rank + top-20 real-retrieval-verify method (`wdc_maxd_selection_shared.py`), evaluated against `tier_100k`.

**Reproducibility caveat**: 5 of the first 60 rows differ from the original `wdc_shared_maxd_selection_report.csv` in which tied-proxy-score `(attr, value)` was picked for that table (not in whether real |D| was measured correctly) -- a pre-existing tie-break non-determinism in `_proxy_rank`/the selection loop, not introduced here. Real |D| values throughout are freshly, correctly measured.

**Non-monotonicity**: 24/210 (11.4%) candidates have `D_10k > D_100k`, despite `tier_10k` being a literal subset of `tier_100k`. This is the same, already-documented `top_n` budget-crowding effect (`tier_100k` has 10x more tables competing for the same fixed `top_n=5000` semantic-probe slots) -- not a bug. Using `min(D_10k, D_100k)` as the binding constraint is unaffected by direction.

## Threshold curve (candidates retained vs. `min(D_10k, D_100k)` threshold)

| threshold | retained | fraction |
|---:|---:|---:|
| 20 | 122/210 | 58.1% |
| 30 | 116/210 | 55.2% |
| 40 | 109/210 | 51.9% |
| 50 | 107/210 | 51.0% |
| 60 | 102/210 | 48.6% |
| 80 | 96/210 | 45.7% |
| 100 | 94/210 | 44.8% |
| 150 | 79/210 | 37.6% |
| 200 | 67/210 | 31.9% |
| 300 | 50/210 | 23.8% |

## Chosen grid

`k in {5, 10, 20} x alpha in {5, 10}` (6 cells, same count as santosLarge's grid) -- pools: [25, 50, 100, 200] (max 200).

Chosen over the original 30-cell grid (max pool 3000, unachievable unclamped on `tier_10k` at this corpus/retrieval scale -- max observed |D| there is 1450 even at `top_n=30000`) and over keeping santosLarge's exact ceiling (60), since WDC's distribution (median 52, p75 280, p90 416, max 562) supports a materially larger pool once the candidate draw is widened to 210.

## Retained cohort

**67 of 210 candidates** (31.9%) retained -- `min(D_10k, D_100k) >= 200`, guaranteeing unclamped Stage 1 on BOTH tiers at every one of the 6 grid cells. Compare santosLarge: 47/78 (60.3%) at its own threshold/grid.

`tier_10k`-alone retention (`D_10k >= 200`, ignoring `tier_100k`): 70/210. `tier_100k`-alone retention (`D_100k >= 200`, ignoring `tier_10k`): 96/210. Joint (both tiers, the actual cohort criterion): 67/210.

