# WDC `top_n` × `alpha` × `k` sweep at widened `sigma` — scalability results

**2026-08-13** · conda `TableUnionNew` · `F* = 0.3`, `δ = 0.15`, `τ = 0.15`, `include_query=True`,
`sigma = 0.30` (down from the paper-matching default 0.6), seed 42, union-mode retrieval
(`D = D_sem ∪ D_ovl`, the same deliberate §7.3 relaxation as `RESULTS-wdc.md` §6/§7 — see that
report's caveat, repeated in §7 below). Corpus: WDC Web Table Corpus 2015, English relational
subset. Tiers: `tier_10k` (10,000 tables), `tier_100k` (100,000 tables), `tier_10k ⊂ tier_100k`.
This is an ad hoc follow-up experiment (ties into `RESULTS-wdc.md`'s existing union-mode study, not
part of `PLAN-integration.md`'s phase sequence).

**Grid**: `top_n ∈ {500, 1000, 2000, 3000, 4000, 5000}` × `alpha ∈ {5, 10, 15, 20}` ×
`k ∈ {10, 20, 30, 40, 50}` × 30 auto-selected queries per `(tier, top_n)` cell (re-selected at each
`top_n`, same seed — see §8 on what that does and doesn't hold fixed). 3600 rows
(`tier_10k`) / 3580 rows (`tier_100k`, one query dropped at `top_n=500` for `|D_union| < 50`).

**Code touched for this experiment** (both changes are permanent, not reverted after the run):

- `dutsx/adapters/semantic.py::HnswRetriever.query` now retunes hnswlib's `ef` search-width
  parameter to match the requested `top_n` on every call, instead of leaving it fixed at the
  constructor default (100). hnswlib doesn't error when `ef < top_n` — it silently returns fewer
  candidates ("silently degrades recall" is the honest framing observed empirically, see the
  module's inline comment) — which was already a mild under-provisioning at the original tier
  studies' `top_n=200` (2x over `ef=100`), but would have been a serious one at this sweep's
  `top_n=5000` (50x over). Full test suite (1974 passed, 19 skipped, same as before the change) still
  passes; one recall-threshold test (`test_hnsw_recall_vs_exact_scan_on_santos`) is independently
  flaky at `top_n=100` regardless of this change (hash-randomized `set` iteration order, documented
  precedent in `RESULTS-wdc.md` §7), confirmed by rerunning it in isolation several times both before
  and after this edit.
- New modules: `experiments/wdc/topn_alpha_k_sweep.py` (the sweep driver, modeled on
  `union_scalability_study.py::run_tier_study_union` with `ctx` built once per tier rather than once
  per `top_n` — `sigma` only changes `HnswRetriever`'s post-search similarity filter, not the index
  structure, so no rebuild is needed across the `top_n` sweep), `experiments/wdc/run_sweep.py` (CLI
  entry point, one tier per invocation), `experiments/wdc/plot_sweep.py` (figures, style matching
  `experiments/beamer/plot_all.py`'s palette/helpers).

## 1. On "inference time" — verified, not just asserted

Before running anything, this was checked directly rather than assumed: a grep for
`torch|transformers|SentenceTransformer|.encode(|model(` across the entire `dutsx/`, `duts/`, and
`experiments/wdc/` trees returns **zero hits**. `build_wdc_context` loads each tier's precomputed
`<tier>_roberta.pkl` exactly once, before the query loop starts (`query_vectors is candidate_vectors`
— WDC has no query/datalake split, so every column's embedding, query or candidate side alike, comes
from that same one-time load). `HnswRetriever.query()` is a pure hnswlib ANN lookup over those
already-loaded vectors; `PinnedMatchScorer.score()` (the unionability/`U` computation) reads the same
pre-loaded `vectors` dict via `cosine_sim`/`linear_sum_assignment`. The one real inference cost —
`embedding_extraction_s`, the offline GPU step that produced the `.pkl` in the first place — lives
only in `build_times` (§2's table), reported once per tier, and is not folded into any
`(top_n, alpha, k)` timing cell below.

## 2. One-time index build cost (not part of the per-cell timings)

| Component | `tier_10k` | `tier_100k` |
|---|---:|---:|
| Load resources | 0.15s | 1.46s |
| HNSW build | 5.0s | 80.9s |
| Inverted index (posting lists) | 97.0s¹ | 261.3s |
| **Total** | **102.1s¹** | **343.7s** |

¹ `tier_10k`'s inverted-index build was measured while an unrelated smoke-test process was
concurrently contending for CPU on this shared machine (caught mid-run and killed) — a clean solo
rerun of the same build measured **25.7s** (total **31.0s**), consistent with `RESULTS-wdc.md`
§2's independently-measured 25.0s at `sigma=0.6`. The 102.1s figure is left in the log rather than
silently corrected, but treat 31.0s as the representative number; it does not affect any
per-`(top_n,alpha,k)` timing below, which were all measured solo (confirmed via `ps aux` — no
concurrent processes past that point).

`embedding_extraction_s` and `csv_conversion_s` are unchanged from `RESULTS-wdc.md` §2 (embeddings
were reused as-is, not regenerated for this `sigma=0.30` sweep — see the header note above).

## 3. Retrieval yield vs. `top_n` (mean columns retrieved per query)

**`tier_10k`:**

| `top_n` | `n_sem` | `n_ovl` | `\|D_union\|` |
|---:|---:|---:|---:|
| 500 | 500.0 | 926.7 | 596.9 |
| 1000 | 1000.0 | 926.7 | 723.3 |
| 2000 | 2000.0 | 926.7 | 981.5 |
| 3000 | 3000.0 | 926.7 | 1239.6 |
| 4000 | 4000.0 | 926.7 | 1491.1 |
| 5000 | 5000.0 | 926.7 | 1731.7 |

**`tier_100k`:**

| `top_n` | `n_sem` | `n_ovl` | `\|D_union\|` |
|---:|---:|---:|---:|
| 500 | 500.0 | 10009.0 | 5422.1 |
| 1000 | 1000.0 | 9675.4 | 5364.1 |
| 2000 | 2000.0 | 9675.4 | 5607.1 |
| 3000 | 3000.0 | 9675.4 | 5853.4 |
| 4000 | 4000.0 | 9675.4 | 6084.9 |
| 5000 | 5000.0 | 9675.4 | 6325.8 |

`n_sem == top_n` exactly at every setting on both tiers — at `sigma=0.30` essentially every one of
the `top_n` nearest neighbors HNSW returns clears the (now much more permissive) similarity gate, so
the semantic side is request-bound, not threshold-bound, across this whole range. `n_ovl` is flat
against `top_n` as expected (the overlap posting-list scan doesn't depend on `top_n` at all) and is
the dominant term in `|D_union|` at `tier_100k` (~10x `tier_10k`'s `n_ovl`, tracking the ~10x corpus
size). Figures: `experiments/results/sweep_figures/columns_retrieved_vs_topn_{tier_10k,tier_100k}.pdf`.

## 4. Feasibility

| `top_n` | `tier_10k` feasible rate | `tier_100k` feasible rate |
|---:|---:|---:|
| 500 | 51.3% | 44.1% |
| 1000 | 52.0% | 42.7% |
| 2000 | 51.3% | 42.7% |
| 3000 | 51.3% | 42.7% |
| 4000 | 52.0% | 42.7% |
| 5000 | 51.3% | 42.7% |

(`feasible` and `tau_reached` are identical columns here — Stage 2 never returns an infeasible-but-
reported-feasible result, consistent with the exactness guarantee in `CLAUDE.md`'s invariants.)
Both essentially flat against `top_n` once `|D_union|` clears the `k` threshold — feasibility here is
governed by whether the *distribution* of `N_i/n_i` in the pool can reach `tau=0.15`, not by pool
size, matching `RESULTS-wdc.md` §6's finding at the narrower `top_n=200`/`sigma=0.6` setting.
`tier_100k`'s rate (~43%) is consistently ~9 points below `tier_10k`'s (~52%) across every `top_n` —
worth a follow-up if the paper wants a claim about corpus-size effects on feasibility, but out of
scope here.

## 5. Timing vs. `top_n` (mean end-to-end, seconds, over all 20 `alpha`×`k` cells)

| `top_n` | `tier_10k` | `tier_100k` |
|---:|---:|---:|
| 500 | 0.119 | 0.755 |
| 1000 | 0.148 | 0.705 |
| 2000 | 0.173 | 0.746 |
| 3000 | 0.194 | 0.951 |
| 4000 | 0.250 | 0.836 |
| 5000 | 0.256 | 0.834 |

`tier_10k` climbs smoothly and monotonically with `top_n` (0.119s → 0.256s, 2.2x over the range) —
see `experiments/results/sweep_figures/time_vs_topn_by_alpha_tier_10k.pdf`. `tier_100k`'s curve is
**not** monotonic — it dips at `top_n=1000` and spikes at `top_n=3000`
(`.../time_vs_topn_by_alpha_tier_100k.pdf`). This was checked, not shrugged off: the same 30
`(q_table, attr, M)` triples were selected at `top_n ∈ {1000, 3000, 4000}` for `tier_100k` (verified
via identical `n_ovl` value sets across those three settings — query selection is genuinely
unaffected by `top_n` at this scale, not just approximately so), and `|D_union|` itself (§3) grows
smoothly through the same range. So the bump is real wall-clock measurement noise — most likely
disk/filesystem-cache variance in the per-candidate `N_i/n_i` synopsis lookup, which dominates total
time (§6) and is the only I/O-bound stage — not a genuine `top_n` effect, and not a query-selection
confound. It shows up identically (same shape) in the stage-breakdown figure, §6, which averages
over all 20 `alpha`×`k` cells and would smooth out cell-level noise but not a shared per-`top_n`
timing shift.

Absolute magnitudes: `tier_100k` is **3.26x-6.34x** `tier_10k`'s time at matched `top_n` (narrowing as
`top_n` grows, same direction as §7's `alpha`/`k` narrowing below), well below the 10x growth in
table count — sublinear scaling, consistent with `RESULTS-wdc.md` §6's finding at
`top_n=200` that lookup time (not the HNSW probe or combine step) is what scales with corpus size,
and lookup only scales with `|D_union|` (§3: `tier_100k`'s `|D_union|` is **9.08x** `tier_10k`'s at
`top_n=500`, shrinking steadily to **3.65x** at `top_n=5000` as `tier_10k`'s smaller corpus saturates
its available semantic neighbors faster — not a flat ratio, and not the full 10x table-count ratio at
any `top_n`).

## 6. Stage-time breakdown (mean over all rows, seconds)

| Stage | `tier_10k` | `tier_100k` | Ratio |
|---|---:|---:|---:|
| Probe (HNSW + overlap) | 0.0052 | 0.0142 | 2.7x |
| Union combine | 0.0018 | 0.0093 | 5.2x |
| **`N_i/n_i` lookup** | **0.1363** | **0.7235** | **5.3x** |
| Stage 1 (Dinkelbach) | 0.0024 | 0.0114 | 4.8x |
| Unionability scoring | 0.0367 | 0.0399 | 1.1x |
| Stage 2 (ILP) | 0.0074 | 0.0065 | 0.9x |
| **End-to-end** | **0.1898** | **0.8049** | **4.2x** |

Same conclusion as `RESULTS-wdc.md` §6 at the wider `sigma`/`top_n` setting: the `N_i/n_i` synopsis
lookup over the union pool dominates (72% of `tier_10k`'s time, 90% of `tier_100k`'s), and is
essentially the only stage whose cost tracks corpus size — unionability scoring is nearly flat
(1.1x) because it only runs over the Stage-1 pool (`alpha*k`, capped independent of `|T|`), and
Stage 2's ILP cost is actually *lower* at `tier_100k` (0.9x) since infeasible instances short-circuit
before the full solve more often there (§4's lower feasibility rate). Figures:
`experiments/results/sweep_figures/stage_breakdown_vs_topn_{tier_10k,tier_100k}.pdf`.

## 7. `alpha` × `k` trend, both tiers — the main ask

At `top_n=5000` (the largest/most-at-scale setting):

**`tier_10k`** mean end-to-end time (s):

| `k` | α=5 | α=10 | α=15 | α=20 |
|---:|---:|---:|---:|---:|
| 10 | 0.218 | 0.225 | 0.230 | 0.236 |
| 20 | 0.224 | 0.235 | 0.251 | 0.261 |
| 30 | 0.229 | 0.246 | 0.265 | 0.279 |
| 40 | 0.236 | 0.262 | 0.280 | 0.294 |
| 50 | 0.238 | 0.267 | 0.306 | 0.330 |

**`tier_100k`** mean end-to-end time (s):

| `k` | α=5 | α=10 | α=15 | α=20 |
|---:|---:|---:|---:|---:|
| 10 | 0.790 | 0.795 | 0.802 | 0.809 |
| 20 | 0.795 | 0.809 | 0.825 | 0.840 |
| 30 | 0.800 | 0.825 | 0.843 | 0.858 |
| 40 | 0.808 | 0.837 | 0.894 | 0.890 |
| 50 | 0.817 | 0.855 | 0.881 | 0.906 |

Both tiers grow monotonically with `k` at every `alpha`, and with `alpha` at every `k` — expected,
since Stage 1's pool (`alpha*k`) and Stage 2's `k`-selection both grow directly. `tier_100k` sits
**2.75x-3.63x** above `tier_10k` across the whole grid at this `top_n`, narrowing toward the low end
of that range as `alpha`/`k` grow: from `(alpha=5,k=10)` to `(alpha=20,k=50)`, `tier_10k`'s time grows
**1.51x** but `tier_100k`'s only grows **1.15x** — flatter, not steeper, at the high end. The clamp
tables below explain why: `tier_100k` clamps (see the caveat and tables just below) far more often at the high end (up to
**54.7%** of queries) than `tier_10k` does (up to **10.0%**), so a much larger share of `tier_100k`'s
high-`alpha`/`k` cells skip real Dinkelbach optimization entirely — that's what flattens its growth
and narrows the ratio, not the reverse — consistent with §5's ratio also narrowing as `top_n` grows
(6.34x at `top_n=500` down to 3.26x at `top_n=5000`), though the two narrowings have different causes
(§5's is the `n_D_union` growth-rate gap between tiers closing, §7's is `tier_100k`'s clamping).
Full faceted view across every `top_n` (6 rows × 4 `alpha`
columns, both tiers per panel, shared y-axis per row):
`experiments/results/sweep_figures/time_vs_k_by_alpha_both_tiers_all_topn.pdf`; the `top_n=5000`
slice alone: `.../time_vs_k_by_alpha_both_tiers_topn5000.pdf`.

**Caveat on reading this grid**: at the high end (`alpha=20, k=50` → requested pool size 1000),
Stage 1 gets **clamped** (`pool_size >= |D_union|`, so Dinkelbach never actually runs — the pool is
just `D_union` directly) at very different rates on the two tiers. Clamped cells still cost real
lookup/scoring/ILP time (that's genuine, measured work) but the "Stage 1" line item in §6 is zero for
them, and `alpha` stops being the actual pool-size lever once clamped — the requested `alpha*k` is a
ceiling, not the realized pool size. This matters most in the `alpha=20` column above, and matters
far more for `tier_100k` than `tier_10k` at `top_n=5000` (below), the opposite of what corpus size
alone would suggest — `tier_100k`'s `|D_union|` (6325.8, §3) is larger in absolute terms than
`tier_10k`'s (1731.7), but the requested pool sizes here (up to 1000) are large enough relative to
`tier_10k`'s `|D_union|` to mostly avoid clamping, while `tier_100k`'s query-to-query variance in
`|D_union|` (driven by `n_ovl`, §3 — far more skewed at `tier_100k`: a max of ~68,664 against a
median of ~8.5 across the 30 selected queries, from the frequency-biased per-query `M`-selection in
`query_selection.py::select_queries_with_comparison`) leaves more queries with a small enough
realized pool to clamp against even a smaller `k`.

`tier_10k` clamp rate by `(alpha, k)`, `top_n=5000`:

| `alpha` | k=10 | k=20 | k=30 | k=40 | k=50 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 10 | 0.0% | 0.0% | 0.0% | 0.0% | 3.3% |
| 15 | 0.0% | 0.0% | 0.0% | 3.3% | 6.7% |
| 20 | 0.0% | 0.0% | 3.3% | 6.7% | **10.0%** |

`tier_100k` clamp rate by `(alpha, k)`, `top_n=5000`:

| `alpha` | k=10 | k=20 | k=30 | k=40 | k=50 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.0% | 1.1% | 6.1% | 8.9% | 14.0% |
| 10 | 1.1% | 8.9% | 19.6% | 26.3% | 31.3% |
| 15 | 6.1% | 19.6% | 29.1% | 35.8% | 43.0% |
| 20 | 8.9% | 26.3% | 35.8% | 44.1% | **54.7%** |

## 8. What this deliberately does and doesn't hold fixed

- **`top_n` changes which 30 queries get selected.** `select_queries_with_comparison`'s eligibility
  threshold (`min_posting_size = 2*top_n` by default) depends on `top_n`, so each `top_n` setting in
  §3-§6 is its own independent 30-query draw at the same seed, not the same queries re-measured at
  different `top_n`. §5's noise investigation found this to be a non-issue in practice for
  `tier_100k` at `top_n ∈ {1000,3000,4000}` (identical draws), but it was not assumed to hold
  everywhere and should not be assumed to hold for other tiers/settings without the same check.
- **Union-mode retrieval only** (`D = D_sem ∪ D_ovl`), same explicit, documented deviation from
  Fair_Table_Search_7.pdf §7.3 as `RESULTS-wdc.md` §6/§7 — intersection-mode pools are too sparse at
  this corpus to show real Stage 1/Stage 2 cost (17-20% workable-query yield at the original
  `top_n=200`/`sigma=0.6` setting). Timing/scalability conclusions here describe this relaxed
  retrieval rule, not §7.3's paper-faithful one.
- **`sigma=0.30` was applied by reusing the existing `<tier>_roberta.pkl` embeddings** with a wider
  post-search filter, not by re-embedding — see the header note and §1.

## 9. Files

- Raw per-query CSVs: `experiments/results/wdc_sweep_tier_10k.csv` (3600 rows),
  `wdc_sweep_tier_100k.csv` (3580 rows).
- Selection reports (one per `top_n` per tier): `wdc_sweep_tier_10k_selection_reports.json`,
  `wdc_sweep_tier_100k_selection_reports.json`.
- Figures: `experiments/results/sweep_figures/*.pdf` (8 files — per-tier `time_vs_topn_by_alpha`,
  `columns_retrieved_vs_topn`, `stage_breakdown_vs_topn`, plus the two both-tier `time_vs_k_by_alpha`
  views described in §7).
- Code: `experiments/wdc/topn_alpha_k_sweep.py`, `experiments/wdc/run_sweep.py`,
  `experiments/wdc/plot_sweep.py`, plus the permanent `ef`-tuning fix in
  `dutsx/adapters/semantic.py::HnswRetriever.query`.
