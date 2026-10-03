# DUTS on WDC Web Table Corpus — scalability results (`tier_10k`, `tier_100k`)

**2026-08-12** · conda `TableUnionNew`/`tableunion` · `F* = 0.3`, `δ = 0.15`, `τ = 0.15`,
`include_query=True`, `top_n = 200`, seed 42. Corpus: WDC Web Table Corpus 2015, English relational
subset, 50,820,165 tables total. This report covers the two smallest sampled tiers
(`tier_10k` = 10,000 tables, `tier_100k` = 100,000 tables, nested — `tier_10k ⊂ tier_100k`).
`tier_1m` not yet run.

Rows: `experiments/results/wdc_tier_10k.csv` (20 rows / 5 queries), `wdc_tier_100k.csv`
(24 rows / 6 queries). Selection reports: `wdc_tier_10k_selection_report.json`,
`wdc_tier_100k_selection_report.json`.

§6 adds a second experiment on the same two tiers: `experiments/results/wdc_union_tier_10k.csv`
(120 rows / 30 queries), `wdc_union_tier_100k.csv` (116 rows / 29 queries), with their own selection
reports (`wdc_union_tier_<N>_selection_report.json`).

## 1. Not reportable

- **Precision/recall** — WDC ships no groundtruth, same as santosLarge.
- **True embedding quality on WDC** — unionability embeddings reuse the existing santos-trained
  checkpoint (`extractVectors.py`, unmodified) applied out-of-domain; real trained vectors, but with
  no WDC-specific way to validate quality. A deliberate choice (see plan), not an oversight.
- **A fitted cost-vs-`|T|` curve to 50.8M** — two data points isn't a curve; see §2 for the raw
  numbers, reported without extrapolation.

## 2. Index build cost

| Component | `tier_10k` (10K tables) | `tier_100k` (100K tables) | Ratio |
|---|---:|---:|---:|
| CSV conversion (JSON→CSV) | 5.73s | 62.49s | 10.9x |
| Embedding extraction (GPU, offline) | 70.2s | 622.9s | 8.9x |
| Load resources | 0.15s | 1.43s | 9.5x |
| HNSW build | 5.71s | 83.97s | **14.7x** |
| Inverted index (posting lists) | 25.01s | 314.15s | 12.6x |
| **Total (excl. embedding, offline step)** | **30.86s** | **399.55s** | **12.9x** |

Roughly linear-to-slightly-superlinear at 10x scale for every component except embedding extraction,
which scaled sublinearly (8.9x for 10x the data — GPU batching amortizes better at volume). HNSW's
14.7x is the one outlier, consistent with its `O(n log n)` build cost. All conversions were clean:
**0 failed, 0 recovered** at both tiers (100,000/100,000 and 10,000/10,000 tables).

## 3. Categorical column coverage (`theta_cat=50`)

| Tier | Median rows/table | Columns categorical / total | Ratio |
|---|---:|---:|---:|
| santos (reference) | ~5,000 | — | **82%** |
| `tier_10k` | ~6 (corpus-wide) | 48,857 / 51,854 | **94.2%** |
| `tier_100k` | ~6 (corpus-wide) | 487,728 / 518,131 | **94.1%** |

Confirms the finding from `tier_10k` holds unchanged at 10x scale: WDC's ratio is consistent
(94.1–94.2%) regardless of tier size, because it's driven by `nunique(col) ≤ n_rows(table)` being
almost always `≤ theta_cat=50` at WDC's table sizes — a structural property of the corpus, not
something that changes as more tables are added. The ratio is **not** directly comparable to
santos's 82% as evidence of "more/less selective" — see `plan`'s correction: santos's ratio reflects
genuine low-cardinality data at large table sizes, WDC's reflects the filter's inability to
discriminate at small table sizes. Same mechanism, same magnitude, at both WDC tiers.

## 4. Query selection and retrieval yield

| | `tier_10k` | `tier_100k` |
|---|---:|---:|
| Queries auto-selected | 30 | 30 |
| Eligibility rate (has categorical attrs) | 99.5% | 99.5% |
| Workable queries (`|D| ≥ 20`) | **5 / 30** (16.7%) | **6 / 30** (20.0%) |
| Biased median `|D|` | 2.0 | 1.0 |
| Unbiased median `|D|` | 1.0 | 1.0 |
| **Paired mean diff (biased − unbiased)** | **+0.77** | **+1.93** |

The paired comparison (same query table/attribute, only `M` differs — see the plan's methodology
correction) shows frequency-biased `M` selection helping *more*, not less, at `tier_100k` — the
paired mean advantage nearly tripled (+0.77 → +1.93). This is consistent with the mechanism's own
logic: larger corpora have larger absolute posting lists for any given popular value, giving the
biasing more room to find a genuinely well-connected `M`. Workable-query yield also improved slightly
(16.7% → 20.0%), though both remain small — most auto-selected queries still can't clear even the
smallest config's pool size, because `D` requires both semantic similarity and literal value overlap
on an uncurated corpus, and most literal values simply aren't popular enough to survive that
intersection at either scale tested so far.

## 5. Per-stage timing and outcome

| | `tier_10k` (n=20 rows / 5 queries) | `tier_100k` (n=24 rows / 6 queries) |
|---|---:|---:|
| Feasible | 20/20 (100%) | 16/24 (66.7%) |
| Mean retrieval time | — | 31.8 ms |
| Mean Stage 1 time | — | 0.07 ms |
| Mean scoring time | — | 4.8 ms |
| Mean Stage 2 (ILP) time | — | 2.6 ms |
| Mean end-to-end | — | 39.3 ms |
| `|D|` range | 24–115 | 23–196 |

(`tier_10k`'s per-stage breakdown is in `experiments/beamer/duts_presentation.pdf`'s WDC section,
not duplicated here.) Stage 1 and Stage 2 stay in the low-single-digit-millisecond range at both
tiers — consistent with santosLarge's finding that solver cost is pinned to `α×k`, not `|T|` or
`|D|`. The **feasibility rate dropping from 100% to 66.7%** between tiers is a new, real finding —
not previously observable at `tier_10k`'s small, 100%-feasible workable cohort — worth tracking as
more tiers are added: is a lower feasibility rate a property of *which* queries happen to clear the
`|D|≥20` bar at each tier (a sampling artifact of a 5–6-query cohort), or a genuine trend as `|D|`
grows and pools contain more marginal candidates? Not resolvable from two tiers alone.

## 6. Union-pool experiment: what does Stage 1 / scoring / Stage 2 actually cost?

§4/§5 above measure the paper-faithful pipeline (`D = D_sem ∩ D_ovl`, Fair_Table_Search_7.pdf §7.3),
where most auto-selected queries never clear `|D| ≥ 20` — only 5/30 (`tier_10k`) and 6/30
(`tier_100k`) do, so the Stage 1/Stage 2 timing numbers above are drawn from small, retrieval-lucky
cohorts. This section answers a narrower, purely scalability-motivated question instead: **once the
pool is actually large, what does Stage 1 (Dinkelbach), unionability scoring, and Stage 2 (ILP)
cost?** — by relaxing retrieval from intersection to **union**, `D = D_sem ∪ D_ovl`
(`experiments/wdc/union_retrieval.py`). This is a deliberate, explicitly-labeled deviation from
§7.3 kept entirely inside `experiments/wdc/`, never touching `dutsx/retrieval.py` or
`dutsx/runner.py` — nothing outside this experiment uses it. Same 30 auto-selected queries, same
seed/`M` per tier as §4/§5 (paired per query, only the combine rule differs), `top_n=200`.

| | `tier_10k` | `tier_100k` |
|---|---:|---:|
| Queries auto-selected | 30 | 30 |
| Workable (`|D| ≥ 20`) under **union** | **30 / 30** (100%) | **29 / 30** (96.7%) |
| Workable under intersection (§4, for reference) | 5 / 30 (16.7%) | 6 / 30 (20.0%) |
| Median `|D|`, intersection | 2 | 1 |
| Median `|D|`, union | **79** | **94** |
| Max `|D|`, union | 3,107 | 30,665 |
| Rows produced (queries × 4 configs, minus skips) | 120 | 116 |
| Feasible (`τ` reached) | 74 / 120 (61.7%) | 60 / 116 (51.7%) |
| Dinkelbach actually ran (not C5-clamped) | 112 / 120 | 105 / 116 |
| Mean Dinkelbach iterations (of those that ran) | 2.02 | 1.84 |
| Mean retrieval time (probe + combine + lookup, union path) | 60.9 ms | 613.7 ms |
| Mean Stage 1 time | 0.66–0.76 ms | 10.9–11.8 ms |
| Mean scoring time | 1.8–4.8 ms | 1.7–5.1 ms |
| Mean Stage 2 (ILP) time | 4.2–7.4 ms | 3.1–4.2 ms |
| Mean end-to-end | 68.2–73.9 ms | 630.5–633.2 ms |

**Union retrieval turns "almost every query is retrieval-starved" into "almost every query has a
real pool"** — workable-query yield goes from 16.7–20.0% (intersection) to 96.7–100% (union), and
median pool size grows ~40–90×. This is the direct, honest fix for the problem the paper's §7.3
combine rule creates on an uncurated corpus: requiring both signals at once is far too strict when
literal value overlap and semantic similarity rarely coincide in 50M heterogeneous web tables.

With real pools, Stage 1 stops being trivially clamped away: it actually runs Dinkelbach for
93–91% of rows (vs. the intersection study, where `clamped=True` because `α×k ≥ |D|` was common for
`|D|` in the low tens), taking a genuinely small but non-zero 2 iterations on average. **Scoring and
Stage 2 stay in the low-single-digit-millisecond range at both tiers** — reconfirming the paper's
central claim that Stage 2's cost is pinned to `α×k`, never `|D|`, even when `|D|` itself grows by
orders of magnitude (median 79 → 94, max 3,107 → 30,665).

### 6.1 What retrieval time is actually spent on

An earlier version of this experiment computed both the union and intersection pools by calling two
separate retrieval functions back-to-back inside the same timed block — a real measurement bug: the
semantic (HNSW) probe and the overlap (posting-list) probe each ran **twice** (once per function
call), and their combined wall time was reported as "the union path's retrieval cost." Fixed by
probing once and deriving both reduced candidate sets from the same probe
(`experiments/wdc/union_retrieval.py::retrieve_unscored_candidates_union_and_intersection`), with
an explicit per-phase timing breakdown so "how much does the combine operator itself cost" is a
directly reported number, not folded into an undifferentiated total:

| Phase | `tier_10k` (mean, ms) | `tier_100k` (mean, ms) |
|---|---:|---:|
| Probe (semantic query + overlap query — shared, paid once regardless of combine rule) | 0.74 | 4.91 |
| Union combine (set union + per-table `argmax sim` reduction) | 0.54 | 6.47 |
| Intersection combine (set intersection + per-table reduction) | 0.04 | 0.32 |
| **Union `N_i`/`n_i` lookup (the dominant cost)** | **59.65** | **602.32** |
| Intersection `N_i`/`n_i` lookup | 1.59 | 1.78 |
| **Total, union path (`retrieval_time_s`)** | **60.93** | **613.70** |
| Total, intersection path (`intersection_time_s`, comparison only) | 2.37 | 7.01 |

The fix's actual numerical impact was modest — the old (buggy, double-probing) numbers were 63.9 ms
(`tier_10k`) and 627.5 ms (`tier_100k`), so the double-probe only inflated the reported cost by
~3 ms and ~14 ms respectively (≈5% and ≈2%). Feasibility counts are **unchanged** by the fix
(74/120, 60/116 — identical to the buggy run), confirming this was purely a timing-measurement
correction, not an answer change. **What the corrected breakdown actually shows: neither the
combine operator (union vs. intersection) nor the underlying probe (HNSW + posting-list scan) is
where retrieval time goes.** Both are cheap and roughly comparable in magnitude to each other
(low-single-digit ms even at `tier_100k`). The dominant, scaling cost is the **`N_i`/`n_i` synopsis
lookup over the union pool** — one `distribution()` call per retrieved candidate table, growing
directly with `|D_union|` (59.65 ms → 602.32 ms, ≈10×, tracking the ≈10× growth in tier size and
posting-list sizes that drive `|D_union|` up). This is a more precise statement than "retrieval time
scales with `|D|`": it isolates *which part* of retrieval does — the per-candidate synopsis lookup,
not the ANN probe or the set-combine step, both of which stay cheap regardless of how large the
retrieved pool ends up being.

**Feasibility, once retrieval is no longer the bottleneck, is still only ~52–62%** — genuinely
`τ`-unreachable instances (`stage2_infeasible`), not `insufficient_candidates`. This is a different,
more informative failure mode than §4/§5's: it says that even with an ample pool, roughly half the
auto-selected `(q_table, M)` combinations don't admit a `k`-subset at `F_R ≥ τ = 0.15` — a property
of the *distribution* of `N_i/n_i` values in the union pool, not of pool size.

**Caveat, stated plainly**: this experiment intentionally violates §7.3's compound-filter
requirement, so its `D` is not the paper's `D` — a union-mode candidate can be schematically similar
via one attribute while its literal-value match came from a *different* row's context entirely
(the per-table `argmax sim` reduction still holds, so each table contributes at most one attribute,
but that attribute may owe its presence in `D` to only one of the two signals, never both). Timing
findings (Stage 1/scoring/Stage 2 cost vs. `α×k`) generalize; retrieval-yield findings do not — they
describe this relaxed retrieval rule, not §7.3's.

## 7. Attributes indexed vs. retrieved, and queries tried

| | `tier_10k` | `tier_100k` |
|---|---:|---:|
| Columns total | 51,854 | 518,131 |
| Columns indexed (categorical, `theta_cat=50`) | 48,857 (94.2%) | 487,728 (94.1%) |
| Queries tried (auto-selected) | 30 | 30 |
| Queries workable, intersection retrieval | 5 | 6 |
| Queries workable, union retrieval | 30 | 29 |
| Attribute-columns retrieved, intersection (summed over the 30 tried queries) | 444 | 448 |
| Attribute-columns retrieved, union (summed over the 30 tried queries) | 15,732 | 155,079 |

"Indexed" is corpus-wide and computed once (§3): a column clears `theta_cat=50` and gets a synopsis
+ posting-list entry, independent of any query. "Retrieved" is query-time and summed over the 30
auto-selected queries' retrieval calls: each retrieved candidate is exactly one `(table, attribute)`
pair (§7.3's `argmax sim` reduction keeps one per table), so this is literally "how many attribute
columns appeared in some query's `D`," not a distinct-column count — the same column of the same
table can be retrieved by more than one query and is counted once per query that retrieved it.
Under **intersection**, retrieved attributes are a tiny fraction of indexed ones (0.09–0.91%) — as
expected, since §7.3's compound filter is strict. Under **union**, the fraction is far larger:
**~32%** of all indexed columns get touched by just 30 queries' retrieval calls (15,732 / 48,857 at
`tier_10k`, 155,079 / 487,728 at `tier_100k`) — a single query's semantic probe alone (`top_n=200`)
already accounts for most of this, since `n_sem` is pinned at `top_n` per query regardless of tier
size (30 queries × 200 ≈ 6,000 semantic pairs before per-table reduction and before the (often much
larger) overlap side is even counted). This is a real, if unsurprising, consequence of relaxing the
combine rule: union retrieval is not a small perturbation on top of intersection's tiny pools, it
touches a substantial slice of the entire indexed corpus per query batch. These counts vary by a few
percent between reruns at fixed seed (e.g. `tier_100k` intersection: 511 vs. 448 across two runs) —
HNSW's ANN search is approximate, and per-table tie-breaking among equal-similarity pairs depends on
Python's (hash-randomized, per-process) `set` iteration order; the magnitudes and conclusions are
stable, the exact counts are not bit-reproducible.

## 8. Column/label glossary

- `n_D` — retrieved candidate pool size before Stage 1.
- `feasible` / `tau_reached` — Stage 2 found a result with `F_R ≥ τ`.
- `paired mean diff` — mean of (biased `|D|` − unbiased `|D|`) over the *same* query table/attribute
  selections, isolating the effect of `M`-selection strategy from which queries got picked.
- `n_D_intersection` / `n_D_union` (§6/§7 only) — pool size under §7.3's intersection rule vs. the
  union relaxation, for the *same* query table/attribute/`M`.
- `probe_time_s` / `union_combine_time_s` / `intersection_combine_time_s` / `union_lookup_time_s` /
  `intersection_lookup_time_s` (§6.1 only) — the retrieval-time breakdown: `probe_time_s` is shared
  (paid once regardless of combine rule); `*_combine_time_s` is the set-combine + per-table
  `argmax sim` reduction; `*_lookup_time_s` is the `N_i`/`n_i` synopsis lookup over that variant's
  own reduced set. `retrieval_time_s = probe + union_combine + union_lookup`;
  `intersection_time_s = probe + intersection_combine + intersection_lookup` (comparison only, not
  on the critical path).
- Full field list: `experiments/wdc/scalability_study.py::WdcRow` (§2–§5),
  `experiments/wdc/union_scalability_study.py::WdcUnionRow` (§6–§7).

## 9. Next

`tier_1m` (1,000,000 tables) is sampled and ready (`/u6/bkassaie/wdc_data/tiers/tier_1m/manifest.json`)
but not yet run — extrapolating from `tier_100k`'s costs, the inverted-index build alone would be
roughly 3,000–3,500s (~1 hour), embedding extraction roughly 6,000s (~1.7 hours GPU time). A real
three-point cost-vs-`|T|` curve becomes possible only once it completes.
