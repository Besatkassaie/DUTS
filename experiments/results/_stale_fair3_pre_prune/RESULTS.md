# DUTS on santos3 — solvable instances only, α = 2, k ∈ {3, 5}

**2026-09-11** (regenerated; supersedes the 2026-08-10 run below) · conda `TableUnionNew`, py 3.8.5,
scipy 1.10.1 (HiGHS) · seed 42.
`F* = 0.3`, `δ = 0.15`, `τ = 0.15`, `include_query=True`, `top_n = 100`.
Data: `starmie_fair/data/santos3` — 48 queries (all `_fair` rebalanced), 931 datalake tables
(550 originals + 381 `_fair` versions).

**Why regenerated.** The original 2026-08-10 run predates a 2026-08-14 retrieval bugfix
(`experiments/context.py::categorical_only_vectors`, generalized from a fix first found in the WDC
context builder): `HnswRetriever` indexes every column with a nonzero embedding, with no categorical
filter of its own, so before the fix its `top_n=100` nearest-neighbor window per query was partly
filled with non-categorical columns that could never survive the overlap-filter intersection anyway
(`InvertedIndexOverlap` only ever indexes `synopsis.categorical_attrs`). Zeroing non-categorical
embeddings before building the semantic index frees up those wasted top-100 slots for candidates that
*can* actually survive intersection — mean `|D|` on this cohort rose from ≈9 to ≈15 (roughly ×1.76)
as a direct result. Every number below is from a clean rerun of `focused_santos3.py`,
`groundtruth_eval.py`, and `scoped_reachability.py` against the current (post-fix) code; the stale
originals are kept at `experiments/results/_stale_pre_20260814_fix/` for reference.

**Cohort.** Restricted to queries where `τ` is *provably* reachable — brute-force enumeration of
every `k`-subset of `D` confirms some subset attains `F ≥ τ`. Queries that fail only because
retrieval didn't supply `k` candidates are excluded: they carry no Stage 1, no Stage 2 and no
timings, so they dilute every mean without saying anything about the optimizer.

Longer full-parameter version kept at `RESULTS-santos3-full.md` (**not yet regenerated against the
post-fix retrieval code** — treat its numbers as stale until it is).
Data: `experiments/results/santos3_focused_k3_k5_alpha2.csv` (86 rows).

---

## 1. How many instances are solvable, and does the pipeline solve them

| | k = 3 | k = 5 |
|---|---:|---:|
| Queries with `τ` reachable (retained) | **43 / 48** | **43 / 48** |
| Excluded — `\|D\| < k`, retrieval shortfall | 5 | 5 |
| Excluded — `τ` unreachable although `\|D\| ≥ k` | 0 | 0 |
| **Pipeline feasible on retained** | **43 / 43 (100%)** | **43 / 43 (100%)** |
| Mean `\|D\|` (candidate tables) | 14.7 (6–30) | 14.7 (6–30) |
| Mean `k`-subsets enumerated per query | 754 | 15,206 |

**On every instance where a feasible answer exists, the two-stage pipeline finds one — 86 of 86.**
Stage 2 never failed on a solvable instance at these parameters. Every failure on the full 48 is a
retrieval-supply problem, not an optimization one.

---

## 2. Quality

| | k = 3 | k = 5 |
|---|---:|---:|
| `F_P` (Stage 1 pool) | 0.481 | 0.451 |
| `F_R` (result) | 0.474 | 0.457 |
| **`\|F* − F_R\|` achieved** | **0.186** | **0.179** |
| `min \|F* − F(S)\|` attainable by *any* `k`-subset | 0.098 | 0.112 |
| ΣU | 31.31 | 51.93 |
| `\|P\|` = unionability computations | 6.00 | 9.23 |

`F_R` sits at 0.46–0.47 against a target of 0.30 — every result **overshoots**.

**Overshoot is not a defect here — it's within spec.** Stage 2's only constraint is the floor
`F_R ≥ τ`; there is no upper bound, and Stage 1's objective is an *unbounded* `argmax F` by design
(correction C3, resolved as the intended objective, not a placeholder — see `CLAUDE.md`). Neither
stage is asked to land near `F*`, so `F_R ≈ 0.47` is exactly as valid an answer as `F_R = 0.31` would
be. The fourth row is a diagnostic against a *hypothetical* proximity-seeking objective
(`satisfice`, deliberately unimplemented), not a scorecard for the one that's actually running: even
the single `k`-subset closest to `F*` in `D` only gets within 0.098/0.112 of it, so most of that
distance is set by which tables retrieval handed over, before either stage does anything. The
remaining 0.083/0.039 (`F_P − (F* + best_gap)`) is how much further Stage 1's `argmax F` moves past
that point — relevant only if `satisfice` is ever built, not a mark against `max_f` as it stands.
**The more informative comparison** (used on the corresponding beamer slide) scopes the attainable
range to what each stage actually selects from — `[F_min_pool, F_max_pool]` over `α·k`-subsets of `D`
for `F_P`, `[F_min_P, F_max_P]` over `k`-subsets of `P` (not `D`) for `F_R` — rather than a single
`k`-subset-of-`D` range shared by both; see `experiments/scoped_reachability.py`.

---

## 3. Time, split by stage

Mean ms per query, feasible rows.

| | retrieval (index) | Stage 1 | scoring (U) | Stage 2 (ILP) | **end-to-end** |
|---|---:|---:|---:|---:|---:|
| **k = 3** | 4.96 (37%) | **0.032 (0.2%)** | 4.51 (34%) | 3.84 (29%) | **13.34** |
| **k = 5** | 5.02 (33%) | **0.030 (0.2%)** | 6.42 (42%) | 3.83 (25%) | **15.30** |

Three things worth stating plainly:

- **Stage 1 is still free** — ~30 microseconds, 0.2% of end-to-end. Whatever the two-stage design
  costs, it is not the Dinkelbach step.
- **Unionability scoring dominates, 34–42%** — larger `|P|` under the post-fix retrieval (6.0/9.2 vs.
  5.5/8.4 before) pushes this up somewhat, but it's still the term §4.1 bounds by `α×k` rather than
  `|T|`, so it is the right thing to bound.
- **Stage 2's ILP share grew materially, 25–29% (was 16–17%)** — a direct consequence of the larger
  pools; the LP pre-check's near-40% overhead within that (§5) is now a bigger absolute cost too.
- **Retrieval is comparable to scoring at ~33–37%** and is independent of `k`.

---

## 4. Dinkelbach iterations

| | k = 3 | k = 5 |
|---|---:|---:|
| Rows where Stage 1 actually ran Dinkelbach | 39 / 43 | 29 / 43 |
| Rows where the C5 clamp made it unnecessary (`α×k ≥ \|D\|`) | 4 / 43 | 14 / 43 |
| Mean iterations when it ran | 1.23 | 1.28 |
| Max iterations observed | **2** | **2** |

Across all 68 invocations: **51 converged in 1 iteration, 17 in 2. Never more.** The paper's
"converges in only a few iterations" claim holds strongly — and now, with the larger post-fix pools,
Dinkelbach genuinely runs on most rows (68/86) rather than being clamped away on around half of them.

---

## 5. Was the LP feasibility pre-check ever decisive?

**No. Not once.**

| | k = 3 | k = 5 |
|---|---:|---:|
| LP pre-check ran | 43 / 43 | 43 / 43 |
| **LP was decisive** (proved infeasible ⇒ ILP skipped) | **0** | **0** |
| Mean LP time | 1.532 ms | 1.474 ms |
| Mean total Stage 2 time | 3.840 ms | 3.826 ms |

The pre-check ran on all 86 instances, proved nothing on any of them, and consumed **≈39–40% of
total Stage 2 time** — a bigger absolute cost than before, since Stage 2's pools grew along with `|D|`
(§1). On this cohort that is partly by construction — every retained instance is solvable, so there
is no infeasibility for the LP to detect. (The unrestricted santos3 sweep's decisiveness figure —
previously cited as 5 of 260 — predates this same retrieval fix and has **not** been regenerated;
treat it as stale until `santos3_lp_precheck.csv` is rerun.)

**Recommendation: turn it off.** Its documented value is soundness framing
(`LP-infeasible ⇒ ILP-infeasible`, and for this constraint shape the converse holds too), so
disabling it cannot produce a wrong verdict — HiGHS's own presolve already detects these cases
faster. Stage 2 would drop by ~1.5 ms, roughly 10–11% of end-to-end.

---

## 6. Skipping Stage 1: hand all of `D` straight to Stage 2

Same queries, same retrieval, same scoring code. `skip_stage1` scores every candidate in `D` and
gives Stage 2 the whole set at cardinality `k`.

| | k = 3 two-stage | k = 3 skip | k = 5 two-stage | k = 5 skip |
|---|---:|---:|---:|---:|
| Tables scored | **6.00** | 14.72 (+145%) | **9.23** | 14.72 (+59%) |
| End-to-end (ms) | **13.34** | 19.18 (+43.8%) | **15.30** | 20.15 (+31.7%) |
| `F_R` reached | 0.474 | **0.436** | 0.457 | **0.430** |
| `\|F* − F_R\|` | 0.186 | **0.157** | 0.179 | **0.159** |
| ΣU | 31.31 | **32.81** | 51.93 | **53.06** |
| Feasible | 43 | 43 | 43 | 43 |

**Skipping Stage 1 now costs a real 32–44% end-to-end** (up from 1.7–8.6% pre-fix — the larger `|D|`
means skip_stage1 scores far more tables), and ΣU is higher by construction (Stage 2 optimizes over
the superset `D ⊇ P`, so it can only do as well or better). `F_R` also lands closer to `F*` under
`skip_stage1` at both `k` now — but both numbers are feasible, fully valid answers under the τ-floor
constraint; proximity to `F*` is not part of either stage's objective, so "closer to `F*`" is a
diagnostic reading, not a quality score the pipeline is being judged against.

The mechanism is visible in §2: Stage 1 runs an unbounded `argmax F` (C3), pushing `F_P` to
0.48/0.45 before Stage 2 ever sees the pool. Since everything is already above `F* = 0.3`, that push
moves `F_R` further from `F*` (though no further from satisfying `F_R ≥ τ`, which is the actual
requirement) — the pool that maximizes `F` is not the pool that would sit closest to `F*`.

**Post-fix, Stage 1 now buys a real, measurable efficiency benefit at these parameters** (1.3–1.4×
speedup, §6 above and the corresponding beamer slide), unlike the pre-fix regeneration where `|D|`
was too close to `α×k` for Stage 1 to matter. It still costs a small amount of ΣU (2–5%, `skip_stage1`
optimizing over the strict superset) and lands `F_R` a bit further from `F*` on the (non-binding)
proximity diagnostic. Its efficiency case still scales with `|D| ≫ α×k`, which santosLarge
demonstrates far more dramatically (§ santosLarge) — santos3's post-fix `|D|` (mean ≈15) is bigger
than before but still modest next to santosLarge's up to 269.

---

## 7. Precision and recall vs the santos3 groundtruth

Restricted to the same retained queries, α = 2. `ideal_recall(k)` is starmie_fair's ceiling metric
(`checkPrecisionRecall.py`): `mean_i min(k, gt_size_i) / gt_size_i` — the best recall any method
could possibly achieve at that `k`, given each query's groundtruth size, independent of ranking
quality.

| | k = 3 (n=43) | k = 5 (n=43) |
|---|---:|---:|
| **precision @ D / P / R** | **0.993 / 1.000 / 1.000** | **0.993 / 0.997 / 1.000** |
| recall @ D | 0.575 | 0.574 |
| recall @ P | 0.251 | 0.378 |
| **recall @ R** | **0.125** | **0.209** |
| **`ideal_recall(k)`** | **0.125** | **0.209** |
| mean groundtruth size | 25.9 | 25.9 |
| mean hits D / P / R | 14.56 / 6.00 / 3.00 | 14.53 / 9.19 / 5.00 |

**Precision is no longer exactly 1.000 at `D`/`P`** (it was, pre-fix) — with the larger post-fix `|D|`
(§1), a handful of the additional candidates retrieval now surfaces are *not* groundtruth-unionable
(precision @ D = 0.993, i.e. a small number of false positives per query on average). Precision @ `R`
is still exactly 1.000 at both `k`: Stage 2's selection is small enough (`k` = 3 or 5, out of a larger
but still mostly-correct pool) that it never happens to land on one of those false positives.

**Recall at `R` still matches `ideal_recall(k)` exactly, to three decimal places, at both `k`.** This
still isn't approximate agreement — it's forced by two things holding on *every single query* in the
cohort: `precision_R = 1.000` (no false positives at `R`, even though `D`/`P` now have some) and every
query's groundtruth size exceeds `k` (min 12, so `min(k, gt_size) = k` throughout). Under those two
conditions, `recall = k / gt_size = ideal_recall` is an identity, not a coincidence. Recall's decline
through the funnel (`D → P → R`) is real, but by `R` it isn't costing anything recall-wise beyond
what `k` and the groundtruth size already fix — there is no headroom left for a different ranking to
recover.

---

## 8. Caveat that affects every number above

**All 48 santos3 query tables are also present in its datalake**, so a query can retrieve and select
itself. Measured on the current (post-fix) retained cohort: 41 of 43 queries retrieve their own table
at both `k` (self_in_D), and it lands in `R` for 26 of 43 (60%) at k=3 and 34 of 43 (79%) at k=5. A
self-match is a trivially perfect union candidate, and with `include_query=True` its distribution is
counted twice.

The "excluding self-matches" sensitivity sub-analysis (previously: mean `F_R` 0.439 → 0.414 at k=5,
etc.) has **not** been rerun against the post-fix retrieval code — those specific before/after numbers
are stale and are removed here rather than restated unverified. **This is still not applied by
default** in any case — it would change every number in this report, so it stays flagged rather than
silently fixed. santos2 is the only variant without this overlap.

---

## Column reference for `santos3_focused_k3_k5_alpha2.csv`

`n_D` candidates retrieved · `n_subsets` = `C(|D|,k)` enumerated · `F_min`/`F_max` attainable range ·
`best_gap` = `min |F(S) − F*|` over all `k`-subsets — the closest *any* selection rule could land to
`F*` given this `D`; a diagnostic bound, not a measure of defect in the current objective, which
does not target proximity to `F*` ·
`F_P`/`F_R` proportion of pool/result · `abs_gap_R` = `|F* − F_R|` · `sum_U` Stage 2 objective ·
`n_unionability_computations` = `|P|` · `dinkelbach_ran`/`dinkelbach_iterations` ·
`lp_ran`/`lp_feasible`/`lp_decisive` (decisive = LP proved infeasible, ILP skipped) ·
`retrieval_time_s`/`stage1_time_s`/`scoring_time_s`/`stage2_time_s`/`end_to_end_s` ·
`skip_*` the same quantities for the Stage-1-bypass arm.

`experiments/results/santos3_scoped_reachability_k3_k5_alpha2.csv` (86 rows, same cohort) is a
companion file, not a replacement: `F_min_pool`/`F_max_pool` are the attainable range over
`α·k`-subsets of `D` (or all of `D` when C5-clamped) — the correct ceiling for `F_P`, at `F_P`'s own
cardinality, not `k`'s. `F_min_P`/`F_max_P` are the attainable range over `k`-subsets of `P` (not
`D`) — the correct ceiling for `F_R`, since Stage 2 never sees anything outside `P`. See
`experiments/scoped_reachability.py`'s module docstring for why `fraction_reachability.py`'s
`F_min`/`F_max` (over `k`-subsets of `D`) is the wrong reference for either bar.
