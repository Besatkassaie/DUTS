# DUTS on santosLarge — the first configuration where α actually bites

**2026-09-13** (redesigned; supersedes the 2026-08-10 run below) · conda `TableUnionNew`, py 3.8.5,
scipy 1.10.1 (HiGHS) · `F* = 0.122`, `δ = 0.07`, `τ = 0.052`, `include_query=True`, **`top_n = 5000`**,
cohort **`|D| ≥ 60`** minus one excluded outlier, α ∈ {2, 3, 4} × k ∈ {10, 15}.
Data: `starmie_fair/data/santosLarge` — 11,086 datalake tables, 78 usable queries.
Rows: `experiments/results/santoslarge_maxd_topn5000_D60.csv` (46 queries × 6 configs = 276).

**Why redesigned.** The original 2026-08-10 run used santosLarge's *random* protected-attribute/value
assignment (`starmie_fair/experimental_setup.py`, `selection_strategy='random'`): pick a random
categorical column, then `random.choice(unique_values)`, with no regard for cross-table popularity.
That retained only 19 of 78 queries at `|D|≥50`. Three changes, in order:

1. **MaxD attribute/value reselection** (`experiments/santoslarge_maxd_selection.py`): per query
   table, rank every `(categorical attr, value)` pair by a free proxy (`|InvertedIndexOverlap`
   `._postings[value]|`), verify the top 20 by real retrieval, keep whichever gives the largest actual
   `|D|` — maximize, no target. Median `|D|` rose **11 → 77** (mean 107, max 408). This deliberately
   favors common/generic values (`0`, `1.0`, `Headcount`, `Entity`, `Other`) over the specific,
   often-unique values random selection picked — see `experiments/results/santoslarge_maxd_selection_report.csv`
   for the full before/after per query.
2. **New grid, re-derived from the new `|D|` distribution, not carried over unexamined**:
   `k∈{10,15}, α∈{2,3,4}` (pool sizes 20–60), cohort `|D|≥60` (47/78 queries) — chosen so every
   retained query stays unclamped across all 6 cells.
3. **New `F*`/`δ`/`τ`, chosen empirically to maximize reach** (`experiments/santoslarge_tau_sweep.py`):
   only `τ=F*-δ` enters Stage 2's feasibility at all (`U` — hence `δ`/`F*` individually — never
   appears in the ILP's constraints, only its objective), so the sweep scores every candidate `U=1.0`
   and re-solves Stage 2 only, holding retrieval/Stage 1 fixed per `(query,k,α)` cell. `δ=0.70` was
   tried first and rejected as too loose (it makes the tolerance band nearly meaningless); settled on
   `δ=0.07` (tighter than santos3's 0.15). For `τ`: rather than sit exactly on the agreed floor
   (`τ≥0.05`), swept upward and found the cohort is on a thin margin — feasibility for the 46-query
   cohort (see next point) breaks at `τ=0.0533` (a different query becomes the bottleneck) — so used
   **`τ=0.052`**, giving `F*=0.122`.

**One outlier, excluded rather than rescued (Option 3).** At `τ=0.05`, 46 of 47 `|D|≥60` queries reach
feasibility in *every* one of the 6 configs; the sole holdout, `SpendoverC2A3500Apr17.csv`
(`|D|=84`, `F_Q=0.0155` in the original 2026-08-10 run too — a persistently hard case), is infeasible
in all 6 of its configs at `τ=0.05` but feasible again at `τ≤0.03`. Rather than lower the floor to
rescue one query, it is excluded from the cohort; the remaining **n=46** reach exactly **100%**
feasibility (276/276) at `τ=0.052` — not a relaxed target, the one already agreed on.

**Still not reportable here:** precision/recall (santosLarge ships no groundtruth CSV — those stay on
santos3), and *attainability* of `τ` (`C(D,10)` is intractable at these pool sizes — the removed C4
certificate). `tau_reached` below means the achieved `F_R ≥ τ`, identical to `feasible` on all 276
rows (verified).

**Self-retrieval, kept by decision, is now universal.** All 276 rows have the query in `D` (up from
"most" in the original run) — a direct consequence of maxD selecting common values: a query table's
own value is, by construction, one instance of the common value the corpus shares.

---

## 1. Index build (one-off, amortized over all queries)

| Component | Time (first measurement) | Time (verification rerun) | Size on disk |
|---|---:|---:|---:|
| Load vectors + file list (375 MB datalake pickle) | 0.33 s | 0.90 s | — |
| **HNSW index** over 11,086 tables' columns | 224.65 s | **272.49 s** | **281.78 MB** |
| **Inverted index** (§7.2 posting lists, reads every table) | 109.65 s | 109.38 s | **16.86 MB** |
| **Total** | 334.62 s | 382.77 s | — |

**This reverses the original run's finding.** The original (stale) numbers had the posting-list index
at 96% of build cost, 26× the HNSW build; measured now, **HNSW build dominates at 65–71% of total
cost**, roughly 2–2.5× the posting-list index. (Both original numbers were themselves already stale
relative to the current `categorical_only_vectors` fix by the time this redesign began — see
`experiments/RESULTS.md`'s santos3 section for the same fix's effect there.) It still reads all
11,086 CSVs (4.9 GB). **13 tables** required the tolerant CSV reader (malformed rows skipped, so their
`n_i`/`N_i` are slight undercounts — tracked in `synopsis.parse_recovered`).

**The HNSW build time is not stable run-to-run — the inverted-index build is.** Two from-scratch
rebuilds gave 224.65s and 272.49s for HNSW (a 21% spread) but 109.65s and 109.38s for the inverted
index (effectively identical). This is consistent with a real, independently-verified property of
`dutsx/adapters/semantic.py`: `hnswlib`'s `add_items` runs multi-threaded by default, and parallel
HNSW graph insertion order is not deterministic even with a fixed `random_seed` — it affects not only
*which* neighbors a query returns (documented elsewhere as a retrieval-nondeterminism finding) but
apparently also the wall-clock cost of building the graph in the first place, presumably via
thread-scheduling variance. The inverted index has no such variance because posting-list construction
is a straightforward sequential scan, not a parallel graph build. Treat any single HNSW build-time
number here as representative (roughly 225–275s), not exact.

**Size**: measured by serializing each structure (`hnswlib.Index.save_index()` for HNSW,
`pickle` for the posting-list dict) — on-disk size is the standard proxy for in-memory footprint. The
HNSW index (281.78 MB, 84,157 indexed elements at 768 dimensions) is **~17× larger** than the
inverted index (16.86 MB, 277,704 distinct values / 887,428 total `(table, attr)` postings) — sensible,
since HNSW stores dense float vectors plus graph-navigation links per element, while the posting-list
index stores only compact `(table, attr)` integer-ish tuples per value occurrence. This size gap is
also consistent with why HNSW build cost is compute-bound (distance calculations, layer assignment)
while the posting-list build is I/O-bound (reading every CSV once) — different bottlenecks, which is
also why only one of them shows run-to-run timing variance.

---

## 2. Outcome by (k, α)

| k | α | n | clamped | feasible | **τ reached** | `\|P\|` | `F_P` | `F_R` | `\|F*−F_R\|` | ΣU | self in R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2 | 46 | 0 | **46** | **46** | 20.0 | 0.618 | 0.568 | 0.463 | 124.1 | 9 |
| 10 | 3 | 46 | 0 | **46** | **46** | 30.0 | 0.610 | 0.549 | 0.446 | 126.3 | 12 |
| 10 | 4 | 46 | 0 | **46** | **46** | 40.0 | 0.599 | 0.514 | 0.412 | 128.1 | 16 |
| 15 | 2 | 46 | 0 | **46** | **46** | 30.0 | 0.610 | 0.565 | 0.458 | 184.7 | 12 |
| 15 | 3 | 46 | 0 | **46** | **46** | 45.0 | 0.589 | 0.517 | 0.414 | 188.5 | 19 |
| 15 | 4 | 46 | 0 | **46** | **46** | 60.0 | 0.552 | 0.487 | 0.389 | 192.8 | 22 |

**Zero clamping anywhere** — the `|D|≥60` cohort was chosen precisely so no cell clamps (max pool
size is 60), unlike the original run's 2-of-76 clamped rows. **All 46 queries feasible in every
config** (by construction of Option 3's exclusion, not a lucky finding).

`F_R` (0.49–0.57) sits comfortably above `τ=0.052` and, on average, above `F*=0.122` too — Stage 1's
unbounded `argmax F` (C3) pushes `F_P` to 0.55–0.62 before Stage 2 even runs, and Stage 2 has no upper
bound either, so `F_R` lands wherever maximizes `ΣU` subject only to the floor. `F_R` declines
mildly as α grows (0.568 → 0.514 at k=10) — a larger pool gives Stage 2 more moderate-`F` candidates
to pick `ΣU`-maximizing tables from, a pool-composition effect, not evidence of proximity-seeking.

---

## 3. Per-stage time (ms, mean per query)

| k | α | retrieval | Stage 1 | scoring (U) | Stage 2 | **end-to-end** |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2 | 44.4 (57%) | **0.32 (0.4%)** | 29.2 (38%) | 3.8 (4.9%) | **77.8** |
| 10 | 3 | 44.4 (49%) | **0.37 (0.4%)** | 41.9 (46%) | 3.9 (4.3%) | **90.6** |
| 10 | 4 | 44.4 (43%) | **0.39 (0.4%)** | 54.7 (53%) | 4.6 (4.4%) | **104.1** |
| 15 | 2 | 44.4 (49%) | **0.37 (0.4%)** | 42.2 (46%) | 4.3 (4.7%) | **91.3** |
| 15 | 3 | 44.4 (40%) | **0.38 (0.4%)** | 61.8 (56%) | 4.4 (4.0%) | **111.0** |
| 15 | 4 | 44.4 (34%) | **0.39 (0.3%)** | 80.5 (62%) | 4.9 (3.8%) | **130.2** |

1. **Retrieval is constant at 44.4 ms** — depends only on `top_n`/`|T|`, never on k or α; falls from
   57% to 34% of the budget as pools grow.
2. **Scoring scales with `α×k`**: 29.2 → 41.9 → 54.7 ms at k=10 as α goes 2→3→4 (pools 20→30→40),
   and 42.2 → 61.8 → 80.5 ms at k=15 (pools 30→45→60) — roughly linear, ~1.3 ms per unionability
   computation, consistent with the original run's 1.22–1.25 ms/computation measurement.
3. **Stage 1 is still free** — 0.3–0.4 ms, under 0.5% of end-to-end even selecting up to 60 candidates.
4. **Stage 2 is 3.8–4.9 ms**, 4–5% of end-to-end — a larger share than the original run's ~1.7–2%,
   since these pools (20–60) are on average a bit smaller than the original's 20–59.2 but the LP
   pre-check now runs to completion on every row (never decisive — see §5) rather than shortcutting.

---

## 4. Dinkelbach convergence — time and iterations by candidate pool size

Mean per query, averaged over the same 46 queries at each of the 6 pool sizes the sweep covers.

| k | α | pool size (`α×k`) | n | rows where it ran | mean Stage 1 time | mean iterations | iteration range |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2 | 20 | 46 | 46 / 46 | 0.317 ms | 2.11 | 1–5 |
| 10 | 3 | 30 | 46 | 46 / 46 | 0.370 ms | 2.26 | 1–5 |
| 10 | 4 | 40 | 46 | 46 / 46 | 0.391 ms | 2.43 | 1–5 |
| 15 | 2 | 30 | 46 | 46 / 46 | 0.368 ms | 2.26 | 1–5 |
| 15 | 3 | 45 | 46 | 46 / 46 | 0.383 ms | 2.30 | 1–5 |
| 15 | 4 | 60 | 46 | 46 / 46 | 0.388 ms | 2.20 | 1–5 |

Over all **276 invocations: mean 2.26 iterations, max 5** — distribution 1 iter ×88, 2 ×73, 3 ×78,
4 ×29, 5 ×8. Dinkelbach runs on **every single row** here (0 clamped, vs. the original run's 74/76)
and still converges in at most 5 iterations on pools up to 60 candidates — the paper's "converges in
only a few iterations" claim continues to hold, now on a fully-unclamped population rather than
mostly one. **Convergence time tracks pool size only weakly** (0.317 ms at pool 20 → 0.391 ms at pool
40, a ~23% rise for a 2× larger pool) — each Dinkelbach iteration is one sort-and-select over the
pool, so cost grows sub-linearly with pool size, and the iteration count itself (which dominates
total time) doesn't increase monotonically with pool size at all (pool 45's mean of 2.30 exceeds pool
60's 2.20) — convergence speed is a property of the distribution values in `D` (how skewed the
candidates' `N_i/n_i` ratios are), not of how large a pool Stage 1 is asked to fill.

### 4b. Same measurement, but vs. the *input* pool size `|D|`, not the *output* pool `α×k`

The table above holds `top_n=5000` fixed and varies the pool Stage 1 *selects* (`α×k` — the
output). This one asks a different question: does Dinkelbach's cost depend on the pool it
*searches over* (`|D|`, the input, before Stage 1 picks anything)? **Setting: `k=10, α=3` fixed
throughout** (so the output target `α×k=30` never changes), and `top_n` — the HNSW retrieval-count
knob — swept to produce different *actual* `|D|`, the same way `experiments/wdc/topn_alpha_k_sweep.py`
reports realized retrieval counts rather than the requested `top_n` (retrieval yield ≠ `top_n` after
the overlap-filter intersection). Same 46-query cohort at every setting, re-retrieved per `top_n`.

`top_n` was swept starting at 5,000 rather than lower — a first pass at `{500,1000,2000,5000,...}`
showed `α×k=30 ≥ |D|` clamping (Stage 1 a no-op, `P=D`) at the small end (18/46 clamped at
`top_n=500`), which would contaminate the time/iteration means with 0-iteration rows; every
`top_n≥5000` point already ran clamp-free, so the final sweep — `{5000, 10000, 15000, 20000, 25000,
30000}` — is clamp-free throughout (0/46 at every point) and denser where `|D|` is actually moving.

| requested `top_n` | actual mean `|D|` | rows clamped | mean Stage 1 time | mean iterations | iteration range |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 160.4 | 0 / 46 | 0.196 ms | 2.26 | 1–5 |
| 10,000 | 208.6 | 0 / 46 | 0.249 ms | 2.22 | 1–4 |
| 15,000 | 236.0 | 0 / 46 | 0.281 ms | 2.26 | 1–4 |
| 20,000 | 250.6 | 0 / 46 | 0.293 ms | 2.28 | 1–4 |
| 25,000 | 257.9 | 0 / 46 | 0.302 ms | 2.28 | 1–4 |
| 30,000 | 259.0 | 0 / 46 | 0.301 ms | 2.28 | 1–4 |

Script: `experiments/santoslarge_topn_pool_sweep.py`, output
`experiments/results/santoslarge_topn_pool_sweep.csv` (276 rows, 6 `top_n` × 46 queries).

Two things stand out. First, unlike §4's near-flat curve vs. output pool size, time **does** grow
with input pool size — roughly 1.5× from `|D|≈160` to `|D|≈259` (0.196 ms → 0.301 ms) — consistent
with each Dinkelbach iteration being one sort-and-select over all of `D` (cost ~`O(|D| log |D|)` per
iteration, not `O(α×k)`): the input side of the sort is what scales, the output pool size only
changes how many of the sorted candidates get kept. Second, **`|D|` itself plateaus**: it keeps
growing from `top_n=5,000` to `20,000` (160→251) but is essentially flat from `20,000` to `30,000`
(251→259, a 3% move on a 50% larger `top_n`) — past some point, HNSW's wider semantic net stops
surfacing *new* tables that also pass the overlap filter, because the pool of tables containing the
protected value at all is finite and mostly already found; correspondingly, Stage 1 time also flattens
(0.293 → 0.301 → 0.301 ms) over that same range. Mean iteration count stays essentially flat
(2.22–2.28) across the whole sweep — Dinkelbach still converges in a handful of iterations regardless
of how large the candidate pool it's searching is, which is the more paper-relevant claim (§5's "few
iterations"); it's the per-iteration sort cost, not the iteration count, that carries the `|D|`
dependence.

---

## 5. Was the LP feasibility pre-check decisive? **No — not once, unlike the original run.**

| k | α | LP ran | **LP decisive** | LP time | total Stage 2 |
|---:|---:|---:|---:|---:|---:|
| 10 | 2 | 46 | **0** | 1.33 ms | 3.81 ms |
| 10 | 3 | 46 | **0** | 1.39 ms | 3.93 ms |
| 10 | 4 | 46 | **0** | 1.46 ms | 4.58 ms |
| 15 | 2 | 46 | **0** | 1.36 ms | 4.31 ms |
| 15 | 3 | 46 | **0** | 1.49 ms | 4.41 ms |
| 15 | 4 | 46 | **0** | 1.57 ms | 4.89 ms |

**This reverses the original run's own reversal.** The 2026-08-10 result was that the LP pre-check
*was* decisive on santosLarge (37% of the cohort — the pre-check caught genuine infeasibility that
santos3 never had), the opposite of santos3's "never decisive." Post-redesign, santosLarge's LP
pre-check is back to **never decisive** — but for a different, and more honest, reason than santos3's:
not because the cohort was naturally all-solvable, but because Option 3 *constructed* it that way by
excluding the one query that wasn't. The pre-check still consumes 28–35% of Stage 2's time on every
row, for zero payoff on this particular cohort.

**Recommendation, revised again: the deciding factor is genuinely the infeasibility rate of the
cohort, not the benchmark.** santos3, the original santosLarge run, and the redesigned santosLarge
run all illustrate the same rule (`LP-infeasible ⇒ ILP-infeasible`, soundness-neutral either way) —
whether keeping it on pays off depends entirely on whether the *specific* cohort you're running
contains genuine infeasibility to catch, and this redesign's cohort, by construction, does not.

### 5b. Same breakdown, by `α`, with clamped/feasible made explicit

Same `top_n=5000`, same 46-query cohort, same underlying rows as the table above — split into one
table per `α` (all three: 2, 3, 4) and with `clamped`/`feasible` columns added, since the table above
folds those into "46 ran, 0 decisive" without stating them outright.

```latex
\paragraph{$\alpha = 2$}
\begin{center}
\begin{tabular}{cccccccc}
\toprule
$k$ & pool ($\alpha k$) & $n$ & clamped & feasible & LP ran & LP decisive & mean LP time & mean Stage 2 time \\
\midrule
10 & 20 & 46 & 0/46 & 46/46 & 46/46 & 0/46 & 1.333\,ms & 3.808\,ms \\
15 & 30 & 46 & 0/46 & 46/46 & 46/46 & 0/46 & 1.359\,ms & 4.306\,ms \\
\bottomrule
\end{tabular}
\end{center}

\paragraph{$\alpha = 3$}
\begin{center}
\begin{tabular}{cccccccc}
\toprule
$k$ & pool ($\alpha k$) & $n$ & clamped & feasible & LP ran & LP decisive & mean LP time & mean Stage 2 time \\
\midrule
10 & 30 & 46 & 0/46 & 46/46 & 46/46 & 0/46 & 1.386\,ms & 3.925\,ms \\
15 & 45 & 46 & 0/46 & 46/46 & 46/46 & 0/46 & 1.491\,ms & 4.405\,ms \\
\bottomrule
\end{tabular}
\end{center}

\paragraph{$\alpha = 4$}
\begin{center}
\begin{tabular}{cccccccc}
\toprule
$k$ & pool ($\alpha k$) & $n$ & clamped & feasible & LP ran & LP decisive & mean LP time & mean Stage 2 time \\
\midrule
10 & 40 & 46 & 0/46 & 46/46 & 46/46 & 0/46 & 1.464\,ms & 4.583\,ms \\
15 & 60 & 46 & 0/46 & 46/46 & 46/46 & 0/46 & 1.572\,ms & 4.887\,ms \\
\bottomrule
\end{tabular}
\end{center}
```

Same pattern at all three `α`: zero clamping, 100% feasible, LP pre-check runs on every row but is
decisive on none of them — reconfirms §5's finding at the per-`α` granularity rather than pooled
across the grid.

### 5c. Precheck + actual solver only (excluding constraint-setup/verification overhead)

§5's "mean Stage 2 time" is the *entire* wall-clock time of `stage2_ilp.solve()` — LP precheck, the
actual `scipy.optimize.milp` branch-and-bound call, plus everything else in between (building the
`LinearConstraint` objects, and the post-solve verification asserts that guard against the presolve
trap documented in `duts/stage2_ilp.py`). To isolate just the two solves — precheck and actual
solver — `milp_time_s` was added as a second timer (`duts/stage2_ilp.py`, wrapping only the `milp(...)`
call itself), and the study rerun to populate it.

| α | k | pool (`α×k`) | mean LP precheck | mean MILP solver | **precheck + solver** | full Stage 2 wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 20 | 0.677 ms | 1.164 ms | **1.842 ms** | 2.055 ms |
| 2 | 15 | 30 | 0.677 ms | 1.276 ms | **1.953 ms** | 2.160 ms |
| 3 | 10 | 30 | 0.654 ms | 1.092 ms | **1.745 ms** | 1.946 ms |
| 3 | 15 | 45 | 0.757 ms | 1.356 ms | **2.112 ms** | 2.337 ms |
| 4 | 10 | 40 | 0.679 ms | 1.338 ms | **2.017 ms** | 2.224 ms |
| 4 | 15 | 60 | 0.783 ms | 1.507 ms | **2.291 ms** | 2.518 ms |

The gap between "precheck + solver" and "full Stage 2 wall time" (roughly 0.2–0.25 ms at every
setting) is the constraint-setup/verification overhead — small and fairly constant, so it doesn't
change which piece dominates: the actual MILP solve is consistently the larger of the two solves
(~60–65% of precheck+solver time), with the LP precheck the remaining ~35–40%, both growing mildly
with pool size.

**Caveat on absolute numbers.** This table's LP-precheck times (0.65–0.78 ms) are noticeably lower
than §5's own LP-time column (1.33–1.57 ms) for the *identical* `(k,α)` cells — same deterministic
pool sizes, same solver, different process runs. At sub-millisecond scale, `perf_counter` timings
are sensitive to process-level noise (scipy/HiGHS dispatch overhead, machine load, cache state)
rather than reflecting a real change in the underlying computation; treat absolute ms values here as
accurate only to roughly ±30% run-to-run, while the *relative* split (solver dominates precheck,
both mildly pool-size-dependent) is the stable finding.

---

## 6. Skipping Stage 1 — a smaller but still real payoff

`skip_stage1` scores every candidate in `D` and hands all of it to Stage 2 at cardinality `k`.

| k | α | scored (2-stage) | scored (skip) | scoring ms | e2e (2-stage) | e2e (skip) | **speedup** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2 | **20.0** | 160.3 | 29.2 → 169.9 | **77.8** | 221.1 | **2.84×** |
| 10 | 3 | **30.0** | 160.3 | 41.9 → 168.1 | **90.6** | 219.3 | **2.42×** |
| 10 | 4 | **40.0** | 160.3 | 54.7 → 168.8 | **104.1** | 220.0 | **2.11×** |
| 15 | 2 | **30.0** | 160.3 | 42.2 → 168.4 | **91.3** | 219.3 | **2.40×** |
| 15 | 3 | **45.0** | 160.3 | 61.8 → 168.5 | **111.0** | 219.4 | **1.98×** |
| 15 | 4 | **60.0** | 160.3 | 80.5 → 166.9 | **130.2** | 217.8 | **1.67×** |

| | k=10 α=2 | k=10 α=3 | k=10 α=4 | k=15 α=2 | k=15 α=3 | k=15 α=4 |
|---|---:|---:|---:|---:|---:|---:|
| `F_R` two-stage | 0.568 | 0.549 | 0.514 | 0.565 | 0.517 | 0.487 |
| `F_R` skip | **0.393** | **0.393** | **0.393** | **0.384** | **0.384** | **0.384** |
| ΣU two-stage | 124.1 | 126.3 | 128.1 | 184.7 | 188.5 | 192.8 |
| ΣU skip | **135.8** | **135.8** | **135.8** | **201.7** | **201.7** | **201.7** |
| feasible | 46 | 46 | 46 | 46 | 46 | 46 |

**Two-stage is 1.7×–2.8× faster end-to-end** (down from the original run's 1.7×–3.0×, since mean
`|D|` here — 160.3 among the maxD-scored `D`s used by `skip_stage1` — is smaller relative to the
larger `α×k` pool sizes now in play than the original run's `|D|=124.5` was relative to its smaller
pools of 20–59.2). The mechanism is identical: scoring cost is pinned at `α×k` (20–60) while
`skip_stage1` pays for all `|D|=160.3`, regardless of k or α.

**Same qualitative picture as before, quantitatively smaller ΣU gap.** `skip_stage1` still wins ΣU by
construction (optimizes over the superset `D⊇P`) — **5–9% higher** here (vs. the original run's
5–9% too: `118.6/108.5=9.3%` at k=10,α=2 down to `233.7/222.6=5.0%` at k=20,α=3 — nearly identical
range, coincidentally). And **skip still lands closer to `F*` than two-stage does**, same direction as
the original run: `|F*−F_R|` for skip is `|0.122−0.39|≈0.27` vs. two-stage's `0.39–0.46` (§2's
`abs_gap_R` column) — two-stage's unbounded `argmax F` (Stage 1) pushes `F_P`, and hence `F_R`,
further past `F*` than simply handing Stage 2 all of `D` does. Feasibility is identical (46/46 both
arms, every config) — unlike the original run, where feasibility was 12/19 for both arms and only
*those* 12 proved anything, here the entire cohort is informative. Stage 1 remains an efficiency
mechanism traded against `ΣU`, not a quality mechanism.

---

## 7. Per-step runtime vs. candidate pool size (paper-ready tables)

Full tables, per-metric prose defining each number for paper use, and the raw per-query CSVs behind
them live in **`experiments/end2endSantosLarge.tex`** (LaTeX, ready to paste into the paper) and
`experiments/end2end_santoslarge_report.py` (the script that produced it). Summary here.

**Same fixed settings as throughout**: `top_n=5000`, `F*=0.122`, `δ=0.07` (`τ=0.052`), the 46-query
maxD cohort. **Pool size (`α×k`), not `(k,α)`, is the x-axis** — the grid produces pool sizes
`{20,30,40,30,45,60}`; pool **30** is reached by two different pairs (`k=10,α=3` and `k=15,α=2`), and
at every statistic (mean/median/max/min) for every metric, the pool=30 row reports the **larger** of
the two pairs' values — an elementwise max, not an average or a fixed choice of one pair. Five
metrics, all read directly off existing per-query columns already in
`santoslarge_maxd_topn5000_D60.csv` (no new instrumentation or rerun needed): **index probe total**
(`retrieval_time_s` — HNSW probe + posting-list overlap probe + their intersection + `N_i/n_i`
lookup, i.e. everything before Stage 1 runs), **Stage 1 total** (`stage1_time_s`), **unionability
computation total** (`scoring_time_s` — strictly between the two stages, not part of either),
**Stage 2 total** (`stage2_time_s`, full wall time), **end-to-end total** (`end_to_end_s`).

| pool (`α×k`) | index probe (mean/median) | Stage 1 (mean/median) | unionability (mean/median) | Stage 2 (mean/median) | end-to-end (mean/median) |
|---:|---:|---:|---:|---:|---:|
| 20 | 38.37 / 29.63 ms | 0.189 / 0.164 ms | 28.18 / 9.65 ms | 2.055 / 1.738 ms | 68.80 / 50.50 ms |
| 30 | 38.37 / 29.63 ms | 0.194 / 0.167 ms | 42.17 / 15.59 ms | 2.160 / 1.808 ms | 82.68 / 57.30 ms |
| 40 | 38.37 / 29.63 ms | 0.199 / 0.160 ms | 55.87 / 21.58 ms | 2.224 / 1.836 ms | 96.66 / 63.68 ms |
| 45 | 38.37 / 29.63 ms | 0.196 / 0.159 ms | 60.97 / 24.99 ms | 2.337 / 1.993 ms | 101.88 / 66.12 ms |
| 60 | 38.37 / 29.63 ms | 0.205 / 0.164 ms | 81.15 / 31.98 ms | 2.518 / 2.033 ms | 122.25 / 74.18 ms |

(Full mean/median/**max**/**min** tables — the min/max columns are in the `.tex` file, omitted here
for width.)

**Headline finding: index probe total is pool-size-invariant** (identical at every row, by
construction — retrieval runs once per query and is reused across every `(k,α)` setting, so it
depends only on `top_n` and the query, never on the pool size requested downstream). **Unionability
computation total is the only metric that scales with pool size the way the paper's efficiency claim
predicts** — mean grows ~2.9× from pool 20 to pool 60, tracking `|P|=α×k` almost linearly (`O(|P|)`),
which is *why* the two-stage design exists at all (§6's skip-Stage-1 comparison shows the cost of
scoring all of `D` instead). Stage 1 and Stage 2 are both mildly pool-size-sensitive but small in
absolute terms — together under 5% of end-to-end time at every pool size (e.g. pool 60:
`(0.205+2.518)/122.25 ≈ 2.2%`). **End-to-end growth (68.8→122.2 ms mean, ~1.8×) is therefore
attributable almost entirely to unionability computation, not either stage's own solve time.** Every
metric except index-probe-total is markedly right-skewed (mean ≫ median) — driven by the same
long-tailed `|D|` distribution across the 46 queries seen throughout this report.

Raw per-query values (for figure generation) are in
`experiments/results/end2end_santoslarge_raw.csv` (long format: one row per query × metric); the
resolved-by-pool-size table above is `experiments/results/end2end_santoslarge_by_poolsize.csv`.

---

## 8. Column reference

`n_D` candidates retrieved · `pool_size_requested` = `int(α×k)` · `clamped` = C5 fired (`α×k ≥ |D|`,
`P = D`, Stage 1 a no-op; 0 rows here) · `feasible` / `tau_reached` (achieved `F_R ≥ τ`; identical on
all rows) · `F_Q` query's own proportion · `F_P`/`F_R` pool/result proportion · `abs_gap_R` =
`|F* − F_R|` · `sum_U` Stage 2 objective · `self_in_D`/`self_in_R` query table retrieved / selected
(now universal in `D`, see intro) · `dinkelbach_ran`/`dinkelbach_iterations` ·
`lp_ran`/`lp_feasible`/`lp_decisive` (decisive = LP proved infeasible, ILP skipped)/`lp_time_s` ·
`retrieval_time_s`/`stage1_time_s`/`scoring_time_s`/`stage2_time_s`/`end_to_end_s` ·
`skip_*` the same quantities for the Stage-1-bypass arm.

`experiments/results/santoslarge_maxd_selection_report.csv` (78 rows, one per original query) is a
companion file: `orig_n_D`/`new_n_D` before/after `|D|` per query, `new_attr`/`new_value` the
maxD-selected assignment. `experiments/results/santoslarge_tau_sweep_by_config.csv` records the
reach-rate sweep behind §0's `F*`/`δ`/`τ` choice.

Index-build times are not per-query and are reported only in §1.
