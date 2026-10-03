# DUTS on WDC Web Table Corpus — redesigned scalability results (`tier_10k`, `tier_100k`, `tier_1m`)

**2026-09-17.** Supersedes `RESULTS-wdc.md` and `RESULTS-wdc-sweep.md`, both of which used a
deliberate relaxation of §7.3's retrieval rule (`D = D_sem ∪ D_ovl` instead of the paper-faithful
`D_sem ∩ D_ovl`) to work around thin retrieval yield on this uncurated corpus. This report replaces
that relaxation with santosLarge's maxD methodology instead — §7.3-faithful intersection retrieval
throughout — applied to a query set that is **fixed across all three tiers**, not independently
chosen per tier. Old reports and their result files are preserved, not deleted, under
`experiments/results/_backup_pre_maxd_wdc/`.

Structure mirrors `RESULTS-santoslarge.md`: this file grows section by section as each part of the
study is run; §0 records the (now-settled) methodology, §1 is the first result section.

## 0. Methodology: query selection, grid, and `F*/δ/τ`

**Why queries must be shared across tiers, not chosen per tier.** `tier_10k ⊂ tier_100k ⊂ tier_1m`
is an exact nesting (verified directly by file-list comparison: all 10,001 `tier_10k` files appear
in `tier_100k`; all 100,000 `tier_100k` files appear in `tier_1m`). A valid scale comparison holds
the *query* fixed and observes how the surrounding datalake's growth changes outcomes — not a
different, independently-optimized query per tier (an earlier attempt at this redesign did exactly
that and had to be corrected: see git history / session notes, not repeated here since it never
shipped as a result).

**Query selection** (`experiments/wdc/wdc_maxd_selection_shared.py`):
1. 60 candidate query tables drawn from `tier_10k`'s own table universe only (seed 42) — guaranteed
   present in every larger nested tier by construction.
2. For each candidate, maxD attribute/value selection (identical algorithm to santosLarge: rank
   every `(categorical attr, value)` pair by the free posting-list-size proxy, verify the top 20 by
   real intersection retrieval, keep whichever gives the largest `|D|`) runs against the **larger
   `tier_100k` context** — richer posting-list statistics than `tier_10k` alone would give.
3. That one fixed `(table, attr, value)` triple per candidate is then evaluated for real `|D|`
   against every tier's own separate index.

**A real, explainable non-monotonicity was found and is not hidden**: one candidate showed
`|D|_10k = 189 > |D|_100k = 46`, even though `tier_10k ⊂ tier_100k` guarantees `tier_100k` has at
least every match `tier_10k` has. Cause: `top_n=5000` is a fixed budget on the semantic (HNSW)
probe — `tier_100k` has 10× more tables competing for those same 5000 slots, so a genuine
overlap-match that fits inside `tier_10k`'s smaller top-5000 can get crowded out of `tier_100k`'s.
Growing the corpus grows the overlap filter's raw hit count but can *shrink* the semantic probe's
effective recall at a fixed `top_n` — both real, competing effects.

**Grid — revised to an unclamped-by-construction design (santosLarge-style rigor), not the original
ambitious one.** An initial 30-cell grid (`k∈{10..50}×α∈{10..60}`, max pool 3,000) was tried first,
sized deliberately larger than santosLarge's on the theory that WDC's scale could support it — but it
only checked feasibility against whatever pool resulted, clamped or not, unlike santosLarge's own
cohort criterion (`|D|≥60` chosen specifically so every retained query stays unclamped across its
whole grid). Measured directly: **87.1% of that grid's 900 `(query, cell)` pairs were already
clamped** at `top_n=5000`, even the best query (`|D|=563`) had 20/30 cells clamped — the grid was
too ambitious for what this corpus/retrieval setup actually supports. Reverted to the santosLarge
discipline instead:

1. **Widen the candidate pool to 210** (60 originally drawn + 150 more, continuing the same
   `random.Random(seed=42)` shuffle over `tier_10k`'s table universe) — `experiments/wdc/wdc_maxd_selection_unclamped.py`,
   reusing the same maxD proxy-rank + top-20-real-retrieval-verify method, `top_n=5000` throughout.
2. **Use `min(D_10k, D_100k)` as the binding threshold** (the smaller tier is always at least as hard
   to clear) and derive the grid from the *observed* distribution rather than reusing the old grid:
   `k∈{5,10,20} × α∈{5,10}` — 6 cells (same count as santosLarge's grid), pools 25–200 — chosen
   because WDC's realized distribution (median 52, p75 280, p90 416, max 562) supports a materially
   larger ceiling than santosLarge's 60 once the candidate draw is widened, but nowhere near the
   original grid's 3,000.
3. **Retain every candidate with `min(D_10k, D_100k) ≥ 200`** (`= max(α×k)` for this grid) —
   guaranteed unclamped on *both* tiers at *every* cell, exactly mirroring santosLarge's criterion.
   Result: **67 of 210 candidates retained** (31.9%) — comparable to santosLarge's own 47/78 (60.3%)
   retention rate in spirit, though numerically lower since the corpus is far less curated.

**`F*/δ/τ` selection**: swept for **joint** feasibility on the new 6-cell grid + 67-candidate pool —
a query only counts if *every one* of its 6 grid cells is feasible on *both* `tier_10k` and
`tier_100k` simultaneously (`experiments/wdc/wdc_tau_sweep_shared.py`; feasibility depends only on
each candidate's `(N,n)`, never `U` — dummy `U=1.0` used). Chosen: **`F* = 0.20, δ = 0.10, τ = 0.10`**
(comfortably above the 0.05 floor, same round values used throughout this project) — **67/67
candidates fully feasible on both tiers simultaneously, at every one of the 6 cells: 100% retention.**

**Final cohort**: all **67 query tables** (same attribute, same value) used against every tier — no
outlier exclusion needed, unlike the original design's cohort work, since every candidate that
cleared the `|D|≥200` threshold is also fully feasible.

Files: `experiments/results/wdc_unclamped_candidate_pool.csv` (210 candidates, both tiers' `|D|` side
by side), `experiments/results/wdc_unclamped_final_cohort.csv` (the 67 kept),
`experiments/results/wdc_unclamped_selection_notes.md` (the threshold curve and full tradeoff
writeup).

**`tier_1m` construction.** Unlike the two smaller tiers, `tier_1m` had no embeddings or index built
at all before this report — the full pipeline (CSV conversion → GPU embedding extraction → metadata
store) had to be run from scratch:
- CSV conversion: 670.6s, 1,000,000/1,000,000 tables, 0 failed, 0 recovered.
- Embedding extraction: the single-process approach (`scripts/wdc_extract_via_santos_checkpoint.py`,
  reusing `starmie/extractVectors.py` completely unmodified) was **silently killed** (`exit -9`) after
  148 minutes at 64% progress — that script accumulates every embedding in memory before writing one
  output pickle, which doesn't scale to 1M tables. Root cause, confirmed later (§1 below): this
  account has a **hard 32GB per-process RSS limit** (`ulimit -m`, both soft and hard, not adjustable
  — `dmesg`/`journalctl` access is also blocked, so a generic system-wide-OOM explanation could not be
  directly confirmed or ruled out until this limit was found independently while debugging §1's HNSW
  build). Fixed via
  `scripts/wdc_extract_via_santos_checkpoint_chunked.py`: splits the tier into 100,000-table chunks
  (the size already proven safe by `tier_100k`'s own successful run), extracts each chunk as a fresh
  subprocess (so memory never accumulates across chunks), and concatenates the resulting pickles —
  no changes to the reused `extractVectors.py` itself. Total: 3,305.3s across 10 chunks (resumable —
  a mid-run environment reset lost the in-flight chunk 5, but chunks 0–4's completed `.pkl` outputs
  were safely on persistent storage and the script skipped straight to resuming at chunk 5).
- Metadata store build: hit a second, unrelated real bug — one table has a single CSV field
  exceeding Python's default `csv` module limit (128KB), which crashed the *unhandled* tolerant
  fallback path and killed the whole 1M-file build with no checkpointing (all progress lost, unlike
  the chunked extraction step). Fixed by wrapping that fallback in its own `try/except` so a
  genuinely unparseable file is skipped and counted rather than crashing the batch (matching this
  script's existing NUL-byte tolerance). Rerun: 373.85s total, **999,999/1,000,000 tables stored**, 1
  genuinely unparseable file skipped, 34 recovered via the existing NUL-byte tolerance.

---

## 1. Index build cost, all three tiers

Neither index is persisted by the normal pipeline — both are rebuilt in-memory on every
`build_wdc_context` call. Sizes measured by native serialization (`hnswlib.Index.save_index`;
`pickle.dump` on the postings dict) to a temp file, then deleted. Script:
`experiments/wdc/wdc_index_size_report.py`.

| tier | tables | HNSW build | HNSW size | inverted build | inverted size | distinct values | total postings |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tier_10k` | 10,000 | 13.9 s | 163.6 MB | 2.7 s | 8.75 MB | 141,737 | 329,064 |
| `tier_100k` | 100,000 | 200.8 s | 1,633.0 MB | 23.4 s | 76.6 MB | 995,957 | 3,347,443 |
| `tier_1m` | 999,999 | 1,284.8 s | 16,322.8 MB | 49.6 s | 670.5 MB | 6,444,399 | 33,453,833 |

**HNSW size scales almost perfectly linearly with table count throughout** (163.6 MB → 1,633.0 MB →
16,322.8 MB — ~10× at every 10× step) — expected, since each table contributes one fixed-size
embedding regardless of corpus scale. **HNSW build time scaling is *not* uniform across steps**:
~14.5× from `tier_10k`→`tier_100k` (superlinear, consistent with `O(n log n)` graph construction,
and matching the original (pre-redesign) `RESULTS-wdc.md`'s independently-measured ~14.7× at the
same step) but only ~6.4× from `tier_100k`→`tier_1m` (sublinear) — reported honestly as a real,
two-regime pattern rather than smoothed into one exponent; not investigated further here. **Inverted
index vocabulary grows sublinearly at every step**: distinct values 141,737 → 995,957 → 6,444,399
(~7× then ~6.5× per 10× tables) — common values keep getting reused rather than each new table
contributing an entirely new vocabulary, so the posting-list index's size grows more slowly than the
corpus itself.

**A genuine obstacle hit and fixed while measuring `tier_1m`.** The first two attempts at this
measurement were silently killed (`exit -9`, no traceback) at the exact same point — immediately
after `build_wdc_context` finished loading `tier_1m`'s vectors and synopsis, before the HNSW index's
size could even be measured. Root cause: this account has a **hard 32GB per-process RSS limit**
(`ulimit -m` — both soft and hard limits are exactly 33,554,432 KB, confirmed non-adjustable by this
account). `build_wdc_context`'s ordinary path keeps three independent ~16GB copies of `tier_1m`'s
vector data alive simultaneously: the raw loaded pickle, `categorical_only_vectors`'s explicit
`np.array(..., copy=True)` filtered copy, and `hnswlib.add_items`'s own internal copy — comfortably
exceeding 32GB even before `save_index()` does anything. Fixed with a memory-frugal measurement path
(`wdc_index_size_report.py::measure_tier_lean`) that frees each large intermediate (`del` +
`gc.collect()`) as soon as the next step no longer needs it — `vectors` is freed right after the
categorical-filtered copy is built, and that filtered copy is freed right after `hnswlib` finishes
copying what it needs internally. Peak RSS measured directly during the successful run: climbed to
~30.5 GB during the `add_items` bulk-copy phase (uncomfortably close to the 32GB ceiling, but under
it) before settling back to ~28 GB for the remainder of graph construction. This also retroactively
explains §0's embedding-extraction crash, which was previously attributed only generically to
"OOM" — same 32GB per-process ceiling, different code path.

---

## 2. Dinkelbach convergence vs. input pool size `|D|`, all three tiers

**Setting: `k=20, α=10` fixed** (output pool `α×k = 200`) — the largest cell of §0's finalized
6-cell grid (`k∈{5,10,20}×α∈{5,10}`), not the old `k=50/α=50/pool=2500` setting from a since-dropped
earlier draft of this section (that setting was never in-grid; `pool=2500` is 12.5× the grid's own
ceiling). `top_n` is swept over `{5000, 10000, 15000, 20000, 25000, 30000}` to produce different
*actual* `|D|`, same convention as every other top_n sweep in this project. Same 67-query
unclamped-by-construction cohort throughout (`experiments/results/wdc_unclamped_final_cohort.csv`),
re-retrieved per `(tier, top_n)`. Script: `experiments/wdc/wdc_topn_pool_sweep.py`.

| tier | top_n | mean `\|D\|` | n_ran | n_clamped (of 67) | mean stage-1 time | mean iters | iter range |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tier_10k` | 5,000 | 387.0 | 67 | 0 | 0.67 ms | 2.16 | 1–3 |
| `tier_10k` | 10,000 | 618.4 | 67 | 0 | 1.26 ms | 2.48 | 1–4 |
| `tier_10k` | 15,000 | 776.2 | 67 | 0 | 1.53 ms | 2.48 | 1–4 |
| `tier_10k` | 20,000 | 854.1 | 67 | 0 | 1.72 ms | 2.57 | 1–4 |
| `tier_10k` | 25,000 | 877.9 | 67 | 0 | 1.80 ms | 2.57 | 1–4 |
| `tier_10k` | 30,000 | 882.6 | 67 | 0 | 1.88 ms | 2.58 | 1–4 |
| `tier_100k` | 5,000 | 925.5 | 67 | 0 | 1.45 ms | 1.99 | 1–4 |
| `tier_100k` | 10,000 | 1,409.1 | 67 | 0 | 2.25 ms | 2.22 | 1–4 |
| `tier_100k` | 15,000 | 1,755.3 | 67 | 0 | 3.30 ms | 2.31 | 1–4 |
| `tier_100k` | 20,000 | 2,048.0 | 67 | 0 | 4.08 ms | 2.37 | 1–4 |
| `tier_100k` | 25,000 | 2,371.9 | 67 | 0 | 5.33 ms | 2.42 | 1–4 |
| `tier_100k` | 30,000 | 2,630.7 | 67 | 0 | 6.44 ms | 2.51 | 1–4 |
| `tier_1m` | 5,000 | 1,338.7 | 62 | 5 | 2.17 ms | 1.58 | 1–4 |
| `tier_1m` | 10,000 | 2,629.7 | 66 | 1 | 4.82 ms | 1.77 | 1–4 |
| `tier_1m` | 15,000 | 3,440.9 | 65 | 2 | 6.77 ms | 1.85 | 1–4 |
| `tier_1m` | 20,000 | 4,215.8 | 66 | 1 | 7.87 ms | 1.88 | 1–4 |
| `tier_1m` | 25,000 | 4,971.4 | 65 | 2 | 11.04 ms | 1.86 | 1–5 |
| `tier_1m` | 30,000 | 5,633.9 | 65 | 2 | 11.56 ms | 1.97 | 1–5 |

Full per-query CSV: `experiments/results/wdc_topn_pool_sweep.csv` (1,206 rows).

**`|D|` scales with corpus size, roughly proportionally.** At `top_n=30000`, mean `|D|` goes
882.6 → 2,630.7 → 5,633.9 across the three tiers (~3.0× then ~2.1× per 10× tables) — sublinear like
§1's inverted-index vocabulary growth, not the ~10×-per-step scaling `tier_10k`'s own HNSW size
showed, because `|D|` here is bounded by both the overlap filter's hit count *and* the fixed
`top_n` semantic-probe budget (§0's non-monotonicity discussion).

**Dinkelbach's iteration count stays essentially flat across three orders of magnitude of corpus
size and up to 6.4× growth in `|D|` itself** — mean iterations range 1.58–2.58 everywhere, with the
same 1–4 (occasionally 1–5 on `tier_1m`) range seen throughout. This is the direct empirical
substantiation of the paper's "converges in only a few iterations" claim (§5) holding at real WDC
scale, not just on santos-family benchmarks.

**Stage-1 wall-clock time grows with `|D|`, as expected (`O(|D| log |D|)` sort-based selection,
Invariants/C5), but stays sub-12ms even at `tier_1m`'s largest `|D|`** — 0.67ms → 6.44ms → 11.56ms
at `top_n=30000` across the three tiers, tracking `|D|`'s own ~3×/~2× growth rather than iteration
count (which barely moves). Confirms Stage 1's cost is governed by `|D|`, never by `|T|` directly —
`tier_1m` has 100× `tier_10k`'s table count but only ~13× its `top_n=30000` Stage-1 time.

**Clamping (`pool_size=200 ≥ \|D\|`) essentially disappears at this grid's own ceiling, unlike the
dropped `k=50/α=50/pool=2500` draft.** `tier_10k` and `tier_100k` show **zero** clamped cells across
every `top_n` — expected, since the cohort was built to guarantee `min(D_10k, D_100k) ≥ 200` at
every cell (§0). `tier_1m` shows a small amount of clamping only at the smallest `top_n=5000`
(5/67) and 1–2/67 at every larger `top_n` — a genuine, mild instance of §0's documented
non-monotonicity effect (a 100×-larger corpus crowds a fixed `top_n` semantic-probe budget), not a
bug: those queries' `|D|` on `tier_1m` at `top_n=5000` dipped just under 200 even though it cleared
200 on both smaller tiers by construction (the cohort's guarantee was built against `tier_10k`
and `tier_100k` only, before `tier_1m` existed).

**A genuine infrastructure obstacle, worse than §1's, was hit and fixed while building `tier_1m`'s
context for this section.** Unlike §1's index-*size* measurement (which only had to build the index
once, then serialize and discard it), a working `RunnerContext` capable of running real queries has
to stay resident. Five build attempts were needed:
1. The ordinary `build_wdc_context` path (three ~16GB copies alive at once, per §1) — SIGKILLed.
2. A pop-as-you-go filtered dict (`experiments.context.categorical_only_vectors`'s zero-and-keep-
   full-shape output, freed incrementally) — still ~32.3GB, SIGKILLed just over the ceiling.
3. Skipping that filtered dict entirely and streaming categorical columns into one preallocated
   buffer — ~31.9–32GB+, SIGKILLed right at hnswlib's own `init_index`/`add_items` allocation
   (hnswlib's `_flatten()` + `vstack()` inside `HnswRetriever.__init__` was its own hidden
   second-and-third full-size copy).
4. Reducing hnswlib's graph parameter `M` from 32 to 16 alone (without fixing the underlying
   double-buffering) — barely moved the peak, confirming raw per-vector data size, not graph-link
   overhead, was dominant.
5. **The fix that worked**: feed hnswlib's `add_items` in small batches (~100k rows, ~100–300MB
   each), discarding each batch immediately after it's consumed, so the only persistently large
   block alive is hnswlib's own internal storage — never doubled by a second full-size source
   array — **combined with `hnsw_m=16`** (halving hnswlib's own per-element graph-link storage) for
   the final margin. Peak RSS measured directly: ~32.95GB, against a hard, non-adjustable
   33,554,432 KB (32GB) per-process ceiling — succeeded with well under 1GB to spare. Index:
   4,874,987 categorical-column embeddings indexed (`tier_1m`'s equivalent of `tier_10k`'s 48,857
   and `tier_100k`'s ~490K, i.e. consistent ~10×-per-step growth in what actually gets indexed).

**`hnsw_m=16` on `tier_1m` only (vs. `M=32` used everywhere else in this project, including
`tier_10k`/`tier_100k` above) is a memory-driven implementation necessity, not a scalability
finding** — flagged here for anyone comparing `tier_1m`'s numbers against the other two tiers'. It
can shift `tier_1m`'s realized `|D|` somewhat vs. what `M=32` would have given (a graph-construction
recall difference), though it does not change how Dinkelbach itself behaves on whatever `|D|` it
receives — the near-flat iteration counts and sub-linear-in-`|D|` timing above are properties of
Stage 1, not of the retrieval that produced `|D|`. The built index is now cached to disk
(`/u6/bkassaie/wdc_data/cache/tier_1m_hnsw_lean_m16.bin` + a small query-vectors sidecar), so a
future rerun of this exact sweep loads it in seconds instead of repeating the ~14-minute,
memory-marginal build.

---

## 3. Dinkelbach convergence at `k=10, α=3` (pool=30), all three tiers

Same 67-query cohort, same `top_n` sweep, same script (`experiments/wdc/wdc_topn_pool_sweep.py`,
`K=10`, `ALPHA=3.0`), a second grid cell rather than a rerun of §2's. `pool_size=30` is well below
the cohort's own `min(D_10k,D_100k) ≥ 200` construction guarantee, so this cell was expected to be
essentially unclamped on `tier_10k`/`tier_100k` by construction, and it is. This section's numbers
are **additional**, not a replacement for §2 — both grid cells' full per-query data are kept:
`experiments/results/wdc_topn_pool_sweep_k20_a10.csv` (§2) and
`experiments/results/wdc_topn_pool_sweep_k10_a3.csv` (this section, 1,206 rows).

`tier_1m`'s HNSW index was **loaded from the disk cache built during §2's run**
(`/u6/bkassaie/wdc_data/cache/tier_1m_hnsw_lean_m16.bin`) rather than rebuilt — confirmed by the log
(`lean semantic index built (4874987 items indexed)` followed immediately by
`cached lean semantic index to ...`, the same item count as §2's build) — though this run still paid
the full ~822s context-build cost because the caching logic itself only reached the codebase mid-way
through §2's *first* successful attempt; the process that actually succeeded there was already
running the pre-caching code, so nothing was written until this run's own cache-write completed. Any
*third* run of either grid cell will load in seconds.

| tier | top_n | mean `\|D\|` | n_ran | n_clamped (of 67) | mean stage-1 time | mean iters | iter range |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tier_10k` | 5,000 | 387.0 | 67 | 0 | 0.54 ms | 2.06 | 1–3 |
| `tier_10k` | 10,000 | 618.3 | 67 | 0 | 0.87 ms | 1.94 | 1–4 |
| `tier_10k` | 15,000 | 775.7 | 67 | 0 | 1.11 ms | 2.00 | 1–4 |
| `tier_10k` | 20,000 | 853.3 | 67 | 0 | 1.21 ms | 1.93 | 1–4 |
| `tier_10k` | 25,000 | 877.2 | 67 | 0 | 1.23 ms | 1.93 | 1–4 |
| `tier_10k` | 30,000 | 881.9 | 67 | 0 | 1.27 ms | 1.93 | 1–4 |
| `tier_100k` | 5,000 | 807.9 | 67 | 0 | 1.31 ms | 1.76 | 1–4 |
| `tier_100k` | 10,000 | 1,289.4 | 67 | 0 | 1.90 ms | 1.75 | 1–4 |
| `tier_100k` | 15,000 | 1,571.3 | 67 | 0 | 2.12 ms | 1.66 | 1–4 |
| `tier_100k` | 20,000 | 1,847.5 | 67 | 0 | 2.47 ms | 1.73 | 1–4 |
| `tier_100k` | 25,000 | 2,111.3 | 67 | 0 | 3.63 ms | 1.75 | 1–4 |
| `tier_100k` | 30,000 | 2,368.6 | 67 | 0 | 3.54 ms | 1.70 | 1–4 |
| `tier_1m` | 5,000 | 1,071.4 | 66 | 1 | 1.42 ms | 1.47 | 1–3 |
| `tier_1m` | 10,000 | 1,898.0 | 66 | 1 | 2.81 ms | 1.48 | 1–4 |
| `tier_1m` | 15,000 | 2,840.6 | 66 | 1 | 4.90 ms | 1.61 | 1–4 |
| `tier_1m` | 20,000 | 3,559.3 | 66 | 1 | 6.13 ms | 1.52 | 1–4 |
| `tier_1m` | 25,000 | 4,355.4 | 66 | 1 | 7.61 ms | 1.52 | 1–4 |
| `tier_1m` | 30,000 | 5,133.4 | 66 | 1 | 10.57 ms | 1.61 | 1–4 |

**A caveat worth stating plainly: `mean |D|` for `tier_100k` differs between this section and §2**
(e.g. 807.9 here vs. 925.5 in §2 at `top_n=5000`) **even though retrieval does not depend on `k`/`α`
at all** — `HnswRetriever.query`'s `ef` search-breadth parameter is derived from `top_n` alone, never
from the caller's `k`/`α`. The cause is hnswlib's own construction-time non-determinism: `tier_100k`
uses the ordinary (non-cached, rebuilt-per-run) `build_wdc_context` path, and `add_items` runs
multi-threaded by default — `random_seed=42` pins the algorithm's own random choices but not the
*order* concurrent threads finish inserting items, so two independently-built indexes over the exact
same data can end up with slightly different graphs and slightly different approximate-NN results.
`tier_10k` shows the same effect but far more mildly (387.0 identical, 618.3 vs. 618.4 in §2 — noise
at the rounding level, not the ~13% seen on `tier_100k`) because its far smaller graph has less room
for construction-order variance to matter. `tier_1m` cannot be compared at all in this
respect, since it reused §2's cached index outright rather than rebuilding. **None of this reflects
on Dinkelbach or on this section's own findings** — iteration counts and Stage-1 timing are computed
against whatever `|D|` each run's own retrieval actually produced, and both runs' qualitative
conclusions (near-flat iterations, near-zero clamping, sub-linear timing growth) agree.

**At `pool=30` (a much smaller pool than §2's 200), the same qualitative findings hold, more
strongly**: mean iterations are lower still (1.47–2.06 vs. §2's 1.58–2.58) — a smaller pool is a
smaller, cheaper top-`k` selection problem, so convergence is if anything faster. Mean Stage-1 time
is correspondingly lower at every comparable `(tier, top_n)` cell. Clamping is **effectively absent
everywhere**: 0/67 on `tier_10k`/`tier_100k` (as guaranteed by the cohort's construction, since
`30 ≪ 200`), and only **1/67 on `tier_1m` at every single `top_n`** (the same one query, consistently
just under 30, likely the same non-monotonicity effect from §0/§2 — not investigated further here)
— a smaller, more consistent clamp footprint than §2's tier_1m column (which ranged 1–5/67 depending
on `top_n`), consistent with a smaller pool being an easier bar to clear.

---

## 4. Stage 2 execution time: LP pre-check vs. MILP solver, all three tiers

> **Superseded by §5 for Stage 2 timing:** this section used seeded *synthetic* `U`; §5 reruns the same grid
> with real unionability scoring. Kept for the LP-decisive analysis and the with/without-precheck measurement.

WDC analogue of `RESULTS-santoslarge.md` §5c. `top_n=5000`, grid `α∈{10,5}×k∈{5,10,20}` (pools
25–200), `F*=0.20, δ=0.10` (`τ=0.10`), the same 67-query cohort. Times come from
`stage2_ilp.solve`'s own `lp_time_s` (the `linprog` pre-check) and `milp_time_s` (only the `milp(...)`
call), plus an outer wall-clock timer around the whole call. Script:
`experiments/wdc/wdc_stage2_timing.py`; data: `experiments/results/wdc_stage2_timing.csv` (1,206 rows).

| tier | α | k | pool | clamped | feasible | LP-decisive | mean LP precheck | mean MILP solver | **precheck + solver** | full Stage 2 wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tier_10k` | 10 | 5 | 50 | 0 | 67/67 | 0 | 1.071 ms | 1.583 ms | **2.654 ms** | 2.905 ms |
| `tier_10k` | 10 | 10 | 100 | 0 | 67/67 | 0 | 0.994 ms | 1.697 ms | **2.691 ms** | 2.938 ms |
| `tier_10k` | 10 | 20 | 200 | 0 | 67/67 | 0 | 1.202 ms | 2.162 ms | **3.364 ms** | 3.651 ms |
| `tier_10k` | 5 | 5 | 25 | 0 | 67/67 | 0 | 0.873 ms | 1.426 ms | **2.299 ms** | 2.530 ms |
| `tier_10k` | 5 | 10 | 50 | 0 | 67/67 | 0 | 0.891 ms | 1.467 ms | **2.359 ms** | 2.593 ms |
| `tier_10k` | 5 | 20 | 100 | 0 | 67/67 | 0 | 0.980 ms | 1.677 ms | **2.658 ms** | 2.905 ms |
| `tier_100k` | 10 | 5 | 50 | 0 | 67/67 | 0 | 1.183 ms | 2.148 ms | **3.331 ms** | 3.625 ms |
| `tier_100k` | 10 | 10 | 100 | 0 | 67/67 | 0 | 1.181 ms | 2.887 ms | **4.068 ms** | 4.364 ms |
| `tier_100k` | 10 | 20 | 200 | 0 | 67/67 | 0 | 1.376 ms | 4.341 ms | **5.717 ms** | 6.048 ms |
| `tier_100k` | 5 | 5 | 25 | 0 | 67/67 | 0 | 0.992 ms | 2.322 ms | **3.314 ms** | 3.583 ms |
| `tier_100k` | 5 | 10 | 50 | 0 | 67/67 | 0 | 1.019 ms | 2.122 ms | **3.141 ms** | 3.421 ms |
| `tier_100k` | 5 | 20 | 100 | 0 | 67/67 | 0 | 1.135 ms | 3.221 ms | **4.357 ms** | 4.648 ms |
| `tier_1m` | 10 | 5 | 50 | 1 | 60/67 | 7 | 1.682 ms | 2.153 ms | **3.835 ms** | 3.912 ms |
| `tier_1m` | 10 | 10 | 100 | 1 | 60/67 | 7 | 1.201 ms | 2.210 ms | **3.411 ms** | 3.449 ms |
| `tier_1m` | 10 | 20 | 200 | 4 | 60/67 | 7 | 1.354 ms | 2.705 ms | **4.059 ms** | 4.073 ms |
| `tier_1m` | 5 | 5 | 25 | 1 | 60/67 | 7 | 0.976 ms | 1.706 ms | **2.682 ms** | 2.746 ms |
| `tier_1m` | 5 | 10 | 50 | 1 | 60/67 | 7 | 0.990 ms | 1.812 ms | **2.802 ms** | 2.860 ms |
| `tier_1m` | 5 | 20 | 100 | 1 | 60/67 | 7 | 1.086 ms | 1.962 ms | **3.048 ms** | 3.098 ms |

LP precheck is averaged over all 67 rows; the MILP solver over only the rows where the LP was
feasible (the MILP is skipped when the LP proves infeasibility). "Full Stage 2 wall" averages all
rows, so on `tier_1m` (where 7 rows skip the MILP) it sits closer to the precheck+solver sum.

**Findings.**
- **Stage 2 costs a few milliseconds everywhere** (2.3–6.0 ms full wall), and grows only mildly with
  pool size, the same regime as santosLarge (§5c: 1.8–2.5 ms at pools 20–60). Going from pool 25 to pool 200 (8× larger) costs
  ~1.5–1.7× as much (e.g. `tier_10k` 2.30 → 3.36 ms; `tier_100k` 3.31 → 5.72 ms), not 8×.
- **The MILP solver is the larger share** (~55–76% of precheck+solver; `tier_100k` at pool 200: 4.3 ms
  solver vs. 1.4 ms precheck), the LP precheck the remainder, matching santosLarge's ~60–65/35–40 split.
  The precheck is nearly flat in pool size (0.87–1.68 ms); the solver is what grows.
- **Corpus scale barely matters for Stage 2 itself**: the same `(α,k)` cell costs 2.7 → 3.3 → 3.8 ms
  (α=10,k=5) across `tier_10k → tier_100k → tier_1m`, i.e. ~1.4× for 100× more tables, because Stage 2
  only ever sees the pool `α×k`, never `|T|` (the paper's efficiency claim). `tier_100k` is the slowest
  at large pools rather than `tier_1m` — not explained by anything measured here; MILP time depends on
  the pool's `(N,n,U)` structure, not just its size, and the pools differ by tier (see caveats).
- **New on `tier_1m`: the LP pre-check becomes decisive.** On `tier_10k`/`tier_100k` all 67 queries are
  feasible at every cell (the cohort's `F*/δ/τ` was tuned there, §0), so the pre-check never fires. On
  `tier_1m`, **7 of 67 queries are LP-infeasible at every cell** (60/67 feasible, identical across all six
  cells): for those queries the pool retrieved on `tier_1m` cannot reach `τ=0.10`, the LP proves it, and
  the MILP is skipped. That is the first place in this project's WDC study where the pre-check
  does real work, and it is a scale effect: the cohort was validated only against the two smaller tiers.
  Clamping is also present but small (1/67 at most cells, 4/67 at pool 200).

**Caveats.**
1. **`U` is seeded synthetic uniform(0,1), not real unionability.** `tier_1m`'s lean context holds no
   full vectors dict (32GB RSS ceiling, §2), and real matching is not what is being timed. Stage 2
   feasibility depends only on `(N,n)`; `U` shapes only the MILP objective, so MILP times could differ
   somewhat with real scores, LP-decisive counts would not.
2. **Each tier's retrieval was rebuilt/re-run for this experiment**, so `|D|` (and thus pools) is not
   bit-identical to §2/§3 for `tier_10k`/`tier_100k` (hnswlib's multi-threaded build is not fully
   deterministic across rebuilds, §3). `tier_1m` used the cached `M=16` index (same as §2/§3; `M=32`
   elsewhere), so its retrieval matches §2/§3's `tier_1m` runs.
3. One solve per `(query, cell)`, 67 samples per cell. At millisecond scale, absolute values carry
   process-level noise (as santosLarge §5c also cautions); the relative split (solver > precheck, mild
   pool-size dependence, scale-insensitivity) is the stable finding.

---

## 5. Per-stage and end-to-end execution time, real unionability, all three tiers

`top_n=5000`, grid `α∈{10,5}×k∈{5,10,20}` (pools 25–200), `F*=0.20, δ=0.10` (`τ=0.10`), the 67-query
cohort. Stages, composed exactly as `dutsx.runner.run_query` does: **retrieval** (HNSW probe + overlap
filter + the `N_i/n_i` lookup for every candidate in `D`) → **Stage 1** (Dinkelbach, or the C5 clamp) →
**unionability scoring of `P` only** (real §6.2 pinned-match, `σ=0.6`, exactly `|P|=α×k` matchings) →
**Stage 2** (LP pre-check + MILP). Retrieval does not depend on `k`/`α`, so it is timed once per query and
the same value appears in every cell of a tier. End-to-end = retrieval + Stage 1 + scoring + Stage 2.
Script: `experiments/wdc/wdc_end2end_timing.py`; per-row data: `experiments/results/wdc_end2end_timing.csv`
(1,206 rows). All times are means in **ms**.

| tier | α | k | pool | clamped | feasible | retrieval | Stage 1 | unionability scoring | LP precheck | MILP | Stage 2 (full) | **end-to-end** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tier_10k` | 10 | 5 | 50 | 0 | 67/67 | 15.9 | 0.62 | 12.83 | 1.57 | 2.23 | 4.05 | **33.4** |
| `tier_10k` | 10 | 10 | 100 | 0 | 67/67 | 15.9 | 0.89 | 24.96 | 1.27 | 2.22 | 3.73 | **45.4** |
| `tier_10k` | 10 | 20 | 200 | 0 | 67/67 | 15.9 | 0.98 | 45.94 | 1.53 | 2.81 | 4.62 | **67.4** |
| `tier_10k` | 5 | 5 | 25 | 0 | 67/67 | 15.9 | 0.93 | 8.28 | 1.02 | 1.85 | 3.08 | **28.2** |
| `tier_10k` | 5 | 10 | 50 | 0 | 67/67 | 15.9 | 0.79 | 13.85 | 1.05 | 1.98 | 3.25 | **33.7** |
| `tier_10k` | 5 | 20 | 100 | 0 | 67/67 | 15.9 | 0.88 | 25.06 | 1.19 | 2.21 | 3.64 | **45.4** |
| `tier_100k` | 10 | 5 | 50 | 0 | 67/67 | 33.0 | 1.19 | 12.55 | 1.57 | 2.26 | 4.09 | **50.8** |
| `tier_100k` | 10 | 10 | 100 | 0 | 67/67 | 33.0 | 1.76 | 26.69 | 1.28 | 2.43 | 3.96 | **65.4** |
| `tier_100k` | 10 | 20 | 200 | 0 | 67/67 | 33.0 | 1.85 | 49.08 | 1.44 | 2.96 | 4.67 | **88.6** |
| `tier_100k` | 5 | 5 | 25 | 0 | 67/67 | 33.0 | 1.75 | 7.66 | 0.99 | 1.85 | 3.05 | **45.5** |
| `tier_100k` | 5 | 10 | 50 | 0 | 67/67 | 33.0 | 1.61 | 13.14 | 0.98 | 1.88 | 3.08 | **50.8** |
| `tier_100k` | 5 | 20 | 100 | 0 | 67/67 | 33.0 | 1.70 | 25.35 | 1.18 | 2.28 | 3.70 | **63.8** |
| `tier_1m` | 10 | 5 | 50 | 1 | 60/67 | 351.3 | 1.46 | 11.71 | 2.14 | 3.42 | 5.48 | **370.0** |
| `tier_1m` | 10 | 10 | 100 | 1 | 60/67 | 351.3 | 2.07 | 24.52 | 1.27 | 2.37 | 3.62 | **381.5** |
| `tier_1m` | 10 | 20 | 200 | 4 | 60/67 | 351.3 | 2.07 | 46.01 | 1.41 | 5.40 | 6.49 | **405.9** |
| `tier_1m` | 5 | 5 | 25 | 1 | 60/67 | 351.3 | 1.99 | 7.03 | 1.01 | 1.86 | 2.86 | **363.2** |
| `tier_1m` | 5 | 10 | 50 | 1 | 60/67 | 351.3 | 1.93 | 12.68 | 1.01 | 1.94 | 2.95 | **368.9** |
| `tier_1m` | 5 | 20 | 100 | 1 | 60/67 | 351.3 | 1.95 | 24.12 | 1.15 | 2.58 | 3.66 | **381.1** |

MILP is averaged over rows where the LP was feasible (the MILP is skipped otherwise); every other
column is averaged over all 67 rows.

**Findings.**
- **Retrieval is the only stage that grows with corpus size**: 15.9 → 33.0 → 351.3 ms per query across
  `tier_10k → tier_100k → tier_1m` (2.1× then 10.6×). On `tier_1m` it is 86–97% of end-to-end time (less at larger pools); on the two
  smaller tiers it is 24–65%, and unionability scoring overtakes it at large pools (`tier_10k`, α=10,
  k=20: 45.9 ms scoring vs. 15.9 ms retrieval).
- **Unionability scoring is linear in the pool and independent of the corpus**: ≈0.23–0.26 ms per pool table
  on every tier (e.g. 200 tables: 45.9 / 49.1 / 46.0 ms). This is the paper's efficiency claim measured
  directly: scoring cost is `α×k`, never `|T|`.
- **Stage 1 (0.6–2.1 ms) and Stage 2 (2.9–6.5 ms) are small and nearly scale-free.** Together they are
  under 10 ms in every cell, versus 28–406 ms end-to-end.
- **End-to-end**: 28–89 ms on `tier_10k`/`tier_100k`, 363–406 ms on `tier_1m`. Going from 10k to 1M tables
  (100×) costs ~6–13× end-to-end depending on the cell (larger pools dilute it), and almost all of that
  is retrieval.
- **Feasibility and clamping are unchanged from §4** (feasibility does not depend on `U`): all 67 feasible on
  the two smaller tiers; on `tier_1m`, 60/67 feasible at every cell (the same 7 LP-infeasible queries) and
  1–4 clamped.
- **Real `U` makes Stage 2 somewhat slower than §4's synthetic `U`** (e.g. `tier_10k`, α=10, k=5: MILP
  2.23 ms here vs. 1.58 ms in §4). Real scores give the MILP a less degenerate objective; I did not
  isolate this from run-to-run machine load, so treat the size of the gap as indicative only.

**Caveats.**
1. `tier_1m` ran with the cached `M=16` HNSW index (`M=32` elsewhere; memory-driven, §2). Retrieval was rebuilt
   for `tier_10k`/`tier_100k`, so their `|D|` is not bit-identical to §2/§3 (multi-threaded build, §3).
2. `tier_1m` needs the full 16GB vectors file for real scoring alongside the 5GB index and the overlap
   index: peak RSS was ~33.3M KB (≈31.8 GiB) against the 32 GiB per-process cap, so it fit with under 1 GiB
   to spare. It is not a comfortable margin.
3. One run, 67 samples per cell; millisecond-scale values carry process-level noise (as in santosLarge §5c),
   so the relative pattern (retrieval grows with corpus, scoring grows with pool, Stage 1/2 flat) is the
   stable finding. End-to-end is the sum of per-stage means, not a single timed loop.

---

*(Further sections — outcome by `(k,α)`, per-stage timing, LP pre-check, skip-Stage-1 comparison —
pending the full end-to-end study run on the 67-query shared cohort.)*

### 5.1 Where retrieval time goes (why `tier_1m` is slow)

Script `experiments/wdc/retrieval_breakdown.py` (sbatch), same 67-query cohort, `top_n=5000`; per-query means,
CSVs `results/wdc_retrieval_breakdown_<tier>.csv`. Each probe is also repeated immediately to separate cold-cache cost.

| tier | HNSW probe ms | overlap probe ms | join ms | N_i/n_i lookup ms | total ms | \|D_sem\| | \|D_ovl\| | \|D_pair\| | \|D\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tier_10k | 8.1 | 1.2 | 1.0 | 1.2 | 11.4 | 4,974 | 3,289 | 574 | 388 |
| tier_100k | 9.3 | 15.5 | 2.1 | 2.9 | 29.9 | 4,934 | 34,038 | 1,201 | 888 |
| tier_1m | 10.2 | 203.5 | 24.4 | 3.6 | 241.7 | 4,599 | 341,014 | 1,328 | 1,071 |

**Findings.** The HNSW probe is nearly flat (8.1→9.3→10.2 ms). The overlap probe (posting-list lookup of the protected
value) returns every (table, column) pair containing it, which grows ~10× per tier (3.3k→34k→341k); its time grows
~13× per tier and is 84% of `tier_1m` retrieval. The join, a dict plus set intersection over those pairs, grows the same
way (24.4 ms at 1M). Only ~1,000 pairs survive the intersection. Repeating each probe gives almost the same time
(HNSW 8.6 ms, overlap 196 ms at 1M), so cold-cache or paging cost is not the explanation.
This refutes the earlier guess that memory pressure on the M=16 index drives the jump.

**Caveat.** This run had no 32 GiB cap (peak RSS 46 GiB), so its `tier_1m` total (241.7 ms) is lower than §5's 351.3 ms;
the two are different machine conditions and only the split between components is the finding.
