# DUTS — End-to-End Integration Plan

Scope: everything between **real tables on disk** and the **verified core** in
`duts/` — candidate retrieval (§4.2/§7), unionability scoring (§6.2), and an
experiment harness. `PLAN.md`'s core is finished and stays **untouched**.

Design constraint from the outset: **every component that touches real data is
swappable.** JOSIE was the original motivating case for this; `Fair_Table_Search_7.pdf`
has since dropped JOSIE from the paper entirely (§4, I1 update), so the concrete
motivating case is gone, but the design principle it justified stands on its
own — it applies just as much to the ANN index, the synopsis source, and the
unionability scorer. Named interfaces (`ports.py`), one adapter per
implementation, config-driven selection — not `if josie: ... else: ...`
scattered through a driver.

---

## 1. Why a second package, not more of `duts/`

`duts/` is a pure function of `List[CandidateStats]` + `(N_Q, n_Q)`. That's what
made it brute-force verifiable (1761 tests). Anything that reads a CSV, loads a
`.pkl`, or calls hnswlib breaks that property. So:

```
duts/     # core — pure, exact, tested. NOT modified by this plan.
dutsx/    # everything that touches real data. Depends on duts/, never the reverse.
experiments/  # sweeps, metrics, result tables. Depends on dutsx/.
```

The dependency arrow points one way. A bug in an adapter can never make Stage 1
or Stage 2 wrong; it can only feed them wrong numbers, which is a different and
much easier failure to localize.

---

## 2. The seams (`dutsx/ports.py`)

Four `typing.Protocol`s (verified available on py3.8.5). These are the entire
swap surface — replacing an `OverlapFilter` implementation (Roaring bitmaps for
the posting lists, per `Fair_Table_Search_7.pdf` §7.2's recommendation, being
the live case now that JOSIE is off the table) means writing one class that
satisfies it, registering it, and changing one config string.

```python
class SynopsisSource(Protocol):
    """Per-table row counts and per-attribute {value: count} histograms."""
    def n_rows(self, table: str) -> int: ...
    def distribution(self, table: str, attr: AttrRef) -> Dict[str, int]: ...
    def categorical_attrs(self, table: str) -> List[AttrRef]: ...

class SemanticRetriever(Protocol):
    """§7.1 — ANN over attribute embeddings."""
    def query(self, vec: np.ndarray, top_n: int) -> List[Tuple[str, int, float]]:
        """-> [(table, attr_idx, similarity)], already threshold-filtered."""

class OverlapFilter(Protocol):
    """§7.2 (v7: "Value-overlap filtering via posting lists") — |M ∩ V_c| >= 1,
    a Boolean membership test with no ranking, evaluated exactly via an
    inverted index. The size returned is diagnostic only -- never a score
    (§7.3's tie-break uses sim alone)."""
    def query(self, M: Set[str]) -> Dict[Tuple[str, int], int]: ...

class UnionabilityScorer(Protocol):
    """§6.2 — U_{V_D}(Q, T_i) for ONE candidate, with the V_D pair pinned."""
    def score(self, q_table: str, c_table: str, pin: Tuple[int, int]) -> float: ...
```

**Adapters to build (per port, config-selectable by name):**

| Port | Adapter | Notes |
|---|---|---|
| `SynopsisSource` | `MetadataStoreSynopsis` | wraps `starmie_fair`'s `MetadataStore`; `get_num_records` → `n_i`, `get_distribution` → integer histogram → exact `N_i` (C2). **Verified working**: loads 550 santos tables, values are `int`. |
| | `CsvSynopsis` | recomputes from CSVs; the staleness oracle for the above |
| `SemanticRetriever` | `HnswRetriever` | §7.1, over existing Starmie column vectors (I2 — confirmed, already computed) |
| | `ExactScanRetriever` | brute-force cosine over all 6322 santos columns — the **recall oracle**, not a baseline to ship |
| `OverlapFilter` | `InvertedIndexOverlap` | **the paper's own design (§7.2 v7), not a fallback.** Exact `value → {(table, attr)}`, probe `⋃_{c∈M}`. `Fair_Table_Search_7.pdf` rewrote §7.2 to specify posting lists directly, dropped JOSIE from every citation, and retargeted `[13]` to Zobel & Moffat 2006 — confirmed conformant point-by-point, unmodified. |
| | ~~`JosieOverlap`~~ | **will not be built** — JOSIE is not part of the paper as of v7. See PLAN-full.md I1. |
| | `NullOverlap` | returns everything — for ablating the overlap filter's contribution |
| `UnionabilityScorer` | `PinnedMatchScorer` | §6.2 + M3: pin `(V_D, V_{D_T})`, delete that row/col, Hungarian on the rest, return `s(pin) + Σ matched` |
| | `StarmieVerifyScorer` | wraps `bounds.py:verify_constrained` — comparability check, not the spec (M3: it pins the row only, letting Hungarian choose the column) |
| | `ConstantScorer` | test double; makes retrieval effects observable without alignment noise |

---

## 3. The one architectural change the core's API forces

**`duts.pipeline.run(D, query)` takes `D` with `U` already populated for every
candidate. That defeats §4.1's central efficiency claim.** Stage 1 and the C4
certificate read only `(N_i, n_i)`; `U` is used *exclusively* by Stage 2, which
runs over `P` only. Scoring `U` for all of `D` would do `|D|` bipartite
matchings where the paper promises `α×k`.

So `dutsx/runner.py` **composes the stages directly rather than calling
`pipeline.run`**:

```
retrieve  → D with U unset          (|D| candidates, no alignment yet)
[certify] → optional (C4; currently default-off)
Stage 1   → P                        (uses N_i, n_i only)
score U   → for the α×k tables in P ONLY   ← the efficiency claim, made measurable
Stage 2   → R
```

This keeps `duts/` unmodified and makes **#unionability computations** a
directly countable telemetry field — the metric §4.1 actually argues about, and
the one PLAN-full.md S5 requires for a fair §4-vs-§8 comparison. `pipeline.run`
stays available and is still what the core's own tests exercise; it's the right
entry point when `U` is already known (fixtures, synthetic data).

---

## 4. Phases

**Phase A — ports + synopsis adapter. DONE.** `dutsx/ports.py` (4 Protocols),
`dutsx/adapters/synopsis.py` (`MetadataStoreSynopsis`, `CsvSynopsis`),
`dutsx/registry.py` (config-driven `build`/`build_from_config`),
`tests/test_dutsx_synopsis.py` (19 tests against real santos data).
*Accept, with actual numbers:*
- santos: 550 datalake tables. Fresh-from-CSV at the **configured**
  `θ_cat = 50`: **3583 attrs survive, 52 tables have zero categorical attrs**
  (`CsvSynopsis`, independent of `MetadataStore`).
- **Staleness check: no staleness found.** All 550 datalake `n_i` and all 50
  query `n_i` in the `.pkl`s match a fresh CSV row count exactly (0
  mismatches); a 40-table sample of full `{value: count}` histograms matched
  byte-for-byte too. The Dec-2025-pickle-vs-2021-CSV date gap turned out not
  to indicate drift.
- ⚠️ **New finding, not anticipated when this phase was scoped: the pickle's
  own baked-in "categorical" filter is NOT `θ_cat = 50`.** Every one of the
  2220 distributions actually stored in `metadata_datalake.pkl` has domain
  size ≤ 10 (checked exhaustively); columns with true domain size in
  `(10, 50]` genuinely exist in the CSVs (e.g. `issue_type`, domain 33) and
  are silently absent from the pickle — not filtered by a downstream reader,
  never stored. So `MetadataStoreSynopsis.categorical_attrs` (2220 attrs, 96
  zero-attr tables) and `CsvSynopsis(theta_cat=50).categorical_attrs` (3583
  attrs, 52 zero-attr tables) **disagree, mostly by design** — that's the
  staleness oracle doing its job, though 125 of the original gap (2345→2220,
  95→96, corrected 2026-08-13, NOTES.md) was a genuine `MetadataStoreSynopsis`
  bug (domain size wrongly counted the `""` empty-cell bucket), not the
  pickle's pre-filtering. Root cause of the *remaining* gap not fully
  recovered: the pickle's
  `_global_stats` dict is keyed `total_categorical_columns`, but the
  currently-checked-in `TableMetadata.py` pickles `_global_stats` keyed
  `total_columns_with_distributions` and computes distributions for *all*
  columns unconditionally — proof the pickle was built by a since-edited
  version of the class that filtered by domain size, but no script in the
  repo currently reproduces that build. **Any later phase that needs
  `θ_cat = 50` specifically (not "whatever the pickle happened to keep")
  must use `CsvSynopsis`, not `MetadataStoreSynopsis`, or must re-derive
  `categorical_attrs` from `distribution()` + its own threshold.**
- ⚠️ **Resolved:** `MetadataStore`'s docstring claims distributions for *all*
  columns, but `biodiversity_2.csv` came back with none. Its 3 columns
  (`scientific name`, `family name`, `common name`) have domain sizes
  257/91/257 over 257 rows — every one exceeds *both* the configured
  `θ_cat = 50` and the pickle's actual ~10 cutoff. Not a bug in either
  version of the code; the table genuinely has zero categorical columns
  under any threshold in play. Full writeup with evidence:
  `dutsx/adapters/synopsis.py` module docstring.

**Phase B — retrieval. DONE.** `dutsx/adapters/semantic.py` (`HnswRetriever`,
`ExactScanRetriever`), `dutsx/adapters/overlap.py` (`InvertedIndexOverlap`,
`NullOverlap`), `dutsx/retrieval.py` (§7.3: `D_pair = D_sem ∩ D_ovl` at pair
level, then per-table `argmax sim`), registered as `hnsw`/`exact_scan` and
`inverted_index`/`null`. `tests/test_dutsx_retrieval.py`, 31 tests against
real santos data. Full suite: **1889 passed, 19 skipped**, lint clean,
`duts/` untouched.

*Correction to this plan's own wording, found while implementing:* the
original text above said "build hnswlib with `space='cosine'`" (from
PLAN-full.md I3) but described the accept criterion in Euclidean-distance
terms (`θ_dis = sqrt(2−2σ)`). Those aren't the same code path — `space='l2'`
over normalized vectors and `space='cosine'` over raw vectors are two
different indexes. Resolved (coordinator clarification, 2026-08-09):
`HnswRetriever` **ships `space='cosine'` over the raw, unnormalized
vectors** — hnswlib's cosine space normalizes internally and returns
`1 − cos_sim` directly, so no explicit L2-normalization or `θ_dis`
conversion is needed in the adapter itself, matching starmie_fair's own
usage pattern (`HNSWSearcher_Fair.py:152-156`, cited for the four-call API
shape only — `dutsx/` does not import or wrap that module). The
`θ_dis = sqrt(2−2σ)` **identity itself is still verified**, independently of
what ships: `test_l2_theta_dis_equivalent_to_cosine_sigma_exact_brute_force`
checks the algebraic identity `‖a−b‖² = 2 − 2·cos(a,b)` exactly (max abs
error `2.24e-07`, 0 mismatched threshold sets over 156 real query columns),
and `test_l2_vs_cosine_hnsw_indexes_agree_closely` routes the same check
through two independently-built real hnswlib indexes (`space='l2'` over
explicitly-normalized vectors vs `space='cosine'` over raw vectors), which
introduces the ANN's own approximation on top of the math (mean Jaccard
`0.9953`, min `0.9608` over 274 query columns) — small and expected, not a
sign the identity is wrong.

*Accept, with actual numbers:*
- **HNSW recall vs `ExactScanRetriever`** on santos (550 datalake tables,
  6322 columns — confirmed exactly, small enough to brute force): over every
  santos query column with a non-empty exact-scan result at `top_n=100,
  σ=0.6` (615 of them), **mean recall 0.998, min recall 0.95** (numbers vary
  slightly run to run with hnswlib's construction RNG; the test asserts
  conservative floors of `≥0.95` mean / `≥0.85` min, not the exact observed
  values).
- **`θ_dis = sqrt(2−2σ)` / cosine `>σ` equivalence** — verified as described
  above, both exactly (brute force) and through real HNSW search.
- **Pair-level intersection, not table-level.** A table entering `D_sem` via
  attribute `A1` and `D_ovl` via a different attribute `A2` yields **no**
  surviving pair, checked directly with a constructed witness
  (`test_table_entering_via_different_attributes_yields_no_pair`) — the
  converse (same attribute in both) survives, and a table with two
  attributes clearing both filters keeps only the higher-`sim` one
  (`test_table_with_two_qualifying_attributes_keeps_only_the_higher_sim_one`).
- **Telemetry.** `RetrievalTelemetry(n_sem, n_ovl, n_pair, n_d)` is returned
  alongside the candidate list by `dutsx/retrieval.py::retrieve_candidates`
  on every call; demonstrated against real adapters both for an arbitrary
  probe (frequently `n_pair=0` — a correct, uninteresting answer when the
  probe and `M` don't happen to align) and for a self-referencing probe
  constructed to guarantee `n_pair≥1` (`|D_sem|=100 |D_ovl|=137 |D_pair|=1
  |D|=1`), so both the "empty" and "non-empty" paths are exercised with real
  data, not just synthetic stubs.
- `InvertedIndexOverlap` checked against an independent brute-force
  recompute over the same table sample; `NullOverlap` checked to return
  every indexed pair regardless of `M`, including `M = ∅`.

**Phase C — unionability. DONE.** `dutsx/adapters/unionability.py`:
`PinnedMatchScorer` (the §6.2/M3 spec), `StarmieVerifyScorer` (wraps
`bounds.py:verify_constrained`, for comparability only), `ConstantScorer`
(test double). Registered as `pinned_match` / `starmie_verify` / `constant`.
`tests/test_dutsx_unionability.py`, 78 tests.
*Accept, with actual numbers:*
- **The M3 disagreement witness is real**, and designed rather than
  searched for. Similarity matrix `[[0.70, 0.95], [0.65, 0.23]]`, pin
  `(q₀, c₀)`: §6.2-as-specified returns **0.70**, `verify_constrained`
  returns **1.60**. The gap is exactly M3 — `verify_constrained` pins the
  *row* via its big-bonus trick, so Hungarian re-routes `V_D` to the
  higher-similarity column `c₁` (0.95) and picks up `q₁→c₀` (0.65), which
  is precisely the re-optimization §6.2 forbids ("fixed and excluded from
  the optimization"). Kept as a regression fixture; if it ever starts
  reporting agreement, the pin has been misimplemented.
- Converse verified: the two **agree** when the pinned column is the one an
  unconstrained matcher would have chosen anyway.
- Property over 60 random instances: `pinned ≤ starmie` always, since
  forcing `q_i → t_j` optimizes over a strict subset of the matchings
  `verify_constrained` considers. A violation would mean the pinned
  implementation is double-counting or reusing the pinned column.
- Returns `0.0` when `s(pin) ≤ σ`; `U` is an unnormalized raw sum and is
  tested to exceed `1.0` (M4: Table 1's `[0,1]` is the paper's error, not
  §6.2's formula).
- Kept `bounds.py`'s strict `sim > σ` rather than §6.2's stated `≥`, for
  comparability with Starmie's published numbers — noted in code, not
  silently changed.

*Implementation note:* the matching uses `scipy.optimize.linear_sum_assignment`
rather than `munkres`. Both solve max-weight assignment exactly; scipy is
already a hard dependency and avoids `munkres`'s `DISALLOWED` sentinel
handling. Deleting the pinned row **and** column also makes this strictly
cheaper than the big-bonus trick — one smaller assignment problem.

**Phase D — end-to-end runner. DONE.** `dutsx/runner.py`: `QueryTask` (query
table, `V_D` attr, `M`, `F*`, `δ`, `k`, `α`, `include_query`, `certify`,
`top_n`) → `RunResult` (`selected`=R, `pool`=P scored, `candidates`=D
unscored, `RunTelemetry` incl. `n_unionability_computations`); plus
`load_queries_from_csv` and `retrieve_unscored_candidates`.
`tests/test_dutsx_runner.py`, 18 tests (stub-adapter unit tests + two
real-santos-data tests). Full suite: **1907 passed, 19 skipped** (1889 + 18
new), lint clean, `duts/` untouched, `duts.pipeline.run` not called.

**Composition, not `pipeline.run`, confirmed as planned** (§3): `run_query`
calls `duts.stage1_dinkelbach.solve`/`certify_feasible` directly on
`List[UnscoredCandidate]` — a NamedTuple with **no `U` field at all** — then
scores only Stage 1's output before calling `duts.stage2_ilp.solve`. This
makes the efficiency guarantee a structural fact, not a comment: Stage 2
reads `.U`, so it cannot silently run against `D` (`AttributeError`, not a
wrong answer); `tests/test_dutsx_runner.py::test_unscored_candidate_has_no_U_field`
pins the type-level half of the guarantee, and
`test_scoring_calls_are_exactly_the_pool_tables` pins the runtime half by
inspecting the scorer's own recorded call list.

*Accept, with actual numbers (k=3, α=2, F*=0.3, δ=0.15, top_n=100,
`certify=False`):*
- **Correction to this section's original wording: not all 50 santos
  queries have a usable row.** `protected_attributes_santos.csv` has 96
  rows, but only **48** have a `q_name` that is an actual file in
  `santos/query/` — the other 48 all end in `_fair.csv` and belong to a
  DIFFERENT benchmark directory (`santos3/query/`), never mounted on this
  path. Conversely 2 of the 50 real query files (`albums_b.csv`,
  `film_locations_in_san_francisco_a.csv`) have no row in the CSV at all.
  `load_queries_from_csv` returns `(tasks, skipped)` and skips
  non-resolving rows rather than erroring — confirmed exactly 48 tasks from
  96 rows, both ways (see its docstring and NOTES.md "Phase D").
- **48-query run:** causes = `{feasible: 18, stage2_infeasible: 17,
  insufficient_candidates: 13}`; `Δ(F*,F_R) ≤ δ` held on all 18 feasible
  results (checked per-query, not just in aggregate);
  `n_unionability_computations == len(P) == len(pool_tables)` held on
  every one of the 48 queries, not just on average.
- **`n_unionability_computations` independence from `|T|`, shown against
  real data**: same query (a self-referencing probe against
  `analytics-iris-externe-...csv`, `k=2, α=3` ⇒ pool size 6) run against a
  550-table full datalake vs. a 38-table shrunk one (the needed candidates
  plus 30 filler tables): `n_unionability_computations` = **6 in both
  cases**, `|D|` differing slightly (8 vs 9 — approximate-HNSW noise on a
  much smaller index, not the claim under test).
- Two supporting stub-adapter tests isolate the same claim from all
  retrieval mechanics: growing a synthetic `|D|` from 20→200→2000 at fixed
  `k=5, α=2` holds scoring calls at exactly 10 throughout.
- All four infeasibility causes reproduced at this layer directly:
  `insufficient_candidates` (0 scoring calls — short-circuits pre-Stage-1),
  `stage2_infeasible` (`certify=False`; scoring DID happen, `alpha*k` calls,
  before Stage 2 discovers no feasible k-subset), `intrinsic`
  (`certify=True`; certificate fails before Stage 1 — 0 scoring calls, the
  more specific cause `stage2_infeasible` never even gets a chance to
  apply), and `alpha_induced` (`certify=True`; the exact numeric witness
  from `tests/test_pipeline.py::test_infeasibility_cause_alpha_induced`,
  replayed through the runner's own composition — certify passes but
  Stage 1's pool doesn't contain a clearing k-subset, `alpha*k` scoring
  calls did happen since the certificate passed).

**Phase E — experiments. DONE.** `experiments/` package: `schema.py` (tidy
`ResultRow`, 42 columns, one row per `(query, config)`), `context.py`
(config-driven adapter/context building via `dutsx.registry`, shared-resource
loading to avoid redundant CSV/pickle reads across adapter combinations),
`alpha_sweep.py`, `lp_precheck.py`, `retrieval_ablation.py`, `analysis.py`
(pure aggregation functions), `cli.py` (`python -m experiments.cli
{alpha-sweep,lp-precheck,retrieval-ablation,all}`, seeded, `--limit-queries`
for smoke runs). `tests/test_experiments.py`, 18 tests (16 stub/synthetic +
2 tiny real-data smoke tests). Full suite: **1925 passed, 19 skipped**, lint
clean, `duts/` untouched. Real sweeps run and committed under
`experiments/results/` (`alpha_sweep.csv` 720 rows, `lp_precheck.csv` 230
rows, `lp_precheck_synthetic.csv` 240 rows, `retrieval_ablation.csv` 192
rows, `.parquet` twins of each).

*Accept, with actual numbers (all 48 usable santos queries, `F*=0.3,
δ=0.15, top_n=100, certify=True`, `pinned_match`/`hnsw`/`inverted_index`):*

- **α sweep `{1,2,3,5,10} × k∈{5,10,20}`, 720 rows, 0 efficiency-invariant
  violations.** Infeasibility split by cause is **identical across every α
  at a given k** — `k=5`: 15 feasible / ~19–20 `insufficient_candidates` /
  ~13–14 `intrinsic` (±1, see HNSW-determinism note below); `k=10`: 8 / 37 /
  3; `k=20`: 0 / 48 / 0 (every query's retrieved pool is smaller than 20 —
  matches the task brief's explicit prediction). **`alpha_induced` and
  `stage2_infeasible` never occur, at any α from 1 to 10, even retried at
  much tighter `τ` (`F*=0.6/δ=0.05`, `F*=0.9/δ=0.02`) — checked directly,
  not assumed.** Root cause: santos' retrieval funnel (`InvertedIndexOverlap`
  probing `M`, then HNSW/exact-scan on top) returns very small pools (mean
  `|D|` 6–10 tables, Phase B/D numbers) — well under `α×k` for `α≥2` at every
  tested `k`, so Stage 1's pool is already clamped to all of `D` for most of
  the α range, and whenever *any* k-subset of `D` clears `τ` (the C4
  certificate), Stage 1's F-maximizing selection at every α happens to find
  one too. **This is the honest answer to M6's open question on this
  dataset**: the certificate still correctly separates `insufficient_candidates`
  from `intrinsic` (a real, load-bearing distinction — `intrinsic` is 0/13/3
  across the three `k`s, not negligible), but the *additional* `alpha_induced`
  case it exists to diagnose — proven reachable on synthetic data
  (`tests/test_pipeline.py::test_infeasibility_cause_alpha_induced`) — does
  not materialize anywhere in this benchmark's realistic operating range.
  Whether that argues for or against including the certificate in the paper
  is a judgment call now backed by a real number, not a guess.
- **LP pre-check savings — the plan's assumption about where the benefit
  comes from does not hold up empirically.** Real-data pass: 115 Stage2
  instances harvested from the α sweep (every `(query,k,α)` cell that
  reached Stage 2) replayed with both `lp_precheck` settings — **0 of them
  were ever LP-infeasible**, because (as above) every certified-feasible
  query stays Stage-2-feasible at every α; there is no infeasible case in
  real santos-scale pools (`|P|≤~10`) for the pre-check to shortcut.
  Supplementary **synthetic** demo (`lp_precheck_synthetic.csv`, distinct
  `experiment` column, not merged with real-data rows) at `|P| ∈
  {100,400,1000,3000}`, mixed feasible/genuinely-infeasible instances:
  **`lp_precheck=True` is consistently SLOWER than calling `milp` directly**,
  on both outcomes, and the gap widens with pool size (`|P|=3000`,
  infeasible case: 9.7ms with pre-check vs 2.9ms without — pre-check ~3.3x
  slower; feasible case: 84ms vs 72ms). Reason, worth recording since it
  contradicts `stage2_ilp.py`'s own docstring reasoning ("skip
  branch-and-bound when the LP already proves infeasible"): NOTES.md's own
  LP/ILP-equivalence proof for this constraint shape means there is no
  branch-and-bound to skip in the first place — an infeasible instance is
  *always* caught at `milp`'s own initial LP relaxation, which costs the
  same as calling `linprog` separately, so the explicit pre-check only adds
  a second Python-level solver dispatch on top. **The pre-check's
  demonstrated value remains the correctness one already documented**
  (pairing it with `presolve=False` avoids the presolve trap, NOTES.md) —
  the speed rationale in the comments should be corrected, not the code
  (out of scope here; `duts/` is not touched by this phase).
- **Retrieval ablation** (`k=10, α=2`, 192 rows): `hnsw` and `exact_scan`
  are statistically indistinguishable (both 8 feasible / 37
  `insufficient_candidates` / 3 `stage2_infeasible`, matching mean `n_D`
  within noise) — consistent with Phase B's ≥0.95 mean recall finding.
  `inverted_index` vs `null` is the real lever: dropping the overlap filter
  (`null`) roughly doubles `n_D` (6.2→9.9) by admitting semantically-similar
  but `M`-irrelevant tables, which *raises* feasible count slightly (8→9)
  but **increases `stage2_infeasible` sharply (3→17)** — more candidates
  that are semantically close but don't actually carry `M`'s value make
  Stage 2's distribution constraint harder to satisfy, not easier. A real,
  non-tuned finding: the overlap filter is not just a speed optimization,
  it is load-bearing for feasibility.
- **Efficiency claim, across the whole sweep, not spot checks.**
  `n_unionability_computations == n_P == min(⌊α×k⌋, n_D)` checked on all
  200 alpha-sweep rows that reached Stage 1 (excludes `insufficient_candidates`
  and `intrinsic`, which never call Stage 1 at all) — **0 violations**.
  `|T|`-independence itself is not re-demonstrated here (santos' datalake
  size is fixed at 550 throughout this sweep); that specific claim is Phase
  D's dedicated real-data test
  (`test_n_unionability_computations_independent_of_datalake_size_real_data`),
  not repeated in Phase E.
- **Minor, documented nondeterminism**: two otherwise-identical full-sweep
  reruns split `k=5`'s 48 queries as 15/19/14 vs 15/20/13 across
  feasible/insufficient_candidates/intrinsic — one query moved category.
  `HnswRetriever(seed=42)` pins hnswlib's RNG but not thread-scheduling
  order in its (multi-threaded) index construction (Phase B's DISTANCE-METRICS.md
  already flags this for recall numbers); every other column, and
  everything not touching the semantic retriever, was reproduced exactly.

---

## 5. Config

One block, adapters chosen by name — the swap surface in practice:

```json
{
  "benchmark": "santos",
  "adapters": {
    "synopsis":     "metadata_store",
    "semantic":     "hnsw",
    "overlap":      "inverted_index",
    "unionability": "pinned_match"
  },
  "theta_cat": 50, "sigma": 0.6, "top_n": 100,
  "k": 10, "alpha": 2, "F_star": 0.5, "delta": 0.05,
  "include_query": true, "certify": false, "seed": 42
}
```

The current shipping overlap adapter (`InvertedIndexOverlap`) already matches
the paper's §7.2 design (v7) exactly. Adding a Roaring-bitmap-backed variant
per §7.2's implementation recommendation — the live example of the swap
surface now that JOSIE is off the table — would still be one new adapter
class + one config string. No change to `retrieval.py`, `runner.py`, or
`duts/`.

---

## 6. Verified starting facts

Checked against the live data while writing this, not assumed:

- **santos**: 550 datalake tables, 50 query tables, 6322 datalake columns.
- **Vectors** (`cl_{datalake,query}_drop_col_tfidf_entity_column_0.pkl`):
  `List[(table_name, ndarray(n_cols, 768))]`, float32, **not L2-normalized**.
- **`MetadataStore`** loads from `data/santos/indexes/metadata_datalake.pkl`;
  `categorical_distributions` values are Python `int` ⇒ exact `N_i`, no float drift.
  **Update (Phase A):** `n_i` and stored histograms are NOT stale relative to
  the CSVs (verified exactly, 0 mismatches over all 550+50 tables), but the
  pickle's own notion of "categorical" caps at domain size ≤ 10 empirically,
  not the configured `θ_cat = 50` — see Phase A below and
  `dutsx/adapters/synopsis.py`'s docstring for the full finding.
- **`protected_attributes_santos.csv`**: 96 rows, `q_name,
  protected_attribute_id, protected_value`; attribute given by **index**, value
  is scalar.
- **`typing.Protocol`** is available on py3.8.5 — no `typing_extensions` needed.

## 7. Open, and blocking what

1. ~~**JOSIE delivery** (I1)~~ — **moot as of `Fair_Table_Search_7.pdf`.** The
   paper's own §7.2 rewrite dropped JOSIE in favor of posting lists, which is
   what `InvertedIndexOverlap` already implements, exactly, unmodified. No
   `JosieOverlap` adapter is needed or should be built. The only remaining
   item from the old §7.2 recommendation is a Roaring-bitmap-backed posting
   list for efficiency at scale — not blocking, not needed at santos scale.
2. **santos copy** — **resolved (Phase A).** Used `/u6/bkassaie/starmie_fair/data/santos/`
   as-is; the staleness check ran against it and found no staleness (0
   mismatches, `n_i` and full histograms, all 550+50 tables), so no new copy
   is needed for Phase A or later phases.
3. **Set-valued `M`** — the protected-attributes CSV is scalar-per-query. Do
   experiments need genuine `|M| > 1` cases, and if so where do they come from?
   Blocks nothing now (`M = {v}` works); shapes Phase E's design.
