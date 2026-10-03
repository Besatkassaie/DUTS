# NOTES

One line per phase (PLAN.md §5), plus anything discovered during
implementation that the plan didn't anticipate.

- **Phase 0 (scaffold).** `duts/types.py`, `duts/config.py`, `configs/duts.json`.
  Environment: `pytest` was missing from `TableUnionNew` (checked all 8 conda
  envs on this machine, PLAN.md's flagged risk) but installs cleanly via pip;
  installed. `assert_environment()` runs at package import.
- **Phase 1 (`stats.py`).** Weighted-average identity and the
  aggregation-sensitivity witness pass exactly. 21 tests.
- **Phase 2 (`stage1_dinkelbach.py` + oracle).** Exact vs. brute force over
  500 random instances (`m ≤ 16`, ties/boundary cases forced). `λ⁽⁰⁾` seeded
  from a real size-(α×k) subset per the C5 correction, not the full-pool
  ratio. The C4 counterexample and the `λ⁽⁰⁾` overshoot are both regression
  fixtures now, not just plan prose. 707 tests.
- **Phase 3 (`stage2_ilp.py`).** Exact vs. brute force over 400 random
  instances (`m ≤ 18`). Found and fixed a real `scipy.optimize.milp` bug
  along the way — see below. 706 tests.
- **Phase 4 (`pipeline.py`).** `certify → Stage 1 → Stage 2` with full
  `Telemetry`; `PipelineResult` now also exposes `pool` (P) so
  `R ⊆ P ⊆ D` is directly checkable, not just inferable. All three
  infeasibility causes (`insufficient_candidates`, `intrinsic`,
  `alpha_induced`) exercised with concrete instances — `alpha_induced` was
  found by random search rather than hand-built, since Stage 1's
  aggregate-optimal pool and "the k tables forming the best k-subset" are
  genuinely different combinatorial objects once `α > 1`. α-sweep
  (`α ∈ {1,2,3,5,10} × k ∈ {5,10,20}`) run on synthetic `D`; `satisfice`
  excluded since it has no defined objective yet (C3, open). 326 tests.
- **Generators + E-kKP fixture.** `tests/generators.py`:
  `random_candidates`, `knapsack_to_tables`, `tau_from_knapsack_capacity`
  (using the corrected `τ = 1 − C/(k·c)`, not the original `(k−1)/k`).
  40 random knapsack instances solved via `stage2_ilp.solve` and checked
  against a brute-force knapsack optimum — all matches, some skipped where
  the random instance had no feasible knapsack solution at that cardinality.
- **Full suite.** `pytest` (installed into `TableUnionNew`): **1761 passed,
  19 skipped**, 0 warnings, ~7.5s.

## Post-implementation change (2026-08-09)

**`pipeline.run` gained a `certify: bool = False` parameter — the C4 certificate
is now opt-in, off by default.** User request, "for now": skip the extra
Dinkelbach run at cardinality `k` over all of `D` before Stage 1. Stage 1 and
Stage 2 are unaffected and remain exact either way — only the diagnostic on a
`∅` result changes. A new `infeasibility_cause` value, `"stage2_infeasible"`,
covers the case where `certify=False` and Stage 2 fails on `P`: without
running the certificate first there's no way to tell `intrinsic` from
`alpha_induced`, so the code no longer guesses. `"intrinsic"` and
`"alpha_induced"` are now only reachable with `certify=True`. Tests updated:
the two dedicated certificate tests
(`test_infeasibility_cause_intrinsic`/`_alpha_induced`) now pass
`certify=True` explicitly; the property-based sweep exercises both settings
per instance and checks the appropriate cause set for each; a new
`test_certify_defaults_to_false` pins the default behavior directly. PLAN.md's
Phase 4 description ("certify → Stage 1 → Stage 2") is the `certify=True`
path — still fully built and tested, just no longer the default invocation.

## Found during implementation, not anticipated by PLAN.md

**A real `scipy.optimize.milp` correctness bug (scipy 1.10.1 and 1.13.1).**
With HiGHS's default `presolve=True`, `milp` can return `success=True` on a
genuinely infeasible instance, with a solution that *violates* the
constraint it was asked to satisfy. Reproduced with a concrete 7-variable
instance (two independent constraint formulations), confirmed via brute
force that the instance is truly infeasible, and confirmed
`options={"presolve": False}` gives the correct verdict. Frequency in random
search: ~1-in-3000. Fix applied in `stage2_ilp.py`: presolve disabled on
both `milp` and `linprog` calls, plus a defensive re-verification of every
"successful" solution against the actual constraints before it's trusted.
Permanent regression fixture: `tests/test_stage2.py::test_milp_presolve_trap_regression`.
This is now documented as "trap 3" in `PLAN.md` Phase 3, alongside the two
traps already known (minimize-vs-maximize, non-binary bounds) — all three
are necessary for Stage 2 to actually deliver the exactness the paper
claims, not just theoretical hazards.

**LP/ILP equivalence for Stage 2's specific constraint shape.** PLAN.md's
Phase 3 originally asked for an LP-feasible-but-ILP-infeasible test fixture.
That fixture cannot be constructed: Stage 2's feasible region (one
cardinality equality + one linear inequality + box bounds) is LP/ILP
equivalent by a standard exchange argument, proven and verified by 1000
random instances (zero gaps, once the presolve bug above is worked around).
`PLAN.md` updated to replace that acceptance item with an equivalence check.
This is a stronger fact than the paper's one-directional soundness claim and
worth a line in the paper itself.

**`alpha_induced` infeasibility needed a search, not hand construction.**
At `α = 1`, Stage 1's pool computation is *literally the same Dinkelbach
call* as the C4 certificate, so the pool always contains the certifying
k-subset and `alpha_induced` infeasibility is structurally impossible. It
only appears for `α > 1`, where Stage 1 optimizes the *aggregate* ratio over
`α×k` tables — a different combinatorial choice from "the k tables forming
the best k-subset" — so the winning k-subset can fall partly outside the
pool. Found by random search at trial 17930; kept as
`tests/test_pipeline.py::test_infeasibility_cause_alpha_induced`.

## Phase A (`dutsx/` — ports + synopsis adapter, PLAN-integration.md)

`dutsx/ports.py` (4 `Protocol`s, `@runtime_checkable`), `dutsx/adapters/synopsis.py`
(`MetadataStoreSynopsis`, `CsvSynopsis`), `dutsx/registry.py` (config-driven
`build`/`build_from_config`), `tests/test_dutsx_synopsis.py` (19 tests against
real santos data at `/u6/bkassaie/starmie_fair`, skips cleanly if that path is
absent). Full suite: 1780 passed, 19 skipped (1761 + 19 new), 0 warnings,
`pyflakes dutsx tests` clean.

**No staleness found** — contrary to what the Dec-2025-pickle-vs-2021-CSV date
gap suggested. `metadata_datalake.pkl`/`metadata_query.pkl`'s `n_i` matches a
fresh CSV row count exactly for all 550+50 tables, and a 40-table sample of
full `{value: count}` histograms matched byte-for-byte. The pickle appears to
have been re-serialized from the same CSVs, not built from different data.

**Found, not anticipated by PLAN-integration.md: the pickle's own
"categorical" filter is not `θ_cat = 50`.** Every one of the 2220¹
distributions actually stored in `metadata_datalake.pkl` has domain size ≤ 10
(checked exhaustively); columns with true domain size in `(10, 50]` genuinely
exist in the CSVs (e.g. `311_calls_historic_data_0.csv`'s `issue_type`,
domain 33) and are silently absent from the pickle itself — not filtered by
a downstream reader. Recomputing from scratch at the *actually-configured*
`θ_cat = 50` (`CsvSynopsis`) gives 3583 surviving attributes and 52
zero-categorical-attribute tables; the pickle as shipped gives 2220¹ and 96¹.
Root cause not fully recovered — the pickle's `_global_stats` dict is keyed
`total_categorical_columns`, but the checked-in `TableMetadata.py` pickles
`_global_stats` keyed `total_columns_with_distributions` and computes
distributions for *all* columns unconditionally, proving the pickle was built
by a since-edited version of the class. No script in the repo currently
reproduces that build. Practical consequence for later phases: **anything
that needs the configured `θ_cat = 50` must use `CsvSynopsis`, not
`MetadataStoreSynopsis`** — the two are intentionally independent
implementations for exactly this reason (`CsvSynopsis` is the staleness/
correctness oracle), and disagreeing on `categorical_attrs` here is *mostly*
that oracle doing its job, not a bug in either adapter¹. `PLAN-integration.md`
Phase A updated in place with the numbers.

¹ **Corrected 2026-08-13** — see "WDC metadata-store pickle + a real
`MetadataStoreSynopsis` bug" below. The original count here was 2345/95: 125
of those were a genuine bug (domain size wrongly included the `""` empty-cell
bucket), not attributable to the pickle's own pre-filtering. Fixed in
`dutsx/adapters/synopsis.py::MetadataStoreSynopsis.categorical_attrs`.

**Resolved: `biodiversity_2.csv`'s empty `categorical_distributions`.** Its 3
columns (`scientific name`, `family name`, `common name`) have domain sizes
257/91/257 over 257 rows — every one exceeds *both* the configured
`θ_cat = 50` and the pickle's actual ~10 cutoff. Not a bug in either version
of `TableMetadata.py`; the table genuinely has zero categorical columns under
any threshold in play. 96/550 datalake tables hit this under the pickle's
actual filter (95 before the 2026-08-13 correction above); 52/550 under the
correctly-configured `θ_cat = 50`.

## Phase C (`dutsx/adapters/unionability.py` — PLAN-integration.md)

`PinnedMatchScorer` (§6.2/M3 spec), `StarmieVerifyScorer` (wraps
`bounds.py:verify_constrained`, comparability only), `ConstantScorer` (test
double); registered as `pinned_match`/`starmie_verify`/`constant`.
`tests/test_dutsx_unionability.py`, 78 tests. Full suite: **1858 passed, 19
skipped**, lint clean, `duts/` untouched.

**M3 is now demonstrated rather than asserted.** Designed witness (not found
by search): similarity matrix `[[0.70, 0.95], [0.65, 0.23]]`, pin `(q₀, c₀)`.
§6.2-as-specified scores **0.70**; `verify_constrained` scores **1.60**.
`verify_constrained` pins only the *row* (big-bonus trick), so Hungarian
re-routes `V_D` to the higher-similarity column `c₁` and picks up `q₁→c₀` —
exactly the re-optimization §6.2 forbids, since the pair was already fixed
during candidate generation. Converse also tested (they agree when the pin is
the unconstrained choice), plus a 60-instance property test that
`pinned ≤ starmie` always — forcing `q_i→t_j` optimizes over a strict subset
of what `verify_constrained` considers, so a violation would indicate
double-counting or reuse of the pinned column.

Two deliberate deviations, both recorded in code: kept `bounds.py`'s strict
`sim > σ` rather than §6.2's stated `≥` (comparability with Starmie's
published numbers), and used `scipy.optimize.linear_sum_assignment` instead of
`munkres` (both exact; scipy is already a hard dependency, and deleting the
pinned row *and* column makes this strictly cheaper than the big-bonus trick).

**Built inline, not by a subagent.** Phases B and C were dispatched to
parallel agents; both died on an API session-quota limit before writing
anything, leaving the repo clean at Phase A. Phase C was then implemented
directly in the main session while the quota was exhausted. Phase B
(retrieval) was completed in a later session — see below.

## Phase B (`dutsx/adapters/{semantic,overlap}.py` + `dutsx/retrieval.py` — PLAN-integration.md)

`HnswRetriever` + `ExactScanRetriever` (`SemanticRetriever`);
`InvertedIndexOverlap` + `NullOverlap` (`OverlapFilter`); `retrieval.py`'s
`retrieve_candidates` (§7.3: `D_pair = D_sem ∩ D_ovl` at the `(table,
attr_idx)` pair level, then per-table `argmax sim`). Registered as
`hnsw`/`exact_scan` and `inverted_index`/`null` — additive only, Phase C's
`UNIONABILITY_REGISTRY` untouched. `tests/test_dutsx_retrieval.py`, 31 tests.
Full suite: **1889 passed, 19 skipped** (1858 + 31 new), lint clean, `duts/`
untouched.

**Plan correction, found while implementing.** PLAN-integration.md's Phase B
paragraph and PLAN-full.md's I3 both said "build hnswlib with
`space='cosine'`" *and* described the accept criterion in Euclidean-distance
terms (`θ_dis = sqrt(2−2σ)`, requiring explicit L2-normalization). Those are
two different code paths, not one. Resolved via coordinator clarification
(2026-08-09): ship `space='cosine'` over the raw, unnormalized vectors —
hnswlib's cosine space normalizes internally and returns `1 − cos_sim`
directly, so `HnswRetriever` needs no explicit normalization step or
`θ_dis` conversion at all. **The `θ_dis = sqrt(2−2σ)` identity is still
verified, just not through the shipped code path**: one test checks the
algebraic identity `‖a−b‖² = 2−2·cos(a,b)` exactly by brute force (max abs
error `2.24e-07`, 0 mismatched threshold sets over 156 real santos query
columns), and a second routes the same check through two independently-built
real hnswlib indexes (`space='l2'` over explicitly-normalized vectors vs
`space='cosine'` over raw vectors) to see how much ANN approximation adds on
top of the exact math (mean Jaccard `0.9953`, min `0.9608` over 274 query
columns — small, expected, not a sign the identity is wrong).
`PLAN-integration.md` Phase B updated in place with this and the recall
numbers below.

**`HnswRetriever` is self-contained vanilla `hnswlib`, not a subclass or
wrapper of anything in `starmie_fair`.** `HNSWSearcher_Fair.py:152-156` was
read only as an API-usage reference (the four-call shape:
`Index(space=...)` / `init_index` / `add_items` / `knn_query`); `dutsx/`
never imports it, and `PLAN-full.md`'s asset-table entry for a `hnsw_fair/`
built fork is stale — no such directory exists on disk, confirmed while
building this phase.

**Real recall numbers (santos, 550 datalake tables / 6322 columns — small
enough to brute-force exactly, not sampled for the table/column counts).**
`ExactScanRetriever` (brute-force cosine, the oracle) vs `HnswRetriever`
(`ef_construction=200, M=32, ef=100`) at `top_n=100, σ=0.6`, over every
santos query column with a non-empty exact-scan result (615 of 50 query
tables' columns): **mean recall 0.998, min recall 0.95**. Test asserts
conservative floors (`≥0.95` mean, `≥0.85` min) rather than the exact
observed values, since hnswlib's construction RNG can shift results slightly
run to run.

**§7.3's pair-level (not table-level) intersection, demonstrated with a
constructed witness, not just asserted.** A table entering `D_sem` via
attribute `A1` and `D_ovl` via a *different* attribute `A2` yields **no**
surviving pair, even though both filters independently "found" the table —
`tests/test_dutsx_retrieval.py::test_table_entering_via_different_attributes_yields_no_pair`.
Converse (same attribute both times) survives; a table with two attributes
clearing both filters keeps only the higher-`sim` one. `RetrievalTelemetry`
(`n_sem, n_ovl, n_pair, n_d`) demonstrated against real adapters for both an
arbitrary probe (often `n_pair=0` — a correct, uninteresting outcome when
the probe and `M` don't happen to align) and a self-referencing probe
(query vector = a table's own column embedding, `M` drawn from that same
column's own values) constructed to guarantee `n_pair≥1`, exercising both
the empty and non-empty paths against real data.

## Phase D (`dutsx/runner.py` — PLAN-integration.md)

`QueryTask`/`RunResult`/`RunTelemetry`/`RunnerContext`/`UnscoredCandidate`,
`run_query`, `load_queries_from_csv`, `retrieve_unscored_candidates`.
`tests/test_dutsx_runner.py`, 18 tests. Full suite: **1907 passed, 19
skipped** (1889 + 18 new), lint clean, `duts/` untouched,
`duts.pipeline.run` never called (composes `stage1_dinkelbach` +
`stage2_ilp` directly, per PLAN-integration.md §3).

**The "score P only" guarantee is structural, not commented.**
`UnscoredCandidate` (the type `D` and Stage 1's output are built from) has
no `U` field at all — `duts.stage1_dinkelbach` and `certify_feasible` only
ever read `.table`/`.N`/`.n` so they run unmodified against it (duck
typing), but `duts.stage2_ilp.solve` reads `.U` and would fail with
`AttributeError` if ever handed `D` directly instead of the scored pool.
`_score_pool` (in `runner.py`) is the only call site of
`UnionabilityScorer.score` in the module.

**`protected_attributes_santos.csv` does not cover all 50 santos queries —
found while building the loader, not anticipated by PLAN-integration.md
Phase D's original wording.** 96 rows, but only 48 have a `q_name` matching
an actual file in `santos/query/`; the other 48 all end in `_fair.csv` and
belong to `santos3/query/`, a different benchmark directory never mounted
under `santos/`. Separately, 2 of the 50 real query files (`albums_b.csv`,
`film_locations_in_san_francisco_a.csv`) have no row in the CSV at all —
genuinely absent, not a parsing bug. `load_queries_from_csv` returns
`(tasks, skipped)`; skipped rows are not an error. PLAN-integration.md's
Phase D accept criteria and §4 wording updated in place with the real count
(48, not 50) and actual run numbers.

**All four infeasibility causes and the efficiency claim verified with real
santos data, not just stubs.** 48-query run at `k=3, α=2, F*=0.3, δ=0.15`:
18 feasible, 17 `stage2_infeasible`, 13 `insufficient_candidates`;
`Δ(F*,F_R) ≤ δ` and `n_unionability_computations == len(P)` checked
per-query. `n_unionability_computations` independence from `|T|` shown two
ways: a real-data test (same query, 550-table vs. 38-table datalake, pool
size 6 in both) and a stub-adapter test isolating the claim from retrieval
mechanics entirely (`|D|` 20→200→2000, scoring calls fixed at 10). The
`alpha_induced` cause — structurally rare, per Phase 4's note that it
needed random search to find even inside `duts/`'s own test suite — is
reproduced at the runner layer by replaying
`tests/test_pipeline.py::test_infeasibility_cause_alpha_induced`'s exact
numeric witness through stub adapters rather than searching for a fresh one
in real data (48 queries at three different `(k,α,F*,δ)` settings never hit
it by chance, consistent with it being a rare structural interaction, not
absent from the runner's logic).

## Phase E (`experiments/` — PLAN-integration.md, final phase)

`experiments/{schema,context,alpha_sweep,lp_precheck,retrieval_ablation,analysis,cli}.py`.
`tests/test_experiments.py`, 18 tests (16 stub/synthetic, 2 tiny real-data
smokes). Full suite: **1925 passed, 19 skipped** (1907 + 18 new), lint clean,
`duts/` untouched. Real sweeps run against all 48 usable santos queries and
committed under `experiments/results/*.csv` (+ `.parquet` twins) — see
`PLAN-integration.md` Phase E for the full numbers; summarized here are the
two findings that weren't anticipated by the plan.

**M6's answer, on this dataset: `alpha_induced` infeasibility never occurs,
anywhere in the real α sweep.** `α ∈ {1,2,3,5,10} × k ∈ {5,10,20}`,
`certify=True` throughout (720 rows) — the infeasibility-cause split is
*identical across every α* at a fixed `k`, and the only two causes that ever
appear are `insufficient_candidates` and `intrinsic`. Retried at much
tighter `τ` (`F*=0.6/δ=0.05` and `F*=0.9/δ=0.02`, both checked directly, not
assumed) — same result. Root cause: santos' retrieval funnel returns very
small pools (mean `|D|` 6–10 tables), so for `α≥2` at every tested `k`,
Stage 1's pool is already `D` itself (clamped, C5), and whenever the C4
certificate says some k-subset of `D` clears `τ`, Stage 1's F-maximizing
selection at every α happens to contain one too — there's no room for the
"aggregate-optimal pool misses the best k-subset" failure mode
`test_infeasibility_cause_alpha_induced` proves is real on synthetic data
(NOTES.md Phase 4) to show up when `α×k` this consistently exceeds `|D|`.
The certificate still earns its keep for the `insufficient_candidates` vs.
`intrinsic` split (`intrinsic` is 0/13/3 across `k=20/5/10` — not
negligible), just not for the α-induced case it was originally motivated by.

**A genuine, unanticipated finding that corrects `stage2_ilp.py`'s own
comment: the LP pre-check's claimed *computational* benefit does not survive
measurement, though its correctness benefit (the presolve-trap workaround)
is untouched.** The real-data harvest (115 Stage2 instances across the whole
α sweep) found zero LP-infeasible pools at all — santos pools are too small
(`|P|≤~10`) to ever exercise the "skip branch-and-bound" path — so a
synthetic supplementary demo was added (`experiments/lp_precheck.py::run_lp_precheck_synthetic_demo`,
`experiments/results/lp_precheck_synthetic.csv`, distinct `experiment`
column, never merged with real-data rows) at `|P| ∈ {100,400,1000,3000}`
with genuinely infeasible instances built the same way `tests/test_stage2.py`
does. Result: `lp_precheck=True` is **consistently slower** than calling
`milp` directly, and the gap widens with pool size (`|P|=3000`: 9.7ms vs.
2.9ms on infeasible instances — pre-check ~3.3x slower). Why: the
LP/ILP-equivalence proof already on record in this file (previous section)
means an infeasible instance is *always* caught at `milp`'s own initial LP
relaxation — there is no deeper branch-and-bound search to skip, so the
explicit `linprog` pre-check is pure added overhead (a second Python-level
solver dispatch), not a shortcut. `stage2_ilp.py`'s docstring reasoning
("its only value here is computational -- skip branch-and-bound when the LP
already proves infeasible") is the part that's wrong; the presolve-trap
mitigation the pre-check is paired with (`options={"presolve": False}` on
both calls, post-solve re-verification) is unaffected and still load-bearing
— `duts/` was not modified to fix the comment, since this phase's scope is
`experiments/` only; flagging it here per this repo's convention of
recording plan/code contradictions rather than letting them drift.

**Retrieval ablation confirms the overlap filter is load-bearing for
feasibility, not just speed.** `hnsw` vs. `exact_scan` are statistically
indistinguishable (consistent with Phase B's recall numbers). Dropping
`InvertedIndexOverlap` for `NullOverlap` roughly doubles mean `n_D` (6.2→9.9)
but **increases `stage2_infeasible` from 3/48 to 17/48** — semantically
similar tables that don't actually carry `M`'s value make Stage 2's
distribution constraint harder to satisfy, not easier, when admitted into
the pool.

**Minor documented nondeterminism, not a harness bug.** Two full-sweep
reruns (identical flags, `seed=42`) split `k=5`'s causes as 15/19/14 vs.
15/20/13 (feasible/insufficient_candidates/intrinsic) — one query moved
category between runs. `HnswRetriever`'s `random_seed=42` pins hnswlib's RNG
but not its (multi-threaded) index-construction thread-scheduling order,
already flagged for recall numbers in `DISTANCE-METRICS.md`/Phase B. Every
other column reproduced exactly across reruns; `tests/test_experiments.py`'s
determinism tests use stub/synthetic adapters specifically to avoid this
source of noise.

## Phase F — groundtruth precision/recall, and the Stage-1-skip ablation (2026-08-09/10)

**Groundtruth precision/recall (`experiments/groundtruth.py`,
`experiments/groundtruth_eval.py`, CLI `groundtruth-eval`).** Scored `D`/`P`/`R`
at each stage against `<benchmark>_small_benchmark_groundtruth.csv` (santos's
own table-union-search groundtruth — a different, coarser notion of
"unionable" than DUTS's distribution constraint; see the module docstrings
for exactly what this does and doesn't test — column identity is ignored).
At `k=3, alpha=2, F*=0.3, delta=0.15`: **precision is 1.000 at every stage
(D, P, R) on both santos (18/48 feasible) and santos3 (42/48 feasible)** —
every table DUTS ever retrieves, pools, or selects is independently
considered genuinely unionable by the benchmark. Recall falls off through
the funnel as expected (santos: 0.465→0.300→0.087; santos3: 0.310→0.200→
0.109) — not a flaw: the groundtruth's join column and DUTS's `V_D` are
different columns for 44/48 queries, and DUTS solves a strictly harder
problem (`k` distribution-satisfying tables, not exhaustive union coverage).

**Stage-1-skip ablation (`experiments/stage1_skip_ablation.py`, CLI
`stage1-skip-ablation`, user request: "how long it takes to skip the
fractional optimization step, and directly go to step 2, vs combined").**
Two conditions per query, same retrieval/scoring code: `two_stage` (normal
pipeline, `dutsx.runner.run_query` unmodified) vs. `skip_stage1` (score
every candidate in `D`, hand all of `D` to Stage 2's ILP at cardinality `k`,
bypassing Dinkelbach entirely). Deliberately does NOT reuse
`dutsx.runner._score_pool` — that function's docstring makes a grep-able
"exactly one call site, and it is not `D`" guarantee the ablation would
falsify by calling into it with `D`; `_score_all` in the ablation module
duplicates its five lines instead.

Only informative when Stage 1 actually has candidates to prune — at
santos's default `top_n=100`, retrieved `D` is small enough (~10-20) that
`alpha=2` clamps `P` to `D` (C5) for most queries and the two conditions
collapse to identical numbers. Real separation needs small `alpha` (1) and
wide `top_n` (500). At `k=10, alpha=1, top_n=500`:

| benchmark | n scored (two_stage → skip_stage1) | scoring_time_s | sum_U | F_R |
|---|---|---|---|---|
| santos  | 10.0 → 14.6 (+46%)  | 0.0095s → 0.0134s (+41%)  | 86.98 → 94.44 | 0.525 → 0.486 |
| santos3 | 10.0 → 22.0 (+120%) | 0.0082s → 0.0170s (+108%) | 93.23 → 99.21 | 0.484 → 0.419 |

Two findings: **(1)** `two_stage`'s scoring cost is flat at `alpha*k`
regardless of datalake size; `skip_stage1`'s tracks `|D|` and grows with the
benchmark (santos3's larger, planted-table-enriched datalake retrieves
more candidates) — the paper's central efficiency claim (`duts/`'s docstring,
`CLAUDE.md`), made a real, measured runtime delta rather than an asymptotic
argument. **(2)** `skip_stage1` always wins on raw `sum_U` (guaranteed —
Stage 2 there optimizes over a superset, `D ⊇ P`; the ablation's own test
suite, `tests/test_experiments.py`, asserts this monotonicity holds and it
does on all real-data instances checked) but **`two_stage` consistently
lands closer to `F_star=0.3`** (0.525/0.484 vs. 0.486/0.419) — because Stage 1
explicitly runs an unbounded `argmax F` (C3) before Stage 2 ever sees the
pool, while `skip_stage1` has no such pre-filter and Stage 2 alone only
enforces the `F_R >= tau` floor, not proximity to `F*`. So Stage 1 isn't
just an efficiency device here — it's also *why* the two-stage pipeline's
results track the target proportion more closely, a quality effect the
efficiency framing alone doesn't capture.

## C4 certificate removed from the result path; all sweeps regenerated (2026-08-10)

**User instruction:** "I do not want the certificate C4 to be checked at all as it is not right" —
superseding the 2026-08-09 decision that merely defaulted `certify=False`. The certificate is now
*unreachable* from the experiment/runner path, not just off by default.

**Removed:** `QueryTask.certify`; the `certify_feasible` branch in `dutsx.runner.run_query`;
`RunTelemetry.{dinkelbach_iterations_certify, F_max_k, certify_time_s}`; the matching four
`ResultRow` columns (`certify`, `dinkelbach_iterations_certify`, `F_max_k`, `certify_time_s`);
the `--certify` CLI flag; `certify=True` in the alpha sweep. The causes `intrinsic` and
`alpha_induced` no longer exist — the cause enum is now
`None | "insufficient_candidates" | "stage2_infeasible"`. Two tests were deleted
(`test_certify_defaults_to_false`, `test_intrinsic_cause_when_certify_true_short_circuits_before_scoring`)
and the alpha_induced witness was retained under a new name
(`test_former_alpha_induced_witness_now_reports_stage2_infeasible`) because it is exactly the case
the removal makes indistinguishable. 1929 pass, 19 skip.

**Deliberately NOT removed:** `duts.stage1_dinkelbach.certify_feasible`, `duts.pipeline.run`'s
`certify=` parameter, and `duts.types.Telemetry`'s certificate fields. `duts/` is the verified pure
core with its own C4 counterexample regression fixture, this repo has no git history to restore
from, and nothing in the experiment path can now reach that code. Deleting it is a one-line ask if
wanted.

**The removal changed no answer.** Matching all 720 pre- and post-removal alpha-sweep rows on
`(q_table, attr, k, alpha)`: `feasible` identical on all 720; `n_R`/`sum_U`/`F_R` identical; the 85
`intrinsic` rows became `stage2_infeasible` and nothing else moved. The certificate was purely
diagnostic. Net cost of removal on this data: unionability computations 1158 → 1812 (+56%) and mean
wall time 3.04 → 3.61 ms (+19%), because the 85 instances it used to reject before Stage 1 now run
the full path. It saved 235 Dinkelbach iterations (0.0045 s) — i.e. it short-circuited more work
than it consumed, so it was a net *saving* here, not a net cost. That is the honest read for M6.

**Verified while regenerating** (user asked to confirm `V_D`/`M` come from starmie_fair's file):
`protected_attributes_santos.csv` supplies `protected_attribute_id` (0-based column index → `V_D`)
and `protected_value` (→ `M`). Reading all 48 query tables in full, the value appears in the named
column for **48/48** queries (4/48 at 1-based, confirming 0-based); zero `N_Q = 0` cases. The median
query's own proportion of its protected value is **0.061** against `F* = 0.3` — the queries start
far from their targets, which is most of why `τ = 0.15` is hard to meet.

**Two findings that fell out of the rerun**, both now in `experiments/RESULTS.md`:

- `delta_R` is **not** bounded by `δ`. The comment at `dutsx/runner.py` asserting
  `|delta_R| <= query.delta` on feasible results was wrong and is corrected: 60 of 115 feasible
  rows violate it, `delta_R` reaching −0.700 (`F_R = 1.000` vs `F* = 0.3`). Only the floor is
  enforced; C3's unbounded `argmax F` overshoots freely. This is the strongest empirical case yet
  for revisiting `satisfice`.
- The **LP pre-check is a net loss at every pool size this pipeline produces** — on real santos
  pools (~5–11 candidates) it costs 0.934 ms vs 0.729 ms, and on synthetic pools it only wins at
  `n_P ≈ 100`, losing 4× by `n_P = 3000`. Keep it for soundness framing, not for speed.

**Results file:** `experiments/RESULTS.md` — all five sweeps, plus a full label/column glossary
(the outcome labels, the query parameters, the funnel counts, and the quality/cost columns), which
is the part to read first when interpreting any `experiments/results/*.csv`.

## santos3-only report + fraction-reachability verification (2026-08-10)

**User instruction:** report only santos3 ("query tables are enriched and datalake tables as well"),
and "verify that the fraction indeed can be reached". Both done; `experiments/RESULTS.md` is now a
santos3-only document.

**Enrichment quantified** (verified by `ls` + set difference, not from docs): santos3's query set is
**48 of 48 `*_fair.csv`** rebalanced variants (santos: 0 of 50), and its datalake is **931 tables of
which 381 are `*_fair.csv`** planted — the 381 santos3 datalake tables absent from santos are exactly
that set. Enriched on both sides, as the user described.

**New: `experiments/fraction_reachability.py`** (CLI `fraction-reachability`, 3 tests). Retrieves `D`
as the pipeline does, then **brute-force enumerates every `k`-subset** and computes the exact
attainable set `{F(S)}`. Reports `F_min`/`F_max`, `tau_reachable`, `fstar_bracketed`, and `best_gap`
= `min |F(S) − F*|`. Deliberately does **not** call `duts.stage1_dinkelbach` — enumeration only, so
the numbers are independent of the solver they judge and nothing here reintroduces C4. Cost is
`C(|D|,k)` and `|D| ≤ 17` on santos3, so worst case ~24k subsets.

**The answer, three parts:**

1. **`τ = 0.15` is reachable for 100% (k=3, k=10) / 97.1% (k=5)** of queries that clear `|D| ≥ k`.
   End to end this shows as `stage2_infeasible` = 1 of 240 runs at k=5 and **0** at k=10 (base santos:
   14 and 3), and 0 at k=3 in groundtruth-eval. The enrichment works.
2. **`F*` itself is usually *not* bracketed** — median attainable range `[0.28, 0.42]`, `F*` inside it
   for only 38–40% of queries. Cause: **42 of 48 enriched query tables have `F_Q > F*`**, median
   `F_Q = 0.434` vs `F* = 0.3`. With `include_query=True` the query's own tuples anchor the ratio, so
   a query at 0.43 can't be pulled to 0.30. **The rebalancing overshot the target it's measured
   against** — worth raising with the benchmark's authors, and the mirror image of base santos, where
   median `F_Q` was 0.061, i.e. far *below* `F*`.
3. **The residual is mostly the data's fault, and that bounds the `satisfice` payoff.** Some
   `k`-subset lands within `δ` of `F*` for ~75% of queries (median best gap 0.018–0.029). Against the
   pipeline's actual `|F* − F_R|`: **81% / 19%** (k=5) and **90% / 10%** (k=10) data-imposed vs
   objective-imposed. So replacing C3's unbounded `argmax F` can recover at most ~a fifth of the
   remaining error. This is the first quantitative bound on what `satisfice` could buy.

**Also found on santos3, differing from base santos:**

- **`hnsw` vs `exact_scan` is no longer byte-identical** — outcomes match exactly, but `n_pair`
  8.958 vs 8.979 and `n_D` 7.625 vs 7.646 (≈0.3%). At 931 tables the ANN index starts missing
  near-ties. Still free in outcome; the caveat is now measured rather than assumed.
- **The overlap filter now costs real recall.** Dropping it raises feasibility 17 → 21 queries
  (on base santos it bought only 8 → 9). On enriched data the suppressed candidates are genuinely
  usable, so §7.2's 98.7% pair reduction is no longer nearly free.
- **Raising α moves `F_R` further from `F*`** (0.484 → 0.436 as α goes 1 → 10 at k=5): a bigger pool
  gives the unbounded `argmax F` more extreme tables. Least-overshooting config here is α = 1.

**Plumbing change:** `--benchmark` is now honoured by *every* CLI command, not just
groundtruth-eval/stage1-skip — `alpha-sweep`, `lp-precheck` and `retrieval-ablation` previously
hardcoded santos through `build_santos_context()`/`load_santos_base_tasks()` with no `paths=`. Result
filenames are now benchmark-prefixed (`santos3_alpha_sweep.csv`, …). ⚠️ The old unprefixed
`alpha_sweep.csv`, `lp_precheck.csv`, `retrieval_ablation.csv` in `experiments/results/` are the
**base-santos** runs from earlier the same day and are not overwritten by a santos3 run — don't mix
them. 1932 pass, 19 skip.

## WDC metadata-store pickle + a real `MetadataStoreSynopsis` bug (2026-08-13)

**User instruction:** WDC's `N_i`/`n_i` synopsis lookup (`CsvSynopsis`) was found (during the
`top_n`×`alpha`×`k` sweep writeup, `RESULTS-wdc-sweep.md`) to recompute `pandas.value_counts()` from
scratch on every call — never cached — and to be 72-90% of end-to-end time. `MetadataStoreSynopsis`
already exists and gives O(1) dict lookups against a prebuilt pickle (what santos uses), but no such
pickle had ever been built for WDC. Asked to build one.

**New script `scripts/wdc_build_metadata_store.py`** wraps `starmie_fair/TableMetadata.py`'s
`MetadataStore.build_metadata_from_csv()` + `.save()`, completely unmodified (read-only upstream) —
but calls `_process_csv_file` per-file in its own loop rather than the batch method, because 2/100,000
`tier_100k` CSVs have genuine embedded `\x00` bytes inside quoted field values (e.g.
`"v\x00deosxvip.com"`) that make Python's `csv` module raise `_csv.Error: line contains NUL` outright
(0/10,000 on `tier_10k`). Tolerant fallback strips just the NUL byte and keeps the rest of the string
(`_process_csv_file_tolerant`) — deliberately NOT matching what `pandas`' C parser does to the same
byte (silently *truncates* the field at the NUL, C-string-style, losing everything after it — verified
directly, not assumed). Neither recovers the "true" value; corrupted upstream data has no unambiguous
ground truth. Mine preserves more of the string; chose not to force artificial agreement with the
oracle's own worse truncation on this one narrow case.

Build cost: 3.4s (`tier_10k`, 10,000 tables) / 39.6s (`tier_100k`, 100,000 tables, one-time). Pickles
at `/u6/bkassaie/wdc_data/indexes/<tier>_metadata.pkl` (13.6MB / 130MB).

**Correctness verification (`scripts/wdc_verify_metadata_store.py`, matching
`tests/test_dutsx_synopsis.py`'s existing santos precedent):** exhaustive `n_rows` agreement on all
10,000 `tier_10k` tables, 5,000-table sample on `tier_100k`, 300-500 random `distribution()` pairs per
tier — all pass except the 2 NUL-byte files above (expected, documented, not investigated further).

**Two real bugs found and fixed while wiring this in, not just a caching layer:**

1. **`MetadataStoreSynopsis` had no `n_columns` method at all** (`CsvSynopsis` did) —
   `experiments/context.py::column_coverage_stats` needs it, immediate `AttributeError` on first real
   use. Never exercised before because nothing in the repo previously called `column_coverage_stats`
   against a `MetadataStoreSynopsis` instance. Added, mirroring `TableMetadata.num_columns`.
2. **`MetadataStoreSynopsis.categorical_attrs` counted the `""` empty-cell bucket toward
   `domain_size`** — `CsvSynopsis` explicitly excludes it (`df[col] != ""`), matching
   `_find_categorical_columns`'s `0 < nunique(dropna()) <= theta_cat` rule; `MetadataStoreSynopsis`
   just took `len(categorical_distributions[col])` directly, so an all-empty column got
   `domain_size=1` (just the `""` bucket) and was wrongly classified categorical. Surfaced immediately
   on WDC `tier_10k`: `n_columns_categorical` was 50851 via the (buggy) pickle path vs. 48857 via
   `CsvSynopsis` — a real, easily-reproduced discrepancy (274/2000 sampled tables affected), not
   subtle. **This bug predates WDC and also affects santos** — `tests/test_dutsx_synopsis.py`'s
   `test_metadata_store_pickle_filter_is_stricter_than_configured_theta_cat` had the old buggy count
   hardcoded (`total_attrs == 2345`, `tables_no_cat == 95`); fixed to the corrected `2220`/`96`.
   `dutsx/adapters/synopsis.py`'s module docstring, `NOTES.md`'s original Phase A entry (footnoted,
   not silently rewritten), and `PLAN-integration.md` Phase A all updated with the corrected numbers.
   After the fix, WDC's `n_columns_categorical` via `MetadataStoreSynopsis` matches `CsvSynopsis`
   **exactly** on both tiers (48857/48857, 487728/487728) — expected, since WDC's freshly-built
   pickle has no upstream pre-filter baked in (unlike santos's pickle, whose own unrecovered ~10-domain
   -size build-time filter is a separate, still-open issue this fix does not touch).

**Third bug found while measuring the speedup, unrelated to the metadata store itself:**
`dutsx/adapters/semantic.py::HnswRetriever.query`'s `ef`-matches-`top_n` logic (added earlier the same
day for the sweep) set `ef == k` exactly, no headroom — hnswlib occasionally raises `RuntimeError:
Cannot return the results in a contiguous 2D array` at that setting rather than just losing recall
(reproduced non-deterministically: crashed on 2 of 4 reruns of the identical `tier_100k`,
`top_n=5000`, 30-query sweep). Root cause is deeper than query-time `ef`: even `ef =
len(index)` (the full ~518K-vector graph) still fails for some query vectors (7/301 tested) — a real
limit of the graph's *construction*-time parameters (`ef_construction=200`, `M=32`; `k=5000` is 25x
`ef_construction`), not recoverable by any query-time setting. This was **already a latent risk in the
completed `RESULTS-wdc-sweep.md`/beamer-deck sweep** (7180 rows, never hit it — luck, not a guarantee).
Fixed with three layers: (1) `ef = max(k+50, 1.2k)` margin instead of `ef == k` exactly, (2) one retry
at `ef = len(index)` if that still fails, (3) bounded halving of `k` itself (deterministic, ≤~19
steps) if even that fails, returning fewer-than-requested candidates rather than crashing — every
caller already treats `top_n` as a breadth cap, never an exact count, so this is a graceful degrade,
not a correctness change. Verified: 0/301 previously-failing (table, attr) pairs fail after the fix.
Full suite reruns clean throughout (1974 pass, 19 skip) after every change in this entry.

**Speedup, measured directly (not estimated) — `run_topn_sweep` at matched `(tier, top_n)`, identical
30-query selections confirmed by set equality, `CsvSynopsis` vs. the new `MetadataStoreSynopsis` path:**

| | `tier_10k`, top\_n=1000 | `tier_10k`, top\_n=5000 | `tier_100k`, top\_n=1000 | `tier_100k`, top\_n=5000 |
|---|---:|---:|---:|---:|
| `union_lookup_time_s` speedup | 26.1x | 27.0x | 22.8x | 22.4x |
| `end_to_end_s` speedup | 3.06x | 3.03x | 5.07x | 4.55x |

Lookup speedup is consistent (~22-27x) across both tiers and both `top_n` values, as expected for an
O(pandas value\_counts scan) → O(1) dict-lookup swap. End-to-end speedup is lower and tier-dependent
(3x at `tier_10k`, 4.5-5x at `tier_100k`) because lookup was a smaller fraction of `tier_10k`'s total
time to begin with (Amdahl's law) — matches `RESULTS-wdc-sweep.md` §6's finding that lookup dominates
*more* at the larger tier (72% vs. 90% of end-to-end time).

**Wiring:** `experiments/wdc/context.py::build_wdc_context` now prefers `MetadataStoreSynopsis` when
`<root>/indexes/<tier>_metadata.pkl` exists, falling back to `CsvSynopsis` otherwise (e.g. `tier_1m`,
not yet built) — zero caller-visible change, `WdcBuildTimes` gained a `synopsis_source` field so which
path ran is always recorded, not implicit. **The already-completed `RESULTS-wdc.md` and
`RESULTS-wdc-sweep.md` numbers are unaffected and remain valid** — both were generated via
`CsvSynopsis`, and since the fixed `MetadataStoreSynopsis` now agrees with it exactly on WDC's
categorical-column counts, nothing about *which* candidates get retrieved or scored changes, only how
fast. Not rerun against the new pickle path — would only remeasure timing, not change any reported
outcome/feasibility number.
