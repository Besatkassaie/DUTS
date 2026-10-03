# DUTS experimental results — santos3

**Date:** 2026-08-10 · **Environment:** conda `TableUnionNew` (Python 3.8.5, numpy 1.24.4,
scipy 1.10.1, hnswlib) · **Seed:** 42 · **Suite:** 1932 passed, 19 skipped, 0 warnings.

This report covers **santos3 only** — the benchmark variant whose query tables *and* datalake tables
are both fairness-enriched. Every number here was produced on santos3; nothing is carried over from
the other variants.

The C4 feasibility certificate is **removed from the entire result-producing path** (user
instruction, 2026-08-10). There is no flag to re-enable it: `QueryTask.certify` is gone, the
`dutsx.runner` branch calling `certify_feasible` is gone, and the `certify`,
`dinkelbach_iterations_certify`, `F_max_k` and `certify_time_s` columns are dropped from the schema.
The causes `intrinsic` and `alpha_induced` no longer exist — see the glossary in [§7](#7-label-and-column-glossary).

```bash
PY=/u6/bkassaie/.conda/envs/TableUnionNew/bin/python
$PY -m experiments.cli all                   --benchmark santos3                          # §3, §4, §5
$PY -m experiments.cli fraction-reachability --benchmark santos3 --k {3,5,10}              # §2
$PY -m experiments.cli groundtruth-eval      --benchmark santos3 --k {3,5,10} --alpha 2     # §6.1
$PY -m experiments.cli stage1-skip-ablation  --benchmark santos3 --k 10 --alpha 1 --top-n 500  # §6.2
```

---

## 1. The dataset, exactly

**SANTOS `santos3`**, in the copy vendored under the `starmie_fair` repository. Read-only; nothing
here writes to it. Nothing was regenerated or resampled for these experiments — the one exception is
the explicitly-labelled *synthetic* supplement in §4, which uses generated `CandidateStats` and no
tables at all.

```
/u6/bkassaie/starmie_fair/data/
  santos3/datalake/    931 CSV tables    ← 550 base + 381 *_fair.csv planted tables
  santos3/query/        48 CSV tables    ← 48 of 48 are *_fair.csv  (100% enriched)
  santos3/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl   ← Starmie column embeddings
  santos3/vectors/cl_query_drop_col_tfidf_entity_column_0.pkl      ← query-side embeddings
  santos3/santos3_small_benchmark_groundtruth.csv                  ← §6.1's groundtruth
  protected_attributes_santos3.csv    96 rows                      ← V_D and M per query (§1.2)
```

### 1.1 What "enriched" means here, quantified

Verified by `ls` and set difference against the base `santos` benchmark:

| | santos (base) | **santos3** |
|---|---:|---:|
| Datalake tables | 550 | **931** |
| — of which `*_fair.csv` planted | 0 | **381** |
| Query tables | 50 | 48 |
| — of which `*_fair.csv` rebalanced | 0 | **48 of 48 (100%)** |
| On disk | 260 MB | 512 MB |

The 381 santos3 datalake tables absent from santos are exactly the `_fair.csv` set. So santos3 is
enriched on **both** sides: every query is a fairness-rebalanced variant, and 41% of the datalake is
planted material. This is the configuration in which the distribution constraint should be
satisfiable — §2 tests whether it actually is.

### 1.2 Where `V_D` and `M` come from

Read from starmie_fair's own file, not inferred here:
`dutsx/runner.py::load_queries_from_csv` parses `protected_attributes_santos3.csv`, taking
`protected_attribute_id` as `V_D` (a **0-based column index**) and `protected_value` as
`M = {value}` — a singleton set. The API takes a genuine value set per C2/M2; scalar is what this
data provides. All 48 queries resolve to a task (the CSV's other 48 rows name files in other
variants' directories).

### 1.3 Parameters per experiment

| § | Experiment | k | α | top_n | Runs |
|---|---|---|---|---|---:|
| 2 | Fraction reachability | 3, 5, 10 | — | 100 | 144 |
| 3 | α sweep | 5, 10, 20 | 1, 2, 3, 5, 10 | 100 | 720 |
| 4 | LP pre-check | pools harvested from §3 | — | — | 520 + 240 |
| 5 | Retrieval ablation | 5 | 2 | 100 | 192 |
| 6.1 | Groundtruth P/R | 3, 5, 10 | 2 | 100 | 144 |
| 6.2 | Stage-1-skip ablation | 10 | 1 | 500 | 96 |

`F* = 0.3`, `δ = 0.15` (so `τ = 0.15`), `include_query=True` throughout.

---

## 2. Can the target fraction actually be reached?

This is the question the rest of the report cannot answer, because every other experiment reports
what the pipeline **returned**, not what was **attainable**. `experiments/fraction_reachability.py`
retrieves `D` exactly as the pipeline does, then enumerates **every** `k`-subset of `D` by brute
force and computes the exact attainable set `{F(S) : |S| = k, S ⊆ D}`.

It deliberately does not call `duts.stage1_dinkelbach` at all — brute-force enumeration only, so
these numbers are independent of the solver they are used to judge, and nothing here reintroduces
the removed C4 certificate.

| Measure | k = 3 | k = 5 | k = 10 |
|---|---:|---:|---:|
| Queries analysed (of 48) | 42 | 35 | 17 |
| Skipped, `\|D\| < k` | 6 | 13 | 31 |
| **`τ = 0.15` reachable** | **100%** | **97.1%** | **100%** |
| `F* = 0.3` bracketed, `F_min ≤ F* ≤ F_max` | 38.1% | 40.0% | 17.6% |
| **Some `k`-subset within `δ` of `F*`** | **76.2%** | **74.3%** | **70.6%** |
| Median best gap `min\|F(S) − F*\|` | 0.029 | **0.020** | 0.018 |
| Median attainable range `[F_min, F_max]` | [0.286, 0.432] | [0.280, 0.424] | [0.307, 0.380] |
| Median query's own `F_Q = N_Q/n_Q` | 0.434 | 0.439 | 0.439 |

### The answer, in three parts

**1. Yes — the constraint floor `τ` is essentially always reachable on santos3.** 100% at k=3 and
k=10, 97.1% at k=5 (one query out of 35). Wherever retrieval supplies `k` candidates at all, a
distribution-satisfying answer exists. This is the enrichment working exactly as intended, and it
shows up directly in §3: `stage2_infeasible` is **1** of 720 α-sweep runs at k=5 and **0** at k=10.

**2. But `F*` itself is usually *not* bracketed — because the enriched queries sit above it.** The
median attainable range is `[0.28, 0.42]`, sitting mostly *above* `F* = 0.3`, and `F*` falls inside
the range for only 38–40% of queries. The cause is visible in the last row: **42 of 48 enriched
query tables have `F_Q > F*`**, with a median `F_Q` of **0.434**. With `include_query=True` the
query's own tuples are pinned into the ratio, so a query already sitting at 0.43 cannot be dragged
down to 0.30 by adding candidates unless those candidates are themselves poor in `M`. The
rebalancing pushed queries above the fraction DUTS searches for.

**3. This section is a data-attainability diagnostic, not a report card on the implemented
objective.** DUTS's actual constraint is the floor `F_R ≥ τ`, with no upper bound — Stage 1's
`argmax F` is unbounded by design (correction C3, resolved as the intended objective, not a
placeholder; see `CLAUDE.md`). Proximity to `F*` is not something either stage is asked to deliver,
so there is no "error" for the current pipeline to be charged with here. What follows is instead a
bound on what a *different*, currently-unimplemented objective (`satisfice`, which would minimize
`|F−F*|`) could achieve, computed against the same retrieved data:

| | k = 5 (34 queries) | k = 10 (17 queries) |
|---|---:|---:|
| Best attainable distance to `F*`, `min\|F* − F(S)\|`, median | 0.018 | 0.018 |
| Actual distance under `max_f`, `\|F* − F_R\|`, median | 0.080 | 0.035 |
| Additional distance from choosing `max_f` over a hypothetical proximity objective, median | **0.018** | **0.009** |
| Queries where `max_f` already matches the proximity-optimal choice | 8 / 34 | 4 / 17 |
| Split of the distance-to-`F*`: **set by retrieval / set by the `max_f` choice** | **81% / 19%** | **90% / 10%** |

So of the distance between `F*` and what `max_f` delivers on santos3, **81–90% is fixed by which
tables retrieval handed over** — no selection rule could do anything about that part — and only
10–19% is attributable to choosing `max_f` over a proximity-seeking alternative. That bounds the
entire payoff a `satisfice` implementation could ever deliver here: at most a fifth of this distance,
and only once the benchmark's own `F_Q > F*` construction (§1) stops working against it too.

---

## 3. The α sweep

`α ∈ {1,2,3,5,10} × k ∈ {5,10,20}` over all 48 santos3 queries = 720 runs.

| k | α | feasible | insufficient | stage2 infeas. | n_U | eff. α | mean ΣU | mean F_R | ILP ms | wall ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 34 | 13 | 1 | 3.65 | 1.00 | 46.66 | 0.484 | 1.4 | 7.0 |
| 5 | 2 | 34 | 13 | 1 | 6.06 | 1.75 | 47.84 | 0.443 | 1.5 | 7.8 |
| 5 | 3 | 34 | 13 | 1 | 6.79 | 2.17 | 48.19 | 0.438 | 1.5 | 7.9 |
| 5 | 5 | 34 | 13 | 1 | 6.88 | 2.73 | 48.31 | 0.436 | 1.5 | 7.9 |
| 5 | 10 | 34 | 13 | 1 | 6.88 | 4.08 | 48.31 | 0.436 | 1.5 | 7.9 |
| 10 | 1 | 17 | 31 | 0 | 3.54 | 1.00 | 69.42 | 0.476 | 1.5 | 4.9 |
| 10 | 2 | 17 | 31 | 0 | 4.35 | 1.73 | 70.78 | 0.454 | 1.5 | 5.2 |
| 10 | 3 | 17 | 31 | 0 | 4.35 | 2.37 | 70.78 | 0.454 | 1.6 | 5.2 |
| 10 | 5 | 17 | 31 | 0 | 4.35 | 3.66 | 70.78 | 0.454 | 1.6 | 5.2 |
| 10 | 10 | 17 | 31 | 0 | 4.35 | 6.89 | 70.78 | 0.454 | 1.6 | 5.2 |
| 20 | 1–10 | 0 | 48 | 0 | 0.00 | = α | — | — | — | 3.1 |

Mean `|D|` = 7.6 (min 1, max 17).

**Findings.**

1. **The distribution constraint is no longer the bottleneck.** `stage2_infeasible` is 1 of 240 runs
   at k=5 and 0 at k=10 — on the base santos benchmark the same sweep produced 14 and 3. This is §2's
   reachability result showing up end to end: on enriched data, if `k` candidates exist, a
   satisfying answer almost always exists too.
2. **Feasibility is now purely a retrieval-supply problem.** All 605 remaining infeasible runs but
   one are `insufficient_candidates`. At k=20, `|D| ≤ 17 < 20` for every query, so nothing is even
   attempted.
3. **α still does nothing.** The feasible/infeasible split is identical across all five α values at
   every `k`, and `eff. α` sits below the requested α in every row — mean `|D|` of 7.6 means the C5
   clamp binds for `α ≥ 2`. Quality saturates at α=3 (k=5) and α=2 (k=10).
4. **Raising α moves `F_R` further from `F*`** — not a constraint violation, since only `F_R ≥ τ`
   is required and every row here clears it, but a real effect on the diagnostic. `F_R` drifts
   0.484 → 0.436 as α goes 1 → 10 at k=5: a larger pool gives Stage 1's unbounded `argmax F` more
   extreme tables to pull from. Of these five α values, α = 1 lands closest to `F*` on this data.

### 3.1 Where the time actually goes

Mean per-query time in **milliseconds**, over the rows that reached scoring (`n_P > 0`) — the
`insufficient_candidates` rows never score anything and would otherwise dilute these means.

| k | α | n | **scoring** | ILP | LP pre-check | wall | scoring as % of wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 35 | **3.31** | 1.44 | 0.55 | 8.92 | 37.1% |
| 5 | 2 | 35 | **4.87** | 1.47 | 0.54 | 10.10 | 48.2% |
| 5 | 3 | 35 | **5.00** | 1.48 | 0.54 | 10.29 | 48.6% |
| 5 | 5 | 35 | **5.01** | 1.47 | 0.54 | 10.23 | 49.0% |
| 5 | 10 | 35 | **5.00** | 1.49 | 0.54 | 10.22 | 49.0% |
| 10 | 1 | 17 | **3.56** | 1.47 | 0.54 | 8.90 | 40.0% |
| 10 | 2 | 17 | **4.26** | 1.55 | 0.57 | 9.65 | 44.1% |
| 10 | 3 | 17 | **4.27** | 1.56 | 0.56 | 9.67 | 44.1% |
| 10 | 5 | 17 | **4.26** | 1.56 | 0.57 | 9.66 | 44.1% |
| 10 | 10 | 17 | **4.29** | 1.57 | 0.57 | 9.77 | 43.9% |

**Unionability scoring is the single largest cost — 4.47 ms, or 45.6% of the 9.81 ms mean wall
time.** That matters for the paper's efficiency argument: the dominant term is precisely the one
§4.1 pins to `α×k` rather than `|T|`, so bounding it is not a micro-optimization. It is also why
§6.2's Stage-1-skip ablation, which more than doubles the number of tables scored, is the right
experiment to measure the two-stage design against.

Note scoring time flattens from α=3 onward (5.00 → 5.01 → 5.00 ms) for the same reason `n_U` does:
the C5 clamp binds and the pool stops growing. The ILP and LP costs are essentially constant across
every configuration — Stage 2 is not the bottleneck at these pool sizes.

### 3.2 Stage 1 diagnostics: `F_P`, `delta_R`, and Dinkelbach iterations

| k | α | `F_P` (pool) | `F_R` (result) | `delta_R` | Dinkelbach rows that ran | mean iters when it ran |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 0.474 | 0.484 | −0.184 | 30 / 48 | 1.10 |
| 5 | 2 | 0.432 | 0.443 | −0.143 | 14 / 48 | 1.57 |
| 5 | 3 | 0.422 | 0.438 | −0.138 | 2 / 48 | 1.50 |
| 5 | 5–10 | 0.421 | 0.436 | −0.136 | 0 / 48 | — |
| 10 | 1 | 0.476 | 0.476 | −0.176 | 14 / 48 | 1.57 |
| 10 | 2–10 | 0.454 | 0.454 | −0.154 | 0 / 48 | — |

Three things this exposes that the headline table hides:

- **`delta_R` is negative everywhere** — mean −0.151, range −0.700 to +0.107. Every configuration
  lands above `F*` on average, and **66 of 255** feasible rows sit further than `δ` above it — which
  is fine: `δ` only defines `τ = F* − δ`, the floor Stage 2 must clear, not a ceiling. `delta_R` is
  bounded below by that floor and unbounded above, by design (C3).
- **`F_P ≈ F_R`.** Stage 2 barely moves the proportion away from what Stage 1's pool already had
  (0.474 → 0.484, 0.454 → 0.454). Stage 2 is choosing *within* a pool whose distribution is already
  fixed, which is why the α = 1 pool — the least extreme one — lands closest to `F*` on this
  diagnostic.
- **Dinkelbach almost never runs, and converges instantly when it does.** Only **60 of 720** rows
  invoked it at all: for `α ≥ 2` the C5 clamp makes `P = D`, so Stage 1 has nothing to select and
  the solver is skipped entirely. When it does run it takes **1.33 iterations on average, 2 at
  most**. The paper's "converges in only a few iterations" claim holds — but on santos3 it is
  supported by 60 observations, not 720, and that caveat belongs in any write-up of it.

---

## 4. The LP pre-check

260 Stage-2 instances harvested from the §3 sweep, each replayed twice (pre-check on/off) = 520 rows.

| lp_precheck | mean ILP time | mean LP time | n |
|---|---:|---:|---:|
| off | **0.834 ms** | — | 260 |
| on | 1.304 ms | 0.448 ms | 260 |

The pre-check found only **5 of 260** instances LP-infeasible — unsurprising given §2, since almost
everything that reaches Stage 2 on santos3 is satisfiable — and cost 56% more wall time to do it.

Synthetic supplement (mean ILP time on *infeasible* instances, by pool size):

| pool size `n_P` | pre-check off | pre-check on |
|---:|---:|---:|
| 100 | 0.654 ms | **0.562 ms** |
| 400 | **0.848 ms** | 1.068 ms |
| 1000 | **1.199 ms** | 2.341 ms |
| 3000 | **2.399 ms** | 9.121 ms |

It pays only near `n_P ≈ 100` and loses badly as pools grow — HiGHS's own presolve detects these
infeasibilities faster than a separate `linprog` call can. **The LP pre-check is not a performance
feature at any pool size this pipeline produces** (santos3 pools are 4–7 candidates). Its value is
soundness framing — `LP-infeasible ⇒ ILP-infeasible`, and for this constraint shape the converse
holds too — so it can be left off without risking a wrong verdict.

---

## 5. The retrieval ablation

Semantic retriever `∈ {hnsw, exact_scan}` × overlap filter `∈ {inverted_index, null}`, 48 queries
each, at `k=5, α=2`.

| config | feasible | insufficient | stage2 infeas. | n_sem | n_ovl | n_pair | n_D |
|---|---:|---:|---:|---:|---:|---:|---:|
| hnsw + inverted_index | 17 | 31 | 0 | 99.1 | 78.7 | 8.958 | 7.625 |
| exact_scan + inverted_index | 17 | 31 | 0 | 99.1 | 78.7 | 8.979 | 7.646 |
| hnsw + null | 21 | 21 | 6 | 99.1 | 6068.0 | 50.750 | 10.188 |
| exact_scan + null | 21 | 21 | 6 | 99.1 | 6068.0 | 50.792 | 10.167 |

**Findings.**

1. **HNSW is near-free, but on santos3 it is no longer *exactly* free.** Outcomes are identical —
   same feasible/infeasible counts in both overlap conditions — but the candidate counts differ very
   slightly (`n_pair` 8.958 vs 8.979, `n_D` 7.625 vs 7.646, ≈0.3%). On the smaller santos datalake
   the two were byte-identical; at 931 tables the approximate index starts to miss a handful of
   near-ties. The §7.1 claim holds in outcome, with that caveat now measurable rather than assumed.
2. **The overlap filter is the aggressive stage, and here it costs real recall.** It cuts candidate
   pairs 6068 → 78.7 (−98.7%) and `|D|` 10.2 → 7.6. Unlike on the base benchmark, that pruning is
   *not* free: dropping the filter raises feasibility 17 → 21 queries. The extra candidates it
   suppresses on enriched data are genuinely usable ones — 4 queries' worth — bought back at the
   price of 6 new `stage2_infeasible` and a 77× larger pair set.

---

## 6. Two further santos3 measurements

### 6.1 Groundtruth precision/recall, per stage

`D`/`P`/`R` scored against `santos3_small_benchmark_groundtruth.csv` at α=2, **run at all three `k`**
so precision/recall can be read against result size rather than at a single point.

| Measure | k = 3 | k = 5 | k = 10 |
|---|---:|---:|---:|
| feasible | **42 / 48 (88%)** | 35 / 48 (73%) | 17 / 48 (35%) |
| `insufficient_candidates` | 6 | 12 | 31 |
| `stage2_infeasible` | **0** | 1 | **0** |
| **precision @ D** | **1.000** | **1.000** | **1.000** |
| **precision @ P** | **1.000** | **1.000** | **1.000** |
| **precision @ R** | **1.000** | **1.000** | **1.000** |
| recall @ D | 0.307 | 0.309 | 0.306 |
| recall @ P | 0.198 | 0.252 | 0.165 |
| recall @ R | 0.109 | **0.150** | 0.137 |
| **`ideal_recall(k)`** | **0.129** | **0.215** | **0.431** |
| mean / max wall time | 8.7 / 20.1 ms | 8.4 / 20.2 ms | 5.7 / 19.9 ms |
| funnel `n_sem → n_ovl → n_pair → n_D → n_P → n_R` | 99.1 → 78.7 → 9.0 → 7.6 → 4.8 → 2.6 | (same retrieval) | (same retrieval) |

`ideal_recall(k)` is starmie_fair's ceiling metric (`checkPrecisionRecall.py`):
`mean_i min(k, gt_size_i) / gt_size_i` over all 48 queries with a groundtruth entry (mean
`gt_size` 25.5, minimum 12 — so `min(k, gt_size) = k` for every query at every `k` here) — the best
recall any method could reach at that `k`, independent of ranking quality.

**Precision is 1.000 at every stage and every `k`** — every table DUTS retrieves, pools or selects is
independently considered genuinely unionable by the benchmark. That is now established across three
result sizes, not one.

**Recall @ D is flat at ~0.307** across all `k`, as it must be: retrieval doesn't depend on `k`. The
`k`-dependence appears only downstream, and **recall @ R peaks at k = 5 (0.150)** — at k=3 the result
is too small to cover much of the groundtruth, and at k=10 two thirds of queries fail on candidate
supply and contribute zero. Recall declining from D to R is expected, not a defect: DUTS solves a
strictly harder problem (find `k` *distribution-satisfying* tables) than the groundtruth measures
(exhaustive union coverage), and the groundtruth's join column and DUTS's `V_D` are usually different
columns. See `experiments/groundtruth.py`'s docstring for what this does and does not test.

**Recall @ R falls short of `ideal_recall(k)` here, and the gap grows with `k`** (0.109 vs 0.129 at
k=3; 0.150 vs 0.215 at k=5; 0.137 vs 0.431 at k=10) — the opposite of what a ranking problem would
predict. The cause is arithmetic, not ranking: this table averages over **all 48** queries, including
the ones that are infeasible (`R = ∅`, recall contributes 0) — 6/12/31 lose to
`insufficient_candidates` alone at k=3/5/10, before `stage2_infeasible` adds a handful more. Those
zeros pull the mean below the ceiling, and pull it further as `k` grows and more queries fail on
candidate supply (§3, finding 2). Restricted to only the queries where a feasible answer exists —
the *solvable-only* cohort `RESULTS.md` uses at α=2, k∈{3,5} — this gap **closes exactly**: recall @
R there equals `ideal_recall(k)` to four decimal places, because on that cohort precision is 1.000
and every query's groundtruth size exceeds `k`, forcing `recall = k/gt_size = ideal_recall`
identically. So the shortfall visible in this table is entirely a feasibility-rate effect, not
evidence DUTS is leaving recall on the table among the queries it actually answers.

Note the zeros in the `stage2_infeasible` row: at k=3 and k=10 on santos3, **no query fails for
distribution reasons at all** — every failure is a candidate shortfall.

### 6.2 The Stage-1-skip ablation

`two_stage` (normal pipeline) vs `skip_stage1` (score *every* candidate in `D`, hand all of `D` to
Stage 2 at cardinality `k`, bypassing Dinkelbach), at `k=10, α=1, top_n=500`.

| condition | n scored | scoring time | ΣU | F_R |
|---|---:|---:|---:|---:|
| `two_stage` | **10.0** | **7.8 ms** | 93.23 | **0.484** |
| `skip_stage1` | 22.0 (+120%) | 16.3 ms (+108%) | **99.21** | 0.419 |

1. **The efficiency claim, measured.** `two_stage`'s scoring cost is pinned at `α×k = 10` regardless
   of datalake size; `skip_stage1`'s tracks `|D|` and reaches 22.0 on santos3's 931-table lake. The
   paper's central efficiency claim as a real runtime delta rather than an asymptotic argument.
2. **Stage 1 also changes where `F_R` lands, though not through anything either stage is required to
   optimize.** `skip_stage1` always beats `two_stage` on ΣU — necessarily, since it optimizes over
   the superset `D ⊇ P`; 0 of 35 paired-feasible queries violate this monotonicity, asserted in the
   test suite. `two_stage` happens to land closer to `F* = 0.3` (0.484 vs 0.419 — both above it,
   since only the `F_R ≥ τ` floor is enforced): Stage 1's smaller, `argmax F`-selected pool leaves
   Stage 2 fewer options, which incidentally pulls `F_R` down toward `F*` here. Worth reporting as a
   side effect, not as evidence either configuration is doing better at a shared quality target —
   there isn't one for `F*` proximity.

---

## 7. Label and column glossary

Every label appearing above, and every column in `experiments/results/santos3_*.csv`.

### 7.0 Coverage map — every measured quantity, and where it is reported

So nothing measured is silently left out of the write-up:

| Quantity | Reported in |
|---|---|
| `feasible`, `infeasibility_cause` | §3 table, §5, §6.1 |
| `n_sem`, `n_ovl`, `n_pair`, `n_D` | §5 (all four conditions), §6.1 funnel |
| `n_P`, `n_R`, `n_unionability_computations`, `effective_alpha` | §3 table (`n_U`, `eff. α`), §6.1 funnel, §6.2 |
| `F_P` | **§3.2** |
| `F_R` | §3 table, §3.2, §6.2 |
| `delta_R` | **§3.2** |
| `sum_U` | §3 table, §6.2 |
| `dinkelbach_iterations_stage1` | **§3.2** |
| `scoring_time_s` | **§3.1** (and §6.2, where it is the headline) |
| `ilp_time_s` | §3 table, §3.1 |
| `lp_time_s`, `lp_precheck_ran`, `lp_feasible` | §4, **§3.1** |
| `wall_time_s` | §3 table, §3.1, §6.1 |
| `gt_size`, `d_hits`/`p_hits`/`r_hits`, `precision_*`, `recall_*` | §6.1 (all three `k`) |
| `F_min`, `F_max`, `F_closest`, `best_gap`, `tau_reachable`, `fstar_bracketed`, `fstar_within_delta`, `N_Q`/`n_Q`/`F_Q`, `n_subsets` | §2 |
| `seed`, `config_id`, `experiment`, `adapter_*`, `top_n`, `include_query`, `M_size`, `error` | §1.3 (constant across this report; `error` is empty on every row) |

### 7.1 Outcome labels

| Label | Meaning |
|---|---|
| **`feasible`** | Stage 2 returned exactly `k` tables satisfying `F_R ≥ τ`. Recorded as `feasible=True`, `infeasibility_cause=None`. |
| **`insufficient_candidates`** | `\|D\| < k` — retrieval returned fewer candidates than the requested result size. Checked **before** Stage 1; the distribution constraint is never consulted. A retrieval-width property, not an optimization failure. On santos3 this is essentially the *only* failure mode. |
| **`stage2_infeasible`** | `\|D\| ≥ k`, but Stage 2's ILP proved no `k`-subset of the pool `P` reaches `τ`. Since the C4 certificate's removal this is the only post-retrieval failure label: it covers both "no `k`-subset of `D` could ever reach `τ`" and "Stage 1's pool happened not to contain a feasible subset", with no attempt to tell them apart. |
| **`error`** | An exception escaped the runner for this query. Distinct from any infeasibility; the `error` column carries the message. |
| ~~`intrinsic`~~ | **Removed 2026-08-10** with the C4 certificate. Formerly: no `k`-subset of `D` can reach the floor. Such rows now report `stage2_infeasible`. |
| ~~`alpha_induced`~~ | **Removed 2026-08-10.** Formerly: the certificate passed but Stage 2 still failed on `P`. Never observed on any santos variant. |

### 7.2 Query parameters

| Column | Meaning |
|---|---|
| `q_table` | Query table filename. On santos3 always a `*_fair.csv` rebalanced variant. |
| `attr` | `V_D`, the distribution ("protected"/"sensitive") attribute — a **0-based column index** into `q_table`, from `protected_attribute_id`. |
| `M_size` | `\|M\|`, the number of distribution values. Always 1 here: `M = {protected_value}`. |
| `k` | Requested result size. `\|R\| = k` exactly, or `R = ∅`. |
| `alpha` | Pool expansion factor; Stage 1 selects `α×k` tables. `α ≥ 1`. |
| `F_star` | `F*`, the target proportion of `M`-valued tuples in the union. 0.3 throughout. |
| `delta` | `δ`, the maximum allowed deviation **below** `F*`. 0.15 throughout. |
| `τ` (derived) | `τ = F* − δ = 0.15` — the actual floor Stage 2 enforces. `F*` and `δ` never enter the ILP separately. |
| `include_query` | Whether `Q` counts in both objectives (C1/M1). `True` throughout. Decisive on santos3 — see §2. |
| `top_n` | §7.1 ANN probe width — how many nearest columns the semantic retriever returns. |
| `seed`, `config_id`, `experiment` | Run identity: RNG seed, condition label, driver name. |
| `lp_precheck` | Whether Stage 2 ran its LP relaxation before the ILP. |
| `adapter_*` | Which implementation filled each `Protocol` seam: `synopsis`, `semantic`, `overlap`, `unionability`. |

### 7.3 Funnel counts

| Column | Meaning |
|---|---|
| `n_sem` | Column-level hits from the semantic retriever (§7.1 HNSW probe). Capped by `top_n`. |
| `n_ovl` | Column-level hits from the value-overlap filter (§7.2 posting lists). |
| `n_pair` | `(table, column)` pairs surviving the **intersection** of the two filters (§7.3). |
| `n_D` | `\|D\|` — candidate **tables** after reducing pairs to one best column per table. What Stage 1 sees. |
| `n_P` | `\|P\|` — Stage 1's pool, `= min(α×k, \|D\|)` by C5. `0` when the run short-circuited at `\|D\| < k`. |
| `n_R` | `\|R\|` — the final result. `k` when feasible, `0` otherwise. |
| `gt_size`, `d_hits`, `p_hits`, `r_hits` | groundtruth-eval only: the groundtruth set size, and how many of `D`/`P`/`R` fall in it. Precision/recall derive from these. |

Always `R ⊆ P ⊆ D`.

### 7.4 Quality and cost

| Column | Meaning |
|---|---|
| `F_P` | Proportion `F` achieved by Stage 1's pool — Stage 1's own objective value. |
| `F_R` | Proportion `F` achieved by the final result `R`. Feasibility requires `F_R ≥ τ`. |
| `delta_R` | Signed `F* − F_R`. Negative means `R` **overshot** the target — the normal case on santos3. **Not bounded by `δ`**: only the floor is enforced, and C3's unbounded `argmax F` overshoots freely. A code comment asserting the bound was corrected on 2026-08-10. |
| `sum_U` | `ΣU` over `R` — Stage 2's objective value, total unionability. Not comparable across different `k`. |
| `n_unionability_computations` | **The headline efficiency metric.** How many `U` values were computed = `\|P\|`, never `\|D\|`. What makes "cost depends on `α×k`, not `\|T\|`" a countable claim. |
| `effective_alpha` | `\|P\|/k` as realized. Falls below the requested `alpha` whenever C5 clamps the pool to `D` — the direct evidence that α is inert on santos3. |
| `dinkelbach_iterations_stage1` | Dinkelbach iterations to converge in Stage 1. |
| `scoring_time_s` | Wall time computing `U` over `P` (the bipartite matchings). |
| `ilp_time_s` | Wall time in Stage 2's `milp` call. |
| `lp_precheck_ran`, `lp_feasible`, `lp_time_s` | Whether the LP relaxation ran, its verdict, and its cost. |
| `wall_time_s` | Total per-query wall time, retrieval through Stage 2. |

### 7.5 Reachability columns (§2 only, `santos3_fraction_reachability_k*.csv`)

| Column | Meaning |
|---|---|
| `n_subsets` | How many `k`-subsets of `D` were enumerated = `C(\|D\|, k)`. Exhaustive, not sampled. |
| `N_Q`, `n_Q`, `F_Q` | The query table's own matching-tuple count, row count, and proportion `N_Q/n_Q`. `F_Q` is the anchor `include_query=True` pins every `F(S)` toward. |
| `F_min`, `F_max` | The minimum and maximum `F` attainable by any `k`-subset — the exact attainable range. |
| `F_closest`, `best_gap` | The attainable value nearest `F*`, and `\|F_closest − F*\|`. `best_gap` is the closest *any* selection rule could land to `F*` given this `D` — a diagnostic bound against a hypothetical proximity-seeking objective, not a measure of defect in `max_f` (which targets `F ≥ τ`, not proximity). This is what makes the retrieval-vs-objective split in §2 possible. |
| `tau_reachable` | `F_max ≥ τ` — does a feasible answer exist at all? |
| `fstar_bracketed` | `F_min ≤ F* ≤ F_max` — is the target inside the attainable range? |
| `fstar_within_delta` | `best_gap ≤ δ` — can some `k`-subset land within `δ` of the target? |
| `skipped` | Empty when analysed; otherwise why not (`insufficient_candidates`, `too_many_subsets`, `no_valid_subset`). |

---

## 8. Reproducibility caveat

Every sweep is seeded and config-driven, and reruns reproduce all quality and outcome columns
exactly. **One exception, pre-existing:** `n_D` varies run-to-run for a small number of borderline
queries, because hnswlib's multi-threaded index construction does not fix neighbour order for
near-tied candidates. Measured on the base benchmark, two identical reruns differed on `n_D` for
1 of 48 queries, with feasibility verdicts identical. On santos3 the same mechanism explains the
≈0.3% `n_D` difference between `hnsw` and `exact_scan` in §5. Timing columns naturally vary run to
run. The §2 reachability analysis is fully deterministic given `D`.

**Result files** (all prefixed `santos3_`): `santos3_alpha_sweep`, `santos3_lp_precheck`,
`lp_precheck_synthetic`, `santos3_retrieval_ablation`, `santos3_groundtruth_eval_k{3,5,10}`,
`santos3_stage1_skip_ablation`, `santos3_fraction_reachability_k{3,5,10}` — CSV plus parquet twins
in `experiments/results/`.
