# DUTS Implementation Plan — §4 framework, with §5–§8 folded in

Target paper: *Distribution-Aware Unionable Table Search* (Kassaie & Miller).

**Spine:** Section 4's two-stage framework. **Filled in from:** §5 (Dinkelbach),
§6 (ILP + constrained alignment), §7 (indexes), §8 (swap alternative).
§4 is an overview that delegates every mechanism to a later section, so the
later sections *are* the implementation; this plan treats §4 as the assembly
contract and pulls the concrete details from where they actually live.

Repo: `/u6/bkassaie/DUTS` (new, empty). Nothing here writes to
`~/starmie_fair`; it is consumed read-only through one adapter.

---

## 1. What the paper asks for, mapped to work

| Paper | Mechanism | Status |
|---|---|---|
| §4.1 | Two-stage driver: distribution-first pool, then unionability | **Build** (Phase 7) |
| §4.2, §7.3 | `D_pair = D_sem ∩ D_ovl` at attribute level, per-table argmax by `sim` | **Build** (Phase 3) |
| §5 | Exact-(α×k) 0–1 fractional program via Dinkelbach | **Build** (Phase 4) |
| §6.1 | 0–1 ILP with LP-relaxation pre-check | **Build** (Phase 6) |
| §6.2 | Constrained attribute alignment, `V_D` pair pinned | **Adapt** existing (Phase 5) |
| §7.1 | Attribute embeddings + HNSW, top-`N` then `θ_dis` cutoff | **Adapt** existing (Phase 2) |
| §7.2 | Value-overlap filter (JOSIE), retain overlap ≥ 1 | **Replace** — see I1 (Phase 2) |
| §8, §8.1 | Locality of violations, `Greedy_swap` | **Verify + fix** existing (Phase 8) |
| §8.2 | Winnow / preference filtering, NL + BNL | **Verify + fix** existing (Phase 8) |
| §6.1, §8 | The experiments behind the paper's `Section ??` refs | **Build** (Phase 9) |

---

## 2. Existing assets — reuse, do not rebuild

`~/starmie_fair` already implements much of §6.2, §7.1 and nearly all of §8.

| Need | Asset | Note |
|---|---|---|
| `H_i^A`, `n_i` synopses (§2) | `TableMetadata.py:12`, `:173 MetadataStore` | Stores per-attribute `Dict[value,count]` as **integers** → exact `N_i`, no float drift. `compute_union_distribution:358`. |
| `θ_cat` (§2) | `experimental_config.json` `domain_size_threshold: 50` | Matches §2's ndv threshold. |
| `F`, `Δ`, per-table `N_i`/`n_i` (§3) | `HNSWSearcher_Fair.py:627 compute_F(..., include_query=True)`, `:909 compute_Delta` | `include_query` already resolves M1 in code. Scalar value only → M2. |
| Constrained alignment (§6.2) | `bounds.py:53 verify_constrained(..., mandatory_idx)` | Pins the row only → M3. Also `get_edges:152`, `upper_bound_bm:119`, `lower_bound_bm:137`, `cosine_sim:10`. |
| HNSW probe (§7.1) | `HNSWSearcher_Fair.py:503 _find_candidates(query_cols, N)` — **API reference only** | See correction below. |

⚠️ **Corrected 2026-08-09.** This row previously read "`hnsw_fair/` built hnswlib
fork | Ready." Both halves were wrong: **`hnsw_fair/` does not exist** on disk,
and no fork is needed. The env has **vanilla upstream `hnswlib` 0.8.0**
(`github.com/yurymalkov/hnsw`), which is what `dutsx` uses — decided
2026-08-09. `HNSWSearcher_Fair` is read as an API-usage reference for the four
hnswlib calls (`Index(space=...)`, `init_index`, `add_items`, `knn_query`) and
is **not** imported, subclassed, or wrapped; it carries unrelated
fairness/search machinery that `dutsx/` must not depend on. Note also that
`hnswlib`'s `space='cosine'` normalizes internally and returns cosine distance
directly, so I3's `θ_dis = sqrt(2 − 2σ)` conversion is only needed when
indexing with `space='l2'` — it is a justification that the two metrics agree,
not a required build step.
| Column embeddings | `data/<bench>/vectors/cl_{datalake,query}_drop_col_tfidf_entity_column_0.pkl` | Starmie contextualized column vectors → see I2. |
| Prebuilt synopses | `data/<bench>/indexes/metadata_{combined,datalake,query}.pkl` | Verify freshness (Phase 0). |
| §8 `Greedy_swap` | `nl_swap.py:116`, `bnl_swap.py:23`, `exhaustive_swap.py:20`, `HNSWSearcher_Fair.py:1093 exhustive_swap` | Complete; needs conformance check vs Claims 1–4. |
| §8.2 winnow | `preference.py:13 PreferenceConfig.dominates`, `preference_config.json`, `BNL_PREFERENCE_COMPATIBILITY.md` | Encodes Claim 3; one rule under-prunes → S3. |
| Benchmarks | `santos`, `santos2..4`, `santosLarge`, `table-union-search-benchmark/{small,large}`, `protected_attributes_*.csv` | Ready. |

### Environment

Use **`TableUnionNew`** (`/u6/bkassaie/.conda/envs/TableUnionNew`): Python 3.8.5,
`hnswlib` ✓, `munkres` ✓, `networkx` ✓, `scipy 1.10.1` → `scipy.optimize.milp`
(HiGHS branch-and-bound) ✓. **Zero new dependencies for §4–§8.**

Rejected: `tableunion` (py3.7.10, scipy 1.3.3 — no `milp`, no `hnswlib`);
`base` (scipy 1.13.1 but no `hnswlib`).

⇒ **Target Python 3.8 syntax**: `typing.List/Dict/Tuple`, no builtin generics,
no `match`.

---

## 3. Formulation corrections (M) — settle before coding

### M1 — Does `Q` enter the stage objectives? *(highest impact)*
§3 and §4.1 both write `F(T(S ∪ Q))`. §5's Dinkelbach and §6.1's ILP both drop
`Q`. **Algorithm 1 uses both, in the same algorithm** (line 5 tests
`F(T(Q∪R))`, line 8 tests `F(T(R))`). These are different problems.

**Decision:** include `Q` everywhere, flag `include_query=True` (faithful to §3;
`compute_F` already supports it). Corrected derivations — both stay in the same
complexity class:

**Stage 1.** Maximize `F(T(S∪Q)) = (N_Q + Σ_{i∈S} N_i)/(n_Q + Σ_{i∈S} n_i)`,
`|S| = α×k`:

```
Φ(λ) = max_{|S|=αk} [ (N_Q − λ·n_Q) + Σ_{i∈S} (N_i − λ·n_i) ]
```

The `Q` term is **constant in `S`** ⇒ the inner maximizer is unchanged: still one
top-(α×k) selection on `y_i(λ) = N_i − λ·n_i`. `Q` enters only

1. the value of `Φ(λ)` (hence the termination test), and
2. the update `λ^(t+1) = (N_Q + Σ_{i∈S^(t)} N_i)/(n_Q + Σ_{i∈S^(t)} n_i)`.

So §5's algorithm is right *provided* step (3) uses the `Q`-inclusive ratio. As
printed it does not, and therefore solves the `Q`-free problem.

**Stage 2.** With `τ = F* − δ` and `y_Q(τ) := N_Q − τ·n_Q`:

```
(N_Q + Σ N_i x_i)/(n_Q + Σ n_i x_i) ≥ τ
  ⟺  Σ_i y_i(τ)·x_i  ≥  −y_Q(τ)
```

RHS is `−y_Q(τ)`, not `0`. §6.1's `≥ 0` is the `Q`-excluded special case.

### M2 — `M` is a value *set*; `compute_F` takes one value
§2/§3 define `M ⊂ Dom(V_D)` and `N(T,V_D,M) = Σ_{c∈M} count(c)`.
`compute_F` takes a scalar `protected_value`.

**Decision:** new `duts/stats.py` sums over `M` in integer arithmetic straight
from `MetadataStore`. Do not modify `starmie_fair`. Regression test: agreement
with `compute_F` when `|M| = 1`.

### M3 — §6.2 pins the *pair*; `verify_constrained` pins only the row
`bounds.py:53` forces row `mandatory_idx` to be matched but lets Hungarian pick
**which column**. §6.2 says `(V_D, V_{D_T})` is *already determined during
candidate generation* and *fixed and excluded from the optimization* — the column
is `V_{D_T}`, chosen by §7.3's argmax, not by the matcher.

**Decision:** `verify_pinned(t1, t2, threshold, pin=(qi,tj))` — record
`s(V_D,V_{D_T})`, delete row `qi` and column `tj`, match the remainder,
return `s(pin) + Σ_{(x,y)∈M*} s(x,y)`. Faithful to §6.2 **and cheaper** than the
big-bonus trick. Test that the two can disagree.

Also `bounds.py` uses `sim > threshold`; §6.2 says "at least the threshold σ"
(`≥`). Keep `>` for comparability with Starmie's published numbers and say so.

### M4 — `U_{V_D}` range contradicts Table 1
§6.2 defines `U_{V_D}` as a **sum** of similarities; §2/Table 1 declare
`U ∈ [0,1]`. Both cannot hold.

**Decision:** compute the raw sum (Starmie-comparable); expose
`normalize ∈ {none, by_query_attrs, by_matched_pairs}`, default `none`. Stage 2's
argmax is invariant to positive scaling, so this affects reporting only. Fix
Table 1 or the §6.2 formula in the paper.

**Resolved (2026-08-09): no normalization.** Keep the raw sum (`normalize="none"`);
do not change the code to force `U ∈ [0,1]`. Table 1 is what needs revising in the
paper, not §6.2's formula.

### M5 — Stage 1 maximizes `F` without bound
`Δ = F* − F` is signed and overshoot is admissible, so `argmax F` fills the pool
with the most extreme tables (all-`M` tables). For `F* = 0.5` the pool is
distributionally degenerate, the Stage-2 constraint is slack for *every* subset,
and Stage 2 collapses to plain top-`k` unionability inside a biased pool — which
can cost unionability against a pool that merely *satisfies* `τ`. This directly
inflates §8's trade-off (1).

**Decision:** ship §4.1 faithfully as default (`stage1.objective="max_f"`); add
`stage1.objective="satisfice"` (prefer pool diversity subject to `F ≥ τ`) as a
measured ablation. Report both.

**Resolved (2026-08-09): unbounded `argmax F` is the intended Stage 1**, not a
placeholder standing in for an unsettled "satisficing" alternative. This is no
longer an ambiguity about which design is *correct* — `objective="max_f"`
(already built in `duts/stage1_dinkelbach.py`, per PLAN.md C3) is it.
`objective="satisfice"` stays unimplemented (`NotImplementedError` in
`duts/config.py`); if it's ever built, it's purely an experimental ablation for
Phase 9, not a competing default.

### M6 — Free exact infeasibility certificate *(recommend adding to the paper)*
§3 returns `∅` when infeasible; §8 trade-off (3) mentions empty results. Stage 1
already gives an exact global test.

⚠️ **Correction applied 2026-08-09 (mirrors PLAN.md C4): the claim originally
given below is false under M1's resolution (`include_query=True`).** `F_S` is
the `n_i`-weighted average of `{F_i}_{i∈S}` *only in the Q-free case* — with `Q`
forced into the average at every cardinality, a low-`F_Q`, large-`n_Q` query is
pulled *up* by adding more tables, so a bigger set can beat every smaller one.
Counterexample (brute-forced, kept as `tests/test_stage1.py::test_c4_counterexample_regression`
in the core repo): `(N_Q,n_Q)=(0,1000)`, `t1=(N=5,n=10)`, `t2=(N=4,n=10)`,
`k=1`, `α=2` → `max_{|S|=1} F = 0.00495 < max_{|S|=2} F = 0.00882`.
**The certificate below does not need this claim** — it is sound directly from
Problem (1)'s `|R|=k, R⊆D` constraint: `F_max^k` is exactly the best proportion
any feasible solution can reach, by definition, regardless of how it compares
to the `α×k`-cardinality maximum. Do not restate the claim in the paper; state
the certificate on the constraint alone.

So running **the same Dinkelbach solver at cardinality `k` over all of `D`**
gives `F_max^k = max_{|S|=k} F(T(S∪Q))` exactly, and

- `F_max^k < τ` ⇒ Problem (1) is **infeasible over all of `D`** → return `∅`, no ILP;
- `F_max^k ≥ τ` but Stage 2 on `P` infeasible ⇒ infeasibility is **α-induced**.

Cost: one extra `O(I·m·log m)` run. It separates two failure modes §8 currently
conflates and yields a headline metric: *pool-induced infeasibility vs. α*.

**Status (2026-08-09): still open** — not yet answered in the round of
decisions that resolved M1/M4/M5/I1/I2/S4/benchmark below. The certificate
itself is built and tested regardless (`certify_feasible` in
`duts/stage1_dinkelbach.py`), and is sound independent of this decision — see
the correction just below and PLAN.md C4.

---

## 4. Indexing corrections (I)

### I1 — JOSIE is absent, and unnecessary
JOSIE is not installed anywhere on this machine. But §4.2 and §7.2 both use its
output as a **boolean**: retain overlap `≥ 1`. JOSIE's cost-based top-`k`
machinery is wasted on a threshold-1 predicate, and §7.3 explicitly declines to
use overlap size as a ranking signal ("overlap is used throughout only as a
hard, boolean necessary condition").

**Decision:** exact inverted index `value → {(table, attr)}` built once from
`MetadataStore`; probe = `⋃_{c∈M} postings[c]`. Exact, simpler, and faster than
JOSIE here. Keep the interface JOSIE-shaped (also return overlap size) so real
JOSIE can drop in if a reviewer insists. State the substitution in the paper's
implementation details.

**Resolved (2026-08-09): real JOSIE will be provided.** ~~The interface stays
JOSIE-shaped as planned above, but treat that as the *drop-in point*, not the
final answer — once JOSIE is supplied, wire it in as the primary retrieval
path so the paper can name it directly and be comparable to [13], rather than
disclosing a substitution. Keep the exact inverted index as the tested
fallback (it's still correct and cheap for the boolean overlap check this
stage actually needs). **Waiting on:** how JOSIE will be delivered (installable
package, prebuilt index, or a repo to build against) and its query interface —
needed before Phase 2 can wire it in.~~

⚠️ **Superseded (2026-08-09, same day): the paper itself dropped JOSIE.**
`Fair_Table_Search_7.pdf` rewrites §7.2 from "Value-overlap filtering via
JOSIE" to **"Value-overlap filtering via posting lists"** — an exact inverted
index, framed explicitly as the "database system" query paradigm (return
every matching record) against the "search-engine-style" paradigm JOSIE
represents, and citing it that way: **`[13]` is no longer JOSIE** — it is now
Zobel & Moffat, *Inverted files for text search engines* (2006), cited as the
contrast case the design deliberately is *not*. This is not a delay of JOSIE;
it is the paper's authors formalizing the inverted-index substitution as the
permanent design. The "real JOSIE will be provided" answer above is now
contradicted by the source paper itself — flagged to the user rather than
silently overridden; if JOSIE is still wanted for some other reason, that is
now a deviation from the paper, not an implementation of it.

**`InvertedIndexOverlap` (`dutsx/adapters/overlap.py`) already conforms to
the new spec exactly, unmodified** — checked point by point:
- Retention rule `|M ∩ V_c| ≥ 1`, Boolean, no ranking: matches.
- `θ_cat` bounds only how many posting lists **one attribute** appears in (≤
  `θ_cat`, one per distinct value it holds) — does **not** bound posting-list
  **length**: matches (nothing in `InvertedIndexOverlap.__init__` caps
  `self._postings[value]`'s size).
- §7.3's `argmax sim` tie-break never reads the overlap count — confirmed at
  `dutsx/retrieval.py`'s `if current is None or sim > current.sim`, exactly
  the new §7.3 language: "value overlap ... carries no score that could
  otherwise be used to choose among several qualifying attributes."

**One genuine gap, not blocking:** the new §7.2 recommends representing each
posting list as a **Roaring bitmap** (CRoaring / PyRoaring) for implementation
efficiency at scale. `InvertedIndexOverlap` uses plain Python `dict`/`set`.
At santos scale (550–931 tables, thousands of columns) this is very unlikely
to matter for correctness or measured performance; worth reconsidering only if
targeting a much larger benchmark (`santosLarge`, TUS-large).

`JosieOverlap` should **not** be built. `NullOverlap` (the ablation double) is
unaffected — it never depended on JOSIE.

### I2 — Which embedding space? *(architectural)*
§7.1 says attributes are embedded "using a pretrained encoder over its values",
but §6.2 scores alignment with Starmie's contextualized column embeddings at
`σ = 0.6`. Two different spaces would make §7.3's tie-break incommensurable with
§6.2's matching, and would need two artifacts maintained in step.

**Decision:** build the HNSW index over **the Starmie column embeddings that
already exist** (`vectors/cl_*_drop_col_tfidf_entity_column_0.pkl`). One
embedding space, two uses: ANN retrieval (§7.1) and bipartite matching (§6.2).
`σ` stays calibrated.

**Resolved (2026-08-09): confirmed, and already in hand.** The Starmie column
embeddings are already computed — verified present for `santos` at
`/u6/bkassaie/starmie_fair/data/santos/vectors/cl_{datalake,query}_drop_col_tfidf_entity_column_0.pkl`.
No new encoder needs building; Phase 2 indexes these directly.

§7.1's wording should become "the same contextualized
column embeddings used for unionability scoring (§6.2)".

### I3 — Euclidean vs cosine
📄 **See `DISTANCE-METRICS.md` — that is now the authoritative note on this**
(the problem, the math, the empirical verification, what ships, and
paper-ready wording). Summary retained here for continuity:

§7.1 filters on **Euclidean** distance `< θ_dis`; §6.2 and `bounds.py:10` use
**cosine** similarity `> σ`; Table 1 defines `sim(A,V_D)` as "as computed by the
HNSW index". Under I2 these must be reconciled.

**Decision:** for L2-normalized vectors `‖a−b‖² = 2 − 2·cos(a,b)`, so the two
are monotonically equivalent: `θ_dis = sqrt(2 − 2σ)`, and §7.3's per-table
`argmax sim` is identical under either metric. **Without normalization the two
signals genuinely disagree and §7.3's tie-break is ill-defined** — worth one
sentence in §7.1.

⚠️ **Corrected 2026-08-09.** This section originally said "L2-normalize every
attribute embedding at index time" **and** "build hnswlib with
`space='cosine'`". Those are two different code paths and the plan asserted
both. What ships is `space='cosine'` over the **raw, unnormalized** vectors —
hnswlib's cosine space normalizes internally and returns `1 − cos_sim`
directly, so no explicit normalization and no `θ_dis` conversion happens in the
adapter. The `θ_dis` identity is still verified independently (two tests, see
`DISTANCE-METRICS.md` §3) because the *paper* needs it, not because the code
does.

### I4 — Query-time parameters constrain the index
§4.2 closes with "`M` and `δ` are specified at query time and are therefore
unavailable during index construction." Consequences to honour:
- the inverted index is keyed by individual value, never by `M`;
- `N_i` **cannot** be precomputed — it is `Σ_{c∈M} count_i[c]` at query time;
- the synopsis must retain counts for *all* values of the attribute (≤ `θ_cat`
  = 50 entries, so cheap).

### I5 — Attribute-level indexing and table exclusion
§2 and §4.2: index per categorical attribute, not per table; a table with `r`
categorical attributes contributes `r` synopses; tables with **no** attribute
satisfying `ndv ≤ θ_cat` are excluded from the index entirely. This is correct
per §3 (no `V_D` match ⇒ cannot contribute), but those tables may be highly
unionable — so **log the exclusion count and their Starmie unionability** to
quantify §8's trade-off (1). Drop `n_i = 0` attributes at build.

---

## 5. §8 corrections (S) — the existing implementation is close

### S1 — Algorithm 1 returns `∅` for an already-feasible input
Line 1 is a guard (`if Δ(F*, F(T(Q∪R))) > δ then …`) wrapping the body. If `R`
is already feasible the guard fails, control falls to line 10, and the algorithm
returns `∅` — reporting failure on an input that needs no repair. Must return `R`.

### S2 — `Q` inconsistency inside Algorithm 1
Line 5 compares `F(T(Q∪R))`; line 8 tests `Δ(F*, F(T(R)))`. Pick `Q`-inclusive
throughout (M1). Also, the prose says the strict-improvement test is "Line 7";
it is line 5 (line 7 is `C ← C\{b}`).

### S3 — The winnow filter under-prunes relative to Claim 3
Claim 3's four conditions, combined with Eq (3), license exactly:

| Claim 3 | `N` | `n−N` | required `U` | `preference_config.json` | verdict |
|---|---|---|---|---|---|
| cond 1 | `>` | `<` | `≥` | Rule 1 uses `≥` | ✓ |
| cond 2 | `>` | `=` | `≥` | **Rule 5 uses `>`** | **too strict** |
| cond 3 | `=` | `<` | `≥` | Rule 4 uses `≥` | ✓ |
| cond 4 | `=` | `=` | `>` | Rule 3 uses `>` | ✓ |

Conditions 1–3 all yield **strictly greater** `F`, so Eq (3)'s first disjunct
applies and `U(c1) ≥ U(c2)` suffices. Rule 5 demanding `U(c1) > U(c2)` fires
less often ⇒ `C'` is larger than necessary. **Sound but slower** — a missed
pruning opportunity, not a correctness bug. Fix to `≥` and re-measure §8.2's
savings.

Separately: `BNL_PREFERENCE_COMPATIBILITY.md` documents that a "Rule 2"
(all-three-equal) breaks irreflexivity and asymmetry, hence BNL's strict-partial-
order requirement (paper's Claim 4). **Rule 2 is already absent from the live
config** — the fix landed. Keep a load-time assertion that no configured rule is
reflexive, so it cannot regress.

### S4 — The author's two open items (page 9–10 margin note)

**Resolved (2026-08-09): descoped — not part of this work.** Neither the
incompleteness witness nor the sufficient-success condition below will be
produced. Left in place as reference in case reconsidered later, not as a plan
of record.

The note defers (1) a concrete example that `Greedy_swap` is **incomplete** —
fails though a feasible repair exists — and (2) conditions under which it is
guaranteed to succeed. Both are tractable here:

- **(1) is a search problem, not a proof problem.** `exhaustive_swap.py`
  already computes optimal repairs. Enumerate small random instances
  (`|R_vi| ≤ 4`, `|C| ≤ 6`), keep those where exhaustive finds a feasible repair
  and `Greedy_swap` returns `∅`, and minimize the witness. Deliverable: a
  4-row table in §8.1, machine-checked.
- **(2)** proposed line of attack: greedy commits only strict-`F` improvements,
  so `F` increases monotonically; failure requires greedy to *consume* a
  candidate that a feasible repair needs. A sufficient condition should follow
  from an exchange argument — e.g. if every `c ∈ C` with `N_c/n_c ≥ τ*` remains
  available when its violating table is processed. State it as a sufficient (not
  necessary) condition; verify by brute force on the same generator.

### S5 — Fair comparison between §4 and §8 requires controlling retrieval
§8 "operates on the relatively large candidate set returned by Starmie" —
unionability-first. §4 is distribution-first. §4.1 promises the experimental
comparison. To make it meaningful, both must see the **same universe of tables**;
otherwise the comparison measures retrieval, not strategy. Phase 9 pins a shared
table universe and reports the metric §4.1 actually argues about: **number of
query-time unionability computations**.

### E1 — Editorial fixes while implementing
- §6 opening: "`|R| = k < wk`", "removing `wk − k` tables" — `wk` should be `α×k` (twice).
- §6 objective mixes `U(Q,T_i)` and `U_{V_D}(Q,T_i)`; must be `U_{V_D}` throughout (Table 1).
- §5.0.1: stray "since Unlike a linear objective".
- §5 "The governing theorem" cites `[?]`.
- §6.1 "Section ??" for LP-relaxation savings → Phase 9 produces it.
- §4.2 attributes the boolean-overlap justification to "§4.1"; it is in §4.2/§7.3.
- §7.1/§7.2 call JOSIE's check downstream of HNSW, but §7.3 intersects the two
  *independently retrieved* sets — the prose implies a pipeline, the algorithm is
  a parallel intersection. Align the wording.

---

## 6. Module layout

```
DUTS/
├── PLAN.md  NOTES.md  configs/duts.json
├── duts/
│   ├── types.py              # QuerySpec, CandidateStats, StageResult, Telemetry
│   ├── stats.py              # M2/M1: N_i,n_i over set M; F, Δ; (N_Q,n_Q) offset — integers
│   ├── index/
│   │   ├── build.py          # §7 I2/I4/I5: attr-level synopses + embeddings, θ_cat, exclusion log
│   │   ├── semantic.py       # §7.1 I3: L2-normalized HNSW, top-N + θ_dis
│   │   └── overlap.py        # §7.2 I1: exact inverted index, overlap ≥ 1
│   ├── retrieval.py          # §4.2/§7.3: D_pair = D_sem ∩ D_ovl, per-table argmax sim
│   ├── stage1_dinkelbach.py  # §5 + M1 offset, M6 certificate, D1 termination
│   ├── unionability.py       # §6.2 M3 verify_pinned, M4 normalization
│   ├── stage2_ilp.py         # §6.1 milp + LP pre-check, M1 RHS = −y_Q(τ)
│   └── pipeline.py           # §4.1 driver + telemetry
├── baselines/
│   ├── greedy_swap.py        # §8.1 Algorithm 1 (S1/S2 fixed), adapter over starmie_fair
│   └── winnow.py             # §8.2 Algorithms 2 & 3 (S3 fixed), NL + BNL
├── scripts/{build_index,run_duts,run_experiments}.py
└── tests/   # oracle.py + per-module suites
```

**The interface everything flows through** (`types.py`):

```python
class CandidateStats(NamedTuple):
    table: str      # table id
    attr: int       # the ONE attribute aligned to V_D (§7.3 argmax)
    N: int          # |{t : t.attr ∈ M}| — exact, integer
    n: int          # |T_i|
    sim: float      # cosine sim to V_D (§7.3 tie-break; §6.2 pin weight)
    overlap: int    # |values(attr) ∩ M| — reported, never scored (§7.3)
```

Both stages, and §8's swap, consume only `List[CandidateStats]` + `(N_Q, n_Q)`.
That makes them pure functions — cheap to test against brute force, and
independent of HNSW/JOSIE/Starmie.

### Dinkelbach termination (D1)
`Φ` is strictly decreasing and convex, so the fixed-point iteration is monotone.
Terminate on `|λ^(t+1) − λ^(t)| ≤ 1e-12` **or** `S^(t+1) == S^(t)`; cap at 100
and raise. Break `y_i(λ)` ties by `(−n_i, table_id)` — a deterministic tie rule
is *required* for the set-equality test to mean anything. Selection by sort
(`O(m log m)`, matching §4.1's stated bound); `np.argpartition` would give
`O(m)` but complicates tie determinism. Log iteration counts — §4.1's "converges
in only a few iterations" is an empirical claim this code should substantiate.

---

## 7. Phases

Each ends with tests green and one line in `NOTES.md`.

**Phase 0 — Scaffold + adapters.** `configs/duts.json` (benchmark paths,
`theta_cat: 50`, `sigma: 0.6`, `theta_dis` derived per I3, `k`, `alpha`,
`F_star`, `delta`, `include_query: true`, `seed: 42`, pinned package versions
asserted at startup). Adapter loads `metadata_combined.pkl` + column vectors.
*Accept:* on `santos`, report #tables, #categorical attrs at `θ_cat`, #tables
excluded per I5; verify sampled `n_i` against CSV row counts (stale-pickle check).

**Phase 1 — `stats.py`** (M1, M2). *Accept:* matches `compute_F` when `|M|=1`;
reproduces §3's aggregation-sensitivity property (`F_S` is not a function of the
`F_i` alone); reproduces Example 8.2's table exactly, including the pooled
`6/8 = 0.75` that only `{c1,c2}` achieves.

**Phase 2 — Index build** (§7; I1–I5). Attribute synopses, L2-normalized HNSW,
inverted index. *Accept:* HNSW recall vs exact scan on `santos` (small enough to
brute force) — quantifies §7.1's false-negative risk; `θ_dis = sqrt(2−2σ)`
verified to select the same neighbours as cosine `> σ`.

**Phase 3 — `retrieval.py`** (§4.2, §7.3). *Accept:* a table entering `D_sem`
via `A1` and `D_ovl` via `A2` yields **no** surviving pair (§7.3's explicit
requirement); a table with both `nationality` and `citizenship` surviving yields
exactly one pair, the higher-`sim` one; `overlap = 0` excluded; `|D_sem|`,
`|D_ovl|`, `|D_pair|`, `|D|` logged per query, plus HNSW false positives caught
by exact `N_i` (substantiates §7.1's asymmetric-error argument).

**Phase 4 — `stage1_dinkelbach.py`** (§5; M1, M6, D1, and `|D|` edge cases:
`|D| < k` → `∅`; `k ≤ |D| < α×k` → clamp pool to `D`, record
`effective_alpha = |D|/k`, never pad). *Accept:* exact vs brute force over all
`C(m, αk)` for `m ≤ 16`, ≥ 500 random instances including ties, equal `n_i`,
`N_i = 0`, `N_i = n_i`; λ monotone increasing; M6's claim
`max_{|S|=k} F ≥ max_{|S|=αk} F` verified by brute force.

**Phase 5 — `unionability.py`** (§6.2; M3, M4). *Accept:* agrees with
`verify_constrained` when the pinned column is the one the unconstrained solver
picks; a constructed case where they differ; returns `0.0` when `s(pin) ≤ σ`.

**Phase 6 — `stage2_ilp.py`** (§6.1; M1). `y_i(τ)` precomputed once;
`milp(integrality=1, constraints=[LinearConstraint(y, lb=-y_Q(τ)), LinearConstraint(ones, lb=k, ub=k)])`.
LP pre-check via `linprog`: **infeasible ⇒ ILP infeasible** (sound); the converse
does *not* hold, so LP-feasible still requires the ILP solve — §6.1 states this
one-directionally and should stay that way. *Accept:* exact vs brute force over
all `C(|P|, k)` for `|P| ≤ 18`; infeasible ⇒ `∅`, never a violating set; an
LP-feasible-but-ILP-infeasible instance constructed and handled.

**Phase 7 — `pipeline.py`** (§4.1). `retrieve → certify (M6) → Stage 1 →
unionability on P only → Stage 2`. Telemetry: `|D|`, `|P|`, `F_P`, `F_R`, `Δ`,
`U_{V_D}(Q,R)`, Dinkelbach iterations, ILP time/nodes, LP pre-check savings,
**#unionability computations**, feasibility verdict + cause (intrinsic vs
α-induced). *Accept:* end-to-end on `santos`, `k ∈ {5,10,20}`,
`α ∈ {1,2,3,5,10}`; `Δ(F*,F_R) ≤ δ` on every non-empty result; unionability
computations `= |P|`, independent of `|T|` (§4.1's central efficiency claim).

**Phase 8 — §8 alternative design** (S1–S4). Adapt `nl_swap`/`bnl_swap`/
`exhaustive_swap` to `CandidateStats`; fix S1, S2, S3; add the reflexivity
assertion. *Accept:* Claim 1 (locality) verified by brute force — no `R` with
feasible-individually-but-infeasible-union; Claim 2 verified by constructing
`t,t'` with `c*(t) ≠ c*(t')`; Claim 3's four conditions verified exhaustively over
small `(N,n,U)` grids; Claim 4 (strict partial order) property-tested on the
live config; NL and BNL return **identical** winnow sets (they must — same
relation, different algorithm); Example 8.2 reproduced. Deliverables for S4:
minimized incompleteness witness, and the sufficient-success condition with
brute-force verification.

**Phase 9 — Experiments** (fills the paper's `Section ??` gaps). α sweep
(quality vs time vs infeasibility, split by M6 cause); LP-pre-check savings
(§6.1); §4 vs §8 under a shared table universe (S5), measured in unionability
computations; NL vs BNL comparison counts and window sizes (§8.2's "we study
this further in the experiments"); M5 ablation (`max_f` vs `satisfice`).

---

## 8. Testing strategy

The load-bearing device is `tests/oracle.py`: **brute-force exact solvers** for
Stage 1, Stage 2, and Cheapest Repair. Both stages are pure functions of
`List[CandidateStats]` + `(N_Q, n_Q)`, so exhaustive enumeration for `m ≤ 18` is
fast and settles exactness definitively — turning "Dinkelbach is exact", "the ILP
is exact" and "greedy is incomplete" from claims into verified properties.

The paper supplies its own golden tests; use them as fixtures: Example 8.2's
five-table instance, Claims 1–4, and both NP-hardness reductions (Theorem 3.2
via E-`k`KP, Theorem 8.3 via Subset Sum) as *instance generators* — construct
the reduction, solve with our solver, check the objective matches the source
instance's optimum. That validates the reductions themselves, which is cheap
insurance on two proofs.

Property tests: `F_S` invariant under table reordering; `Δ ≤ δ` on every
non-empty result; `|R| = k` or `∅`; `R ⊆ P ⊆ D`; exactly one attribute per table
in `D`; λ monotone; NL ≡ BNL.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| M5 makes Stage-1 pools degenerate, depressing reported unionability | `satisfice` ablation already planned; report both. |
| Small `α` ⇒ empty results that look like lake infeasibility | M6's certificate separates the causes; report both rates. |
| I2/I3 unresolved ⇒ §7.3's tie-break incommensurable with §6.2 | Single normalized embedding space, asserted at index build. |
| `TableUnionNew` is a shared conda env; drift breaks reproducibility | Pin versions in config, assert at startup. |
| Stale `metadata_*.pkl` vs CSVs | Phase 0 samples and cross-checks row counts. |
| §8 code has grown a private data model; adapting it may alter behaviour | Golden-test the existing implementation *before* refactoring; require identical output on `santos`. |

---

## 10. Decisions (resolved 2026-08-09) and one still open

1. **M1** — `Q`-inclusive by default. **Confirmed: yes** — matches what's
   already built (`include_query=True` default, PLAN.md C1).
2. **M5** — unbounded `argmax F`, or satisficing at `τ`? **Resolved: unbounded
   `argmax F` is the intended design** (§3 above) — not an unsettled question
   between two candidate designs; `objective="max_f"` is correct as shipped.
3. **M6** — want the feasibility certificate in the paper? **Still open** — not
   answered in this round. The certificate's justification has been corrected
   regardless (§3 above, mirrors PLAN.md C4) since the original "Claim" was
   found false under M1's resolution.
4. **M4** — normalize `U_{V_D}` to `[0,1]`? **Resolved: no** — keep the raw
   sum; the paper's Table 1 needs revising, not §6.2's formula or the code.
5. **I1** — substitute an exact inverted index for JOSIE, or name JOSIE for
   comparability with `[13]`? **Resolved twice, in opposite directions, same
   day:** first "real JOSIE will be provided," then **superseded** hours later
   by `Fair_Table_Search_7.pdf` itself, which rewrites §7.2 to specify posting
   lists directly and retargets `[13]` to Zobel & Moffat 2006 — JOSIE is no
   longer cited anywhere in the paper. The second resolution wins because it
   comes from the source paper, not an implementation choice. Already-built
   `InvertedIndexOverlap` conforms to the new spec exactly, unmodified (see §4
   I1 for the point-by-point check). `JosieOverlap` will not be built.
6. **I2** — HNSW over Starmie column embeddings, or a separate encoder?
   **Resolved: confirmed** — the embeddings already exist and were verified
   present for `santos` in `starmie_fair`. No new encoder needed.
7. **S4** — produce the two deferred §8 items (incompleteness witness,
   sufficient-success condition)? **Resolved: no** — descoped.
8. Primary benchmark for §4's evaluation: `santos`, `santosLarge`, or TUS?
   **Resolved: `santos`.** Already present at
   `/u6/bkassaie/starmie_fair/data/santos/` (datalake, query, indexes,
   metadata, vectors) — confirm with the user whether that's the intended
   copy or a new one is coming before Phase 0 starts against it.
