# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

The two-stage optimization core (`PLAN.md`'s full scope) is **implemented and tested**: `duts/`
(`types.py`, `config.py`, `stats.py`, `stage1_dinkelbach.py`, `stage2_ilp.py`, `pipeline.py`) plus
`tests/` (`oracle.py`, `generators.py`, `test_{config,stats,stage1,stage2,pipeline}.py`). 1929 tests
pass, 19 skip (genuinely-infeasible knapsack fixtures at that cardinality), 0 warnings. Read
`NOTES.md` before touching any of this — it records two things found during implementation that
aren't in `PLAN.md`'s prose: a real `scipy.optimize.milp` correctness bug (see Commands/Invariants
below) and why the LP/ILP-equivalence acceptance criterion replaced the originally-planned
"LP-feasible-but-ILP-infeasible" fixture. This repo became a git repository on 2026-10-03 — there is no
commit history before that date. `.gitignore` excludes `experiments/results/_baseline_scratch/` (~21 GB of
regenerable HNSW indexes), `*.pkl`, `*.bin` and Python caches.

- **`PLAN.md` is the authoritative scope and has been kept in sync with the implementation** — every
  correction (`C1`–`C5`) and every trap found while building Stage 2 is recorded there, not just in
  code comments. Read the relevant section before touching a formula.
- **`PLAN-full.md` is the superset** (indexing §7, the §8 swap baseline, experiments) — reference
  material for the paper-level corrections (`M*`/`I*`/`S*`), whose open questions were resolved
  2026-08-09 (its §10). Do not pull work forward from it without being asked.
- **`DISTANCE-METRICS.md`** is the authoritative note on the paper's Euclidean-vs-cosine
  inconsistency (§7.1 vs §6.2 vs Table 1): the math, the two empirical verifications, what actually
  ships (`hnswlib space='cosine'`, no normalization step, no `θ_dis` conversion), and paper-ready
  wording for fixing §7.1/Table 1. Read it before touching anything metric-related — `PLAN-full.md`
  I3's original prescription was partly wrong and now points here.
- **`PLAN-integration.md` is the build plan** for everything between real tables and the core:
  retrieval (§7), unionability (§6.2), and the experiment harness. **Phases A–E complete**, plus a
  Phase F (groundtruth precision/recall + the Stage-1-skip ablation) that is recorded in `NOTES.md`
  but **not** in `PLAN-integration.md` §4 — sync that if you touch the plan.
  **`experiments/RESULTS.md` and `RESULTS-santos3-full.md` were removed on 2026-09-20** (moved to
  `experiments/results/_stale_fair3_pre_prune/`): they described the old santos3, which shared one fair copy per
  datalake table across queries. santos3 was rebuilt as **one fair copy per (query, table) pair** (as tusSmall3 /
  tusLarge3), with tables that cannot be augmented for a query dropped from that query's ground truth; the
  rewritten results are `experiments/fair3-report.md`. `--benchmark` is honoured by every CLI command and result
  files are benchmark-prefixed. Its governing constraint is that every data-touching
  component is swappable behind a `Protocol` in a **separate package** (`dutsx/`), so `duts/` stays
  pure and swapping any adapter is a one-line config change.

## Commands

```bash
# all commands run under the TableUnionNew conda env (see Environment below)
PY=/u6/bkassaie/.conda/envs/TableUnionNew/bin/python

$PY -m pytest -q                              # full suite (~7s)
$PY -m pytest tests/test_stage2.py -q         # one module
$PY -m pytest tests/test_stage1.py::test_c4_counterexample_regression -q   # one test
$PY -m pytest -q -k "alpha_induced"           # by keyword, across modules
$PY -m pyflakes duts tests                    # lint (pytest and pyflakes were pip-installed
                                               # into TableUnionNew during Phase 0; not preinstalled)
```

**Release entry point: `python main.py`** (`ExperimentApp` in `experiments/entry/app.py`; user docs in
`README.md`). Prompts for missing parameters, checks prerequisites (`experiments/entry/prereqs.py`),
builds missing embeddings / synopsis / DUTS HNSW index on confirmation and exits 3 asking for a
rerun, then runs `duts` or a Starmie baseline and prints results + execution environment. DUTS goes
through `dutsx.runner.run_query`; baselines reuse `run_starmie_baselines._init/_one` via its
`Settings` tuple. On santos3 (k=10, α=5) it reproduces `fair3-report.md` exactly (39/48, P 0.810,
R 0.3005). Tests: `tests/test_entry.py`. Pinned deps: `requirements.txt` / `environment.yml`.

No build step — it's a pure-Python library, no packaging config, imported as `duts.<module>`.
`pytest.ini` sets `pythonpath = .` so `import duts` and `import tests.oracle` both resolve without
installing the package.

## The source paper

**Two versions in the repo root — read `_7`, not `_6`, for anything touching §7.2.**
`Fair_Table_Search_7.pdf` (13 pages) supersedes `Fair_Table_Search_6.pdf` (11 pages) as of
2026-08-09: it rewrites §7.2 from "Value-overlap filtering via JOSIE" to **"Value-overlap filtering
via posting lists"** (an exact inverted index; JOSIE is no longer cited anywhere — `[13]` now points
to Zobel & Moffat 2006, not JOSIE) and elaborates §7.1's closing and §7.3's tie-break justification
to match. Every other section is unchanged (same headers, identical intro, verified). Both versions
are *Distribution-Aware Unionable Table Search*, Kassaie & Miller, PVLDB submission draft.

**The plans are derived from it and are as much a correction list as an implementation spec**:
`C1`–`C5` (PLAN.md) and `M1`–`M6`/`I1`–`I5`/`S1`–`S5` (PLAN-full.md) record where the code
intentionally **diverges from the printed math**. Read the relevant correction before implementing
any formula.

Section map: §2 notation · §3 problem definition + Theorem 3.2 (NP-hardness via E-`k`KP) ·
§4.1 two-stage framework, §4.2 two relevance notions · §5 Stage 1 / Dinkelbach · §6.1 Stage 2 ILP,
§6.2 constrained alignment · §7.1 HNSW, §7.2 value-overlap via posting lists (v7 — see above), §7.3
combining · §8 the greedy-swap alternative (Claims 1–4, Definition 8.1 Cheapest Repair, Example 8.2,
Algorithms 1–3).

**The paper has no experiments and no conclusion** — it ends at §8.2 followed by REFERENCES, and
carries live `Section ??` / `[?]` placeholders. Experimental numbers the text promises do not exist
yet; producing them is PLAN-full.md Phase 9, out of scope for the core.

The paper's notation is what the code should use: `V_D` distribution attribute, `M ⊂ Dom(V_D)`
distribution values, `F*` target proportion, `δ` max deviation, `α` pool-expansion factor, `k`
result size, `N_i`/`n_i` per-table match count and size, `T(S)` multiset union.

## Environment

Use the conda env **`TableUnionNew`** — verified: Python 3.8.5, numpy 1.24.4, scipy 1.10.1
(`scipy.optimize.milp` present), hnswlib, munkres 1.1.4, networkx 3.1, pandas 2.0.3.

```bash
conda activate TableUnionNew
# or, without activation:
/u6/bkassaie/.conda/envs/TableUnionNew/bin/python ...
```

Rejected envs (documented in PLAN-full.md §2): `tableunion` (py3.7.10, scipy 1.3.3 — no `milp`),
`base` (no hnswlib).

**⇒ Python 3.8 syntax only**: `typing.List/Dict/Tuple`, no builtin generics (`list[int]`), no
`match`. `pytest` and `pyflakes` are the only two packages added beyond what the plan required
(`numpy`/`scipy`); both were pip-installed straight into `TableUnionNew` with no conflicts.

## The core idea, and why it is buildable in isolation

```
D  ──Stage 1 (§5)──▶  P  ──Stage 2 (§6.1)──▶  R
   maximize F,           maximize ΣU,
   |S| = α×k             |S| = k, F ≥ τ
```

**Stage 1** picks a distribution-optimal pool of size `α×k`, ignoring unionability. It is a 0–1
linear-fractional program solved *exactly* by Dinkelbach's method; each iteration is one
top-(α×k) selection on `y_i(λ) = N_i − λ·n_i`.

**Stage 2** maximizes total unionability over `P` only, subject to `F ≥ τ := F* − δ`. With `τ` fixed
the fractional constraint linearizes, so it is an exact 0–1 ILP via `scipy.optimize.milp` (HiGHS).

`α ≥ 1` is the efficiency/quality dial: Stage 2's cost depends on `α×k`, never on `|T|`. That is the
paper's central efficiency claim and what Phase 4's telemetry must substantiate.

**The architectural keystone:** both stages are *pure functions* of `List[CandidateStats]` plus a
`(N_Q, n_Q)` query offset — no embeddings, no HNSW, no Starmie, no index.

```python
class CandidateStats(NamedTuple):
    table: str    # id
    N: int        # |{t ∈ T_i : t.V_D ∈ M}| — exact integer
    n: int        # |T_i|
    U: float      # U_{V_D}(Q,T_i) — supplied, never computed here
```

This is why candidate retrieval (§4.2/§7) and unionability computation (§6.2) cleanly detach: `D` is
an input from fixtures or a synthetic generator, and `U` is a per-candidate float. It is also why
brute-force verification is tractable — exhaustive enumeration over `m ≤ 18` settles "Dinkelbach is
exact" and "the ILP is exact" as verified properties rather than claims. `tests/oracle.py` (exact
brute-force solvers for both stages) is the load-bearing test device.

### Calling it

```python
from duts import pipeline
from duts.types import CandidateStats, QuerySpec

D = [CandidateStats(table="t1", N=7, n=10, U=0.82), ...]
query = QuerySpec(N_Q=3, n_Q=5, k=10, alpha=2, F_star=0.5, delta=0.05, include_query=True)

result = pipeline.run(D, query)          # PipelineResult; certify=False by default (see below)
result.selected                          # R: List[CandidateStats], length k or []
result.pool                              # P: List[CandidateStats] — R ⊆ P ⊆ D always
result.telemetry                         # Telemetry — see duts/types.py for the full field list;
                                          # infeasibility_cause (duts/ core, which still has the
                                          # certificate) is one of None | "insufficient_candidates"
                                          # | "intrinsic" | "alpha_induced" | "stage2_infeasible".
                                          # In dutsx/ + experiments/ only the 1st, 2nd and 5th occur.
```

⚠️ **The C4 certificate is gone from `dutsx/` and `experiments/` entirely** (2026-08-10, user
instruction — see NOTES.md's entry). There is no `certify` flag on `QueryTask`, no `--certify` CLI
option, and `intrinsic`/`alpha_induced` are no longer producible causes anywhere in the
result-producing path. `duts.pipeline.run(D, query, certify=True)` still exists — the pure core was
left untouched, since it carries the C4 counterexample regression fixture and there is no git
history to restore it from — but nothing in the experiment path can reach it. Removing it from
`duts/` too is a one-line ask. Verified on the 2026-08-10 regeneration: the certificate changed no
answer, only the label on 85 of 720 rows.

`stage1_dinkelbach.solve(...)` and `stage2_ilp.solve(...)` are also callable directly — both return
a `StageResult` (`selected`, `feasible`, `objective_value`, `iterations`, `info`) and are what
`pipeline.run` composes. `stage1_dinkelbach.certify_feasible(D, k, tau, ...)` is the C4 certificate,
called once by the pipeline before Stage 1 runs at all.

## Invariants that are easy to get wrong

These are the corrections most likely to be silently violated. Each has a specific failure mode.

- **`Q` is included in both objectives (C1/M1).** `include_query=True` is the default. The paper is
  self-inconsistent here (§3/§4.1 include `Q`; §5/§6.1 drop it; §8's Algorithm 1 does both).
  - Stage 1: `(N_Q − λ·n_Q)` is constant in `S`, so the *inner maximizer is unchanged* — still one
    top-(α×k) selection. `Q` enters only the value of `Φ(λ)` (the termination test) and the update
    `λ⁽ᵗ⁺¹⁾ = (N_Q + Σ N_i)/(n_Q + Σ n_i)`. §5 as printed omits it from the update and therefore
    solves the `Q`-free problem.
  - Stage 2: the constraint RHS is **`−y_Q(τ)`, not `0`**, where `y_Q(τ) := N_Q − τ·n_Q`. §6.1's
    `≥ 0` is only the `Q`-free special case. This one line is the single most consequential typo to
    reproduce by accident.
- **`M` is a value *set*, and `N_i` is integer arithmetic (C2/M2).** `N_i = Σ_{c∈M} count_i[c]`.
  Never store or derive `N_i` from float proportions. `N_i` cannot be precomputed at index time —
  `M` is a query-time parameter.
- **Determinism is required, not cosmetic (C5/D1).** Dinkelbach terminates on
  `|λ⁽ᵗ⁺¹⁾ − λ⁽ᵗ⁾| ≤ 1e-12` **or** `S⁽ᵗ⁺¹⁾ == S⁽ᵗ⁾`; cap at 100 and raise. Break `y_i(λ)` ties by
  `(−n_i, table_id)` — without a deterministic rule the set-equality termination test is
  meaningless. Select by sort (`O(m log m)`, matching the paper's stated bound); `np.argpartition`
  is faster but complicates tie determinism. Log iteration counts — "converges in only a few
  iterations" is an empirical claim this code exists to substantiate.
- **Degenerate cardinalities.** `|D| < k` → `∅`. `k ≤ |D| < α×k` → clamp `P` to `D`, record
  `effective_alpha = |D|/k`, **never pad**. Drop `n_i = 0` candidates.
- **LP pre-check is sound `LP-infeasible ⇒ ILP-infeasible` in general — and for Stage 2's specific
  constraint shape, the converse also holds** (one cardinality equality + one linear inequality: an
  exchange argument shows any fractional LP optimum can be walked to an integral one without leaving
  the feasible region; verified over 1000 random instances, zero gaps, once the `milp` bug below is
  worked around). So the LP pre-check's only value here is computational — skip branch-and-bound when
  the LP already proves infeasible — never a source of a false verdict. Do not resurrect the
  "LP-feasible-but-ILP-infeasible" test fixture PLAN.md originally called for; it cannot be
  constructed for this formulation (PLAN.md Phase 3, `NOTES.md`).
- **`scipy.optimize.milp`'s default presolve can lie.** With `presolve=True` (scipy's default),
  `milp` can return `success=True` with a solution that *violates* the constraint it was asked to
  satisfy — reproduced on scipy 1.10.1 **and** 1.13.1, confirmed via brute force that the witness
  instance is genuinely infeasible, confirmed `options={"presolve": False}` gives the correct
  verdict. Rare (~1-in-3000 random instances) but real, and the whole point of Stage 2 is an
  *exactness* guarantee — a rare wrong answer is still a wrong answer. `stage2_ilp.py` passes
  `options={"presolve": False}` on **both** the `milp` and `linprog` calls, and independently
  re-verifies every "successful" solution against the actual constraints before returning it
  (`tests/test_stage2.py::test_milp_presolve_trap_regression` is the permanent fixture). **Do not
  remove either the `presolve: False` option or the post-solve assertions as "unnecessary" —
  they're load-bearing, not defensive-programming boilerplate.**
- **The C4 feasibility certificate.** Sound directly from Problem (1)'s `|R| = k, R ⊆ D` constraint:
  `F_max^k := max_{|S|=k, S⊆D} F` is exactly the best proportion any feasible solution can reach.
  Running the *same* Dinkelbach solver at cardinality `k` over all of `D` gives `F_max^k`, splitting
  infeasibility into **intrinsic** (`F_max^k < τ` → return `∅`, skip the ILP) vs. **α-induced**
  (certificate passes but Stage 2 fails on `P`). One extra Dinkelbach run; the key diagnostic for
  tuning `α`, and telemetry reports the *cause*, not just the verdict.
  ⚠️ **Do not justify this with `max_{|S|=k} F ≥ max_{|S|=α×k} F`** — that claim is **false** under
  `include_query=True` (counterexample and proof in PLAN.md C4; kept as a regression fixture in
  `tests/test_stage1.py::test_c4_counterexample_regression`). The certificate doesn't need it.
  `α_induced` infeasibility is structurally impossible at `α = 1` (Stage 1's pool computation is then
  *literally the same call* as the certificate) and only appears for `α > 1` — a real instance is in
  `tests/test_pipeline.py::test_infeasibility_cause_alpha_induced`, found by random search because
  it wasn't obvious how to hand-construct.
- **C3/M5 is an open design question, not a bug — and `satisfice` is deliberately unimplemented.**
  Unbounded `argmax F` fills `P` with the most extreme tables, and at `F* = 0.5` Stage 2 degenerates
  to plain top-`k` inside a biased pool. `objective="max_f"` (Dinkelbach, as built) is the only
  implemented path; `duts/config.py::validate_config` raises `NotImplementedError` on
  `objective="satisfice"` rather than silently falling back to `max_f` — it has no defined objective
  function yet (three candidates are listed in PLAN.md C3: max `Σn_i`, min `|F_P−τ|`, max distinct
  `V_D` values). Don't stub it in; pick the objective first, since it likely needs different
  machinery than Dinkelbach (feasibility + a secondary objective, not a ratio maximization).
- **`U` normalization affects reporting only.** Stage 2's argmax is invariant to positive scaling.
  Note the §6.2-vs-Table-1 contradiction; do not let it block the core.

Two further paper inconsistencies, **not yet in the plans' correction lists** (found by checking the
plans against the PDF; unresolved):

- **`H` is defined as proportions, but §5 sums it to get a count.** §2 defines the histogram
  synopsis as `H_T[v] ≥ 0`, the *proportion* of `T`'s tuples taking value `v`, with `Σ_v H_T[v] = 1`.
  §5 then writes `N_i = Σ_{c∈M} H_i^{V_D}[c]`, where `N_i` is a tuple *count*. The two cannot both
  hold. This is independent support for C2 — implement `N_i` as an integer count over
  `Dict[value, count]` and treat §5's formula as using an unnormalized histogram.
- **`α ≥ 1` vs `α > 1`.** §4.1's framework paragraph says "`α ≥ 1` is a user-defined expansion
  factor"; the Stage 1 paragraph immediately below says "`α > 1`". The plans assume `α ≥ 1`, which
  matters because `α = 1` is the degenerate single-stage case and C4's certificate is stated for
  `α ≥ 1`. Keep `α ≥ 1`.

## Layout

```
DUTS/
├── CLAUDE.md  PLAN.md  PLAN-full.md  NOTES.md  Fair_Table_Search_6.pdf
├── pytest.ini                 # pythonpath = . ; testpaths = tests
├── configs/duts.json          # k, alpha, F_star, delta, include_query, seed, objective
├── duts/
│   ├── __init__.py            # calls assert_environment() on import
│   ├── config.py               # load_config/validate_config; version + tau-range assertions
│   ├── types.py                # CandidateStats, QuerySpec, StageResult, Telemetry, PipelineResult
│   ├── stats.py                 # N, n, F, Δ over set M with (N_Q,n_Q) offset — integers
│   ├── stage1_dinkelbach.py    # §5 + C1 offset, C4 certificate, C5 termination/seeding
│   ├── stage2_ilp.py           # §6.1 milp (Bounds(0,1), presolve=False) + LP pre-check, RHS=−y_Q(τ)
│   └── pipeline.py             # §4.1 driver: certify → Stage 1 → Stage 2 + Telemetry
└── tests/
    ├── oracle.py               # brute force: exact Stage 1, exact Stage 2
    ├── generators.py           # synthetic D; E-kKP reduction instances (τ = 1 − C/(k·c))
    └── test_{config,stats,stage1,stage2,pipeline}.py
```

Every phase's acceptance criteria in `PLAN.md` §5 are exactness criteria, not smoke tests — e.g.
Phase 2 requires agreement with brute force over all `C(m, α×k)` for `m ≤ 16` across ≥ 500 random
instances including ties, equal `n_i`, `N_i = 0`, and `N_i = n_i`; all pass (`NOTES.md`). If you add
a new code path, extend the corresponding property/brute-force test rather than adding a narrow
example-based one — that's the pattern every existing test file follows.

Theorem 3.2's **E-`k`KP reduction doubles as an instance generator** (`tests/generators.py`):
`knapsack_to_tables` builds tables from a knapsack instance (`n_j = c`, `N_j = c − w_j`, `U_j = p_j`),
`tau_from_knapsack_capacity(k, c, capacity)` derives the matching `τ`, and
`tests/test_pipeline.py::test_ekkp_reduction_matches_knapsack_optimum` solves via `stage2_ilp.solve`
and checks against a brute-force knapsack optimum. Must run `include_query=False` — the reduction's
arithmetic has no query term.

## Upstream: `~/starmie_fair` (read-only)

`/u6/bkassaie/starmie_fair` is a separate existing repo. **Nothing here writes to it**; the core does
not depend on it at all. It becomes relevant only when plugging in real `D` (§7) and real `U` (§6.2)
*after* the core is verified. Verified pointers:

| Need | Location |
|---|---|
| Per-attribute `Dict[value,count]` synopses (integers ⇒ exact `N_i`) | `TableMetadata.py:12` (`TableMetadata`), `:173` (`MetadataStore`), `:358` (`compute_union_distribution`) |
| `F`, `Δ` reference implementations | `HNSWSearcher_Fair.py:627` (`compute_F`, already has `include_query`), `:909` (`compute_Delta`) |
| Constrained alignment (§6.2) | `bounds.py:53` (`verify_constrained`), `:152` (`get_edges`), `:119`/`:137` (bounds), `:10` (`cosine_sim`) |
| HNSW probe (§7.1) | `HNSWSearcher_Fair.py:503` (`_find_candidates`) |
| §8 swap baselines | `nl_swap.py:116`, `bnl_swap.py:23`, `exhaustive_swap.py:20`, `HNSWSearcher_Fair.py:1093` |
| Column embeddings / prebuilt synopses | `data/<bench>/vectors/cl_{datalake,query}_drop_col_tfidf_entity_column_0.pkl`, `data/<bench>/indexes/metadata_{combined,datalake,query}.pkl` |
| `θ_cat = 50` | `experimental_config.json` → `domain_size_threshold` |

Note `compute_F` takes a **scalar** protected value, so it is not a drop-in for set-valued `M`; use
it only as a regression oracle at `|M| = 1` (PLAN-full.md M2). Prebuilt `.pkl` synopses may be stale
relative to the CSVs — cross-check sampled `n_i` against row counts before trusting them.

## Open questions

From PLAN.md §7 — questions for the paper's authors. **C1 and C3 are now resolved** (2026-08-09,
same round that resolved PLAN-full.md's M1/M4/M5/I1/I2/S4/benchmark — see below); C4 remains open
but doesn't block anything either way.

1. **C1** — `Q`-inclusive by default? **Resolved: yes.** Matches what's already built
   (`include_query=True` default in `QuerySpec`).
2. **C3** — unbounded `argmax F`, or satisficing at `τ`? **Resolved: unbounded `argmax F` is the
   intended design**, not an open choice between two candidates — `objective="max_f"` (the only
   implemented path) is correct as shipped. `"satisfice"` stays unimplemented
   (`NotImplementedError`); if ever built, it's an experimental ablation only, not a competing default.
3. **C4** — does the feasibility certificate go in the paper? **Still open.** Built and tested
   either way (sound independent of this answer — see Invariants); purely a paper-writing decision.

## What's built vs. what's still open

**`PLAN-integration.md`'s Phases A–E are complete** — ports/synopsis → retrieval → unionability →
runner → experiments, all against real `santos` data, results committed under `experiments/results/`.
Two things about the design that aren't obvious from `PLAN-full.md`:

- **`dutsx/` is a separate package from `duts/`**, with `typing.Protocol` seams (`SynopsisSource`,
  `SemanticRetriever`, `OverlapFilter`, `UnionabilityScorer`). The dependency arrow only points
  `dutsx → duts`. Swapping any adapter (e.g. a Roaring-bitmap-backed `OverlapFilter`, per
  `Fair_Table_Search_7.pdf` §7.2's implementation recommendation) is one class plus a config string —
  don't thread conditionals through a driver.
- **The end-to-end runner deliberately does *not* call `duts.pipeline.run`.** That entry point needs
  `U` populated for all of `D`, which would do `|D|` bipartite matchings where §4.1 promises `α×k`.
  The runner composes `stage1 → score U on P only → stage2` instead, which is what makes
  "#unionability computations = `|P|`, independent of `|T|`" a measurable claim. `pipeline.run`
  remains correct and is still the right entry point when `U` is already known.

**Still open**, from `PLAN-full.md` §10: **the §8 swap baseline** (S4's two deferred sub-items are
explicitly descoped), and **C4/M6** — whether the feasibility certificate goes in the paper. Phase E's
α-sweep gives M6 an empirical answer worth reading before deciding: `alpha_induced` infeasibility
never occurred on santos across the full sweep (see `NOTES.md`'s Phase E entry) — the certificate
still separates `insufficient_candidates` from `intrinsic`, just not the failure mode it was
originally designed to catch.

None of this is needed to trust the core — that's the entire point of building it in isolation first.
