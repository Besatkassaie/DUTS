# DUTS — Two-Stage Optimization Core

Scope: **only** the two-stage process of §4.1 — Stage 1 (§5) and Stage 2 (§6.1),
plus the driver that chains them. Nothing else.

Broader plan (indexing §7, swap baseline §8, experiments) preserved in
`PLAN-full.md`; not in scope here.

---

## 1. What the two-stage process is

```
D  ──Stage 1 (§5)──▶  P  ──Stage 2 (§6.1)──▶  R
   maximize F,           maximize ΣU,
   |S| = α×k             |S| = k, F ≥ τ
```

**Stage 1** — distribution only, unionability ignored:

```
P = argmax_{S ⊆ D, |S| = α×k}  F(T(S ∪ Q))
```

Solved **exactly** by Dinkelbach's parametric method on the 0–1 linear-fractional
program. Each iteration is one top-(α×k) selection.

**Stage 2** — unionability, constrained, over `P` only:

```
R = argmax_{S ⊆ P, |S| = k}  Σ_{T_i∈S} U_{V_D}(Q,T_i)
      s.t.  Δ(F*, F(T(S ∪ Q))) ≤ δ
```

Solved **exactly** as a 0–1 ILP with `τ := F* − δ` fixed, so the fractional
constraint linearizes to a single inequality.

That is the whole of it. `α ≥ 1` is the user's efficiency/quality dial: Stage 2's
cost depends on `α×k`, never on `|T|`.

---

## 2. Scope boundary

**In:** `N_i`/`n_i`/`F`/`Δ` arithmetic, Dinkelbach, the ILP, the driver, and a
brute-force oracle to prove both stages exact.

**Out, and why it cleanly detaches:**

| Excluded | Where it belongs | How the core copes |
|---|---|---|
| Candidate retrieval `D` (§4.2, §7: HNSW, posting lists, θ_cat) | upstream of Stage 1 | `D` is an input — supply from fixtures or a synthetic generator |
| `U_{V_D}(Q,T)` computation (§6.2 bipartite matching) | upstream of Stage 2 | `U` is a per-candidate float — supply from fixtures |
| Swap-based repair (§8) | a different design entirely | ignored |

This works because **both stages are pure functions of
`List[CandidateStats]` + `(N_Q, n_Q)`** — no embeddings, no index, no Starmie, no
HNSW. That is what makes the core independently testable and why the two-stage
process can be built and verified first.

```python
class CandidateStats(NamedTuple):
    table: str    # id
    N: int        # |{t ∈ T_i : t.V_D ∈ M}| — exact integer
    n: int        # |T_i|
    U: float      # U_{V_D}(Q,T_i) — given, not computed here
```

Dependencies: `numpy`, `scipy ≥ 1.9` (for `optimize.milp`). Use conda env
**`TableUnionNew`** (py3.8.5, scipy 1.10.1 — `milp` present). ⇒ **Python 3.8
syntax**: `typing.List/Dict`, no builtin generics.

⚠️ **`pytest` is not installed in `TableUnionNew`.** Of the eight conda envs on
this machine only `ocr` (py3.11) and `tableunion` (py3.7) have it, and neither
can run this code. §6's entire testing strategy assumes a suite. Settle it in
Phase 0: install `pytest` into `TableUnionNew`, or write the suite on stdlib
`unittest`.

---

## 3. Corrections that survive the narrowing

Five of the paper's issues bear on the two-stage core. (Indexing and §8 issues
drop out with their sections.)

### C1 — Does `Q` enter the objectives? *(must settle first)*
§3 and §4.1 write `F(T(S ∪ Q))`; §5's Dinkelbach and §6.1's ILP both drop `Q`.
Different problems. **Decision: include `Q`**, flag `include_query=True`.
Corrected math — both stay in the same complexity class:

**Stage 1.** Maximize `(N_Q + Σ_{i∈S} N_i)/(n_Q + Σ_{i∈S} n_i)`:

```
Φ(λ) = max_{|S|=αk} [ (N_Q − λ·n_Q) + Σ_{i∈S} (N_i − λ·n_i) ]
```

`(N_Q − λ·n_Q)` is **constant in `S`** ⇒ the inner maximizer is unchanged: still
one top-(α×k) selection on `y_i(λ) = N_i − λ·n_i`. `Q` enters only

1. the value of `Φ(λ)` (the termination test), and
2. the update `λ^(t+1) = (N_Q + Σ_{i∈S^(t)} N_i)/(n_Q + Σ_{i∈S^(t)} n_i)`.

So §5's iteration is correct *provided* step (3) uses the `Q`-inclusive ratio.
As printed it does not.

**Stage 2.** With `y_Q(τ) := N_Q − τ·n_Q`:

```
(N_Q + Σ N_i x_i)/(n_Q + Σ n_i x_i) ≥ τ   ⟺   Σ_i y_i(τ)·x_i ≥ −y_Q(τ)
```

RHS is `−y_Q(τ)`, not `0`. §6.1's `≥ 0` is the `Q`-free special case.

### C2 — `M` is a value set
`N_i = Σ_{c∈M} count_i[c]`, integer arithmetic throughout. Never store or derive
`N_i` from float proportions.

### C3 — Stage 1 maximizes `F` without bound
`Δ = F* − F` is signed, so overshoot is admissible and `argmax F` fills `P` with
the most extreme tables. At `F* = 0.5` the pool is distributionally degenerate,
Stage 2's constraint is slack for *every* subset, and Stage 2 collapses to plain
top-`k` unionability inside a biased pool. **Decision:** ship §4.1 faithfully as
default (`objective="max_f"`); add `objective="satisfice"` (diversity subject to
`F ≥ τ`) as a measured ablation. This is the sharpest quality question in the
two-stage design and it is entirely internal to Stage 1.

⚠️ **`satisfice` has no objective function yet.** "Diversity subject to `F ≥ τ`"
is not implementable as written, and it is not a Dinkelbach problem — Dinkelbach
maximizes a ratio, whereas this is feasibility plus a *secondary* objective, i.e.
different machinery and probably a different module than `stage1_dinkelbach.py`.
Pin the objective before Phase 2 (candidates: maximize `Σ_{i∈S} n_i`; minimize
`|F_P − τ|`; maximize distinct `V_D` values represented in `P`), or drop it from
Phase 4's acceptance criteria. See open question 2.

### C4 — Free exact infeasibility certificate
Problem (1) fixes `|R| = k` and `R ⊆ D`, so

```
F_max^k := max_{|S|=k, S⊆D} F(T(S ∪ Q))
```

is exactly the largest achieved proportion **any** feasible solution can have.
It comes from running the same Dinkelbach solver at cardinality `k` over all of
`D` — one extra `O(I·m·log m)` run:

- `F_max^k < τ` ⇒ infeasible over all of `D` → return `∅`, skip the ILP;
- `F_max^k ≥ τ` but Stage 2 infeasible on `P` ⇒ infeasibility is **α-induced**.

Distinguishes "the lake can't do it" from "α was too small" — the key diagnostic
for tuning `α`, and it needs nothing outside the core.

⚠️ **Do not justify this with the cardinality-monotonicity claim**
`max_{|S|=k} F ≥ max_{|S|=α×k} F`. **That claim is false under C1.** With `Q`
included, `F(T(S∪Q))` is the `n`-weighted average over `{Q} ∪ S` and `Q` is
forced in at *every* cardinality, so a low-`F_Q`, large-`n_Q` query is pulled
*up* by adding more tables and a larger set can beat every smaller one.
Brute-forced counterexample — `(N_Q,n_Q) = (0,1000)`, `t1 = (N=5,n=10)`,
`t2 = (N=4,n=10)`, `k=1`, `α=2`:

```
include_query=True    max|S|=1 = 0.004950  <  max|S|=2 = 0.008824   ✗
include_query=False   max|S|=1 = 0.500000  ≥  max|S|=2 = 0.450000   ✓
```

The claim holds only `Q`-free, where `F_S` really is the `n_i`-weighted average
of `{F_i}_{i∈S}` — that is the scope of §5's identity. The certificate above
does not need it: it is sound directly from `|R| = k`.

### C5 — Dinkelbach termination and ties
Terminate on `|λ^(t+1) − λ^(t)| ≤ 1e-12` **or** `S^(t+1) == S^(t)`; cap 100,
raise on cap. Break `y_i(λ)` ties by `(−n_i, table_id)` — determinism is
*required* for the set-equality test to mean anything. Select by sort
(`O(m log m)`, matching §4.1's stated bound). Log iteration counts: §4.1's
"converges in only a few iterations" is an empirical claim this code should
substantiate.

**λ monotonicity is one-sided only after the first step.** Dinkelbach converges
monotonically from either side, but §5's `λ^(0) = (N_Q + Σ_D N_i)/(n_Q + Σ_D n_i)`
— the full-pool ratio — is provably `≤ λ*` only in the `Q`-free case. Under C1 it
can overshoot: 100 identical tables `(N=5, n=10)` with `(N_Q,n_Q) = (0,1000)` and
`α×k = 2` give `λ^(0) = 0.2500` against `λ* = 0.0098`, so the first step
*decreases*. Either seed `λ^(0)` from an actual size-(α×k) subset (e.g. the
top-(α×k) by `F_i`), or assert monotonicity from iteration 1 onward. Asserting
"λ monotone increasing" from `λ^(0)` would fail a correct solver.

Degenerate cardinalities: `|D| < k` → `∅`; `k ≤ |D| < α×k` → clamp `P` to `D`,
record `effective_alpha = |D|/k`, never pad. Drop `n_i = 0` candidates.

**Not relevant at this scope:** `U` normalization (§6.2 vs Table 1) — Stage 2's
argmax is invariant to positive scaling, so it affects reporting only. Note it,
ignore it.

---

## 4. Layout

```
DUTS/
├── PLAN.md  PLAN-full.md  NOTES.md
├── configs/duts.json          # k, alpha, F_star, delta, include_query, seed, objective
├── duts/
│   ├── types.py               # CandidateStats, QuerySpec, StageResult, Telemetry
│   ├── stats.py               # C2/C1: N,n,F,Δ over set M with (N_Q,n_Q) offset — integers
│   ├── stage1_dinkelbach.py   # §5 + C1 offset, C4 certificate, C5 termination
│   ├── stage2_ilp.py          # §6.1 milp (Bounds(0,1), presolve=False!) + LP pre-check, RHS = −y_Q(τ)
│   └── pipeline.py            # §4.1 driver: certify → Stage 1 → Stage 2 + telemetry
└── tests/
    ├── oracle.py              # brute force: exact Stage 1, exact Stage 2
    ├── generators.py          # synthetic D; E-kKP reduction instances
    └── test_{stats,stage1,stage2,pipeline}.py
```

---

## 5. Phases

**Phase 0 — Scaffold.** `types.py`, `configs/duts.json`, startup assertion on
`scipy ≥ 1.9` and Python 3.8; test-runner decision (§2). Validate at config load:
`F* ∈ [0,1]`, `δ ≥ 0`, `k ≥ 1`, `α ≥ 1`, and the derived `τ = F* − δ` — `τ < 0`
makes Stage 2's constraint vacuous and `τ > 1` makes it unsatisfiable. Both should
be named at startup, not diagnosed later from an empty result. Half a day.

**Phase 1 — `stats.py`** (C1, C2). `N`, `n`, `F`, `Δ` in integer arithmetic;
`Q` as an explicit `(N_Q, n_Q)` offset.
*Accept:* reproduces §5's weighted-average identity `F_S = Σ w_i F_i`,
`w_i = n_i/Σn_j`, exactly; and witnesses §3's aggregation-sensitivity property
with a concrete pair of instances having **identical `{F_i}` but different `F_S`**.
(The universally-quantified form — "`F_S` is not *any* arithmetic combination of
the `F_i`" — is not testable as an assertion; the witness pair is.)

**Phase 2 — `stage1_dinkelbach.py`** (§5; C1, C3, C4, C5). Fixed-point iteration,
`λ^(0)` seeded from a size-(α×k) subset (C5), deterministic top-(α×k),
`certify_feasible(D, k, τ)`.
*Accept:* **exact vs brute force** over all `C(m, α×k)` for `m ≤ 16`, ≥ 500 random
instances incl. ties, equal `n_i`, `N_i = 0`, `N_i = n_i`; λ monotone **from
iteration 1** (C5); `certify_feasible` agrees with brute-force
`max_{|S|=k, S⊆D} F` on every instance; C4's counterexample kept as a regression
fixture — brute force must confirm `max_{|S|=k} F < max_{|S|=α×k} F` there, so the
retired monotonicity claim cannot creep back in; iteration counts logged.

**Phase 3 — `stage2_ilp.py`** (§6.1; C1). `y_i(τ)` precomputed once, then

```python
milp(c=-U, integrality=1, bounds=Bounds(0, 1),
     constraints=[LinearConstraint(y,    lb=-y_Q(τ)),
                  LinearConstraint(ones, lb=k, ub=k)])
```

**Three silent traps, all load-bearing.**

**1 — `milp` minimizes.** The objective passed is `-U`, not `U`.

**2 — `bounds` defaults to `Bounds(0, np.inf)`.** `integrality=1` on its own means
*non-negative integer*, not binary — without `Bounds(0, 1)` the solver can satisfy
both constraints by **taking one table twice**. Verified on scipy 1.10.1:
maximizing `10x₀ + x₁ + x₂` s.t. `Σx = 2` returns `[2, 0, 0]` unbounded versus
`[1, 0, 1]` with `Bounds(0,1)`.

**3 — HiGHS's default presolve can report `success=True` on a genuinely
infeasible instance, with a solution that violates the constraint it was asked
to satisfy.** Found empirically (not hypothesized) at ~1-in-3000 on random small
`(m, k, y, rhs)` instances, on **both** scipy 1.10.1 and 1.13.1, with two
independent constraint formulations (separate `LinearConstraint`s and one
combined 2-row matrix). Confirmed as a presolve artifact, not a modeling error:
`options={"presolve": False}` on the exact same instance reports the correct
(infeasible) verdict, and brute force over all `C(m,k)` subsets confirms no
feasible subset exists. **Mitigation:** disable presolve on both the ILP and the
LP pre-check (pool sizes are bounded by `α×k`, independent of `|T|`, so
presolve's speed benefit isn't needed here), and independently re-verify every
returned solution against the actual constraints before trusting `success` —
belt and suspenders, once a solver has been caught lying once. This needs a
one-line mention in the paper's implementation details: exactness in Stage 2
depends on this workaround, not just on the formulation being correct.

LP pre-check via `linprog`: **LP infeasible ⇒ ILP infeasible** (sound); §6.1
states this one-directionally, since in general the converse does not hold.

⚠️ **For this specific constraint shape, the converse *does* hold — verified,
not just asserted.** Stage 2's feasible region is `{x ∈ [0,1]^m : Σx = k,
y·x ≥ rhs}` — one cardinality *equality* plus one linear inequality, nothing
else. A standard exchange argument shows this exact shape is LP/ILP
equivalent: given any fractional optimum, take two fractional coordinates
`a, b`, and shift mass between them (`x_a += t, x_b −= t`) — this preserves
`Σx = k` exactly, and moves `y·x` *linearly* in `t`, so at least one direction
is non-decreasing; walking that direction to a boundary integralizes one
coordinate without ever leaving the feasible region. Repeat until nothing is
fractional. So **LP-feasible ⇒ ILP-feasible** here, and an
LP-feasible-but-ILP-infeasible witness cannot be constructed for Stage 2 as
formulated — confirmed by 1000 random `(m,k,y,rhs)` trials (zero gaps, both
directions checked, solutions independently re-verified) once trap 3's fix
(presolve disabled) is applied, plus the proof. **With default presolve the
same search instead surfaces trap 3** — an apparent "LP infeasible, ILP
feasible" case that is not a counterexample to the equivalence but the solver
bug above; do not mistake one for the other. This equivalence is a stronger,
useful fact worth stating in the paper, not just a stray implementation note:
it means the LP pre-check's entire value is computational (skip
branch-and-bound when the cheap LP already proves infeasibility) and — once
trap 3 is mitigated — it can never produce a false green light. Drop the
"LP-feasible-but-ILP-infeasible instance" item from the accept criteria below;
replace with an equivalence check.

*Accept:* exact vs brute force over all `C(|P|, k)` for `|P| ≤ 18`; every returned
`x` is binary with `Σx = k` **and satisfies the distribution constraint when
independently recomputed** (guards traps 2 and 3); infeasible ⇒ `∅` and never a
constraint-violating set; **LP-feasible ⇔ ILP-feasible** verified across ≥ 1000
random instances (the equivalence above, empirically); LP pre-check timing
recorded (fills §6.1's `Section ??`).

**Phase 4 — `pipeline.py`** (§4.1). `certify (C4) → Stage 1 → Stage 2`.
Telemetry per query: `|D|`, `|P|`, `F_P`, `F_R`, `Δ`, `ΣU`, Dinkelbach
iterations, ILP time, LP savings, feasibility verdict **and its cause**
(intrinsic vs α-induced).
*Accept:* `Δ(F*,F_R) ≤ δ` on every non-empty result; `|R| = k` or `∅`;
`R ⊆ P ⊆ D`; α sweep `α ∈ {1,2,3,5,10}` × `k ∈ {5,10,20}` on synthetic `D`
showing the quality/cost trade-off and the C3 ablation (`max_f` vs `satisfice`).

Then — and only then — plug in real `D` (§7) and real `U` (§6.2). The core needs
neither to be finished or trusted.

---

## 6. Testing

`tests/oracle.py` is the load-bearing device: brute-force exact solvers for both
stages. Because both stages are pure functions of `List[CandidateStats]` +
`(N_Q, n_Q)`, exhaustive enumeration for `m ≤ 18` is fast and settles exactness
definitively — turning "Dinkelbach is exact" and "the ILP is exact" from claims
into verified properties. This is the main reason to build the core in isolation.

Bonus fixture: Theorem 3.2's **E-`k`KP reduction as an instance generator** —
construct tables from a knapsack instance (`n_j = c`, `N_j = c − w_j`,
`u_j = p_j`), solve with Stage 2, check the optimum matches the knapsack optimum.
Cheap validation of the reduction itself. Two conditions the recipe must honour:

- **Derive `τ` from the capacity.** `F_S = 1 − Σ_{j∈S} w_j/(k·c)`, so
  `F_S ≥ τ ⟺ Σ w_j ≤ k·c·(1−τ)`; matching a knapsack capacity `C` therefore
  requires **`τ = 1 − C/(k·c)`**. The value `(k−1)/k` hard-codes `C = c`, which
  admits only a near-trivial slice of instances.
- **Run it `Q`-free** (`include_query=False`, or `N_Q = n_Q = 0`). The reduction's
  arithmetic is `F_S = ΣN_j/Σn_j`; a non-null `Q` breaks the equivalence.

Property tests: `F_S` invariant under reordering; λ monotone from iteration 1
(C5); `Δ ≤ δ` on every non-empty result; result size exactly `k` or `∅`;
`R ⊆ P ⊆ D`; Stage 2's `x` always binary.

---

## 7. Open questions

1. **C1** — `Q`-inclusive by default? §3 says yes, §5/§6 say no. Plan assumes yes.
   Note this one choice is what invalidates the cardinality-monotonicity claim
   (C4), the `λ^(0)` bound (C5) and the E-`k`KP fixture (§6); `include_query=False`
   restores all three. It is the highest-leverage decision in the plan.
2. **C3** — is unbounded `argmax F` the intended Stage 1, or satisficing at `τ`?
   Materially changes reported unionability. If satisficing: **what is the
   objective?** Currently unspecified, and it needs a different solver than
   Dinkelbach.
3. **C4** — want the certificate in the paper? Near-free; strengthens §3's `∅`
   claim. Present it as it now stands (sound directly from `|R| = k`); the
   monotonicity claim must not go in.

C1 and C3 gate Phase 1 and Phase 2 respectively. C4 is additive and can land
later without rework.
