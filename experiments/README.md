# Running the experiments

```bash
PY=/u6/bkassaie/.conda/envs/TableUnionNew/bin/python
cd /u6/bkassaie/DUTS
```

All commands write CSV + parquet twins into `experiments/results/` (override
with `--output-dir`). Every sweep is seeded (`--seed`, default 42) and
config-driven — a rerun with the same flags reproduces the same numbers,
modulo wall-clock timing columns.

## The five experiments

```bash
$PY -m experiments.cli alpha-sweep           # alpha x k sweep, quality vs cost vs infeasibility cause
$PY -m experiments.cli lp-precheck           # LP relaxation pre-check: does it actually save time?
$PY -m experiments.cli retrieval-ablation    # hnsw vs exact_scan, inverted_index vs null overlap
$PY -m experiments.cli groundtruth-eval      # precision/recall vs santos's own groundtruth, per stage
$PY -m experiments.cli stage1-skip-ablation  # two-stage pipeline vs skipping Stage 1 entirely
$PY -m experiments.cli all                   # alpha-sweep + lp-precheck + retrieval-ablation
```

`--limit-queries N` truncates the query list for a fast smoke run — omit it
for the real sweep (48 usable santos queries).

### `groundtruth-eval` — the one to reach for on "did this actually work"

```bash
$PY -m experiments.cli groundtruth-eval --benchmark santos  --k 3 --alpha 2
$PY -m experiments.cli groundtruth-eval --benchmark santos3 --k 3 --alpha 2
```

Runs every query end to end and reports, per query and in aggregate:

- **feasibility rate** — how many queries got a non-empty result at all
  (`infeasibility_cause`, split by cause: `insufficient_candidates` or
  `stage2_infeasible` — see `dutsx/runner.py`).
- **per-stage funnel** — mean `n_sem`, `n_ovl`, `n_pair`, `n_D`, `n_P`, `n_R`:
  how many candidates survive the semantic filter, the overlap filter, their
  intersection, the final per-table reduction, Stage 1's pool, and Stage 2's
  result.
- **precision/recall at each of D, P, R** against
  `<benchmark>_small_benchmark_groundtruth.csv` (santos's own table-union
  groundtruth, present for all four santos variants). See
  `experiments/groundtruth.py`'s docstring for exactly what this does and
  doesn't measure — in short, it tests whether distribution-first selection
  throws away or lets through tables the benchmark independently considers
  unionable, **not** whether DUTS picked the "right" join column (the
  groundtruth's join column and DUTS's `V_D` are usually different columns).
- **runtime** — mean/max `wall_time_s` per query.

`--benchmark {santos,santos2,santos3,santos4}` selects which of the four
santos variants to run against (see `experiments/context.py`'s
`BenchmarkPaths` docstring for what differs between them — query set size,
datalake size, and whether the queries are the original or fairness-rebalanced
`_fair.csv` variants). There is no `--certify` flag: the C4 feasibility
certificate was removed from this path on 2026-08-10, so `intrinsic` and
`alpha_induced` are no longer producible causes.

### `stage1-skip-ablation` — what does Dinkelbach's pool selection actually buy?

```bash
$PY -m experiments.cli stage1-skip-ablation --benchmark santos  --k 10 --alpha 1 --top-n 500
$PY -m experiments.cli stage1-skip-ablation --benchmark santos3 --k 10 --alpha 1 --top-n 500
```

For every query, runs **two conditions** side by side and emits two rows per
query (`config_id` prefixed `two_stage__...` / `skip_stage1__...`, joinable
by `q_table`):

- `two_stage` — the normal pipeline: retrieve → Stage 1 Dinkelbach pool
  (size `alpha*k`) → score `U` on the pool only → Stage 2 ILP over the pool.
  This is `dutsx.runner.run_query`, unmodified.
- `skip_stage1` — bypass Stage 1 entirely: score `U` for **every** retrieved
  candidate in `D`, then hand all of `D` straight to Stage 2's ILP at
  cardinality `k`. Same `duts.stage2_ilp.solve` call, just over a bigger
  candidate set.

Both conditions solve the identical Stage 2 problem (`|R|=k, F_R >= tau`)
over different candidate sets — `skip_stage1` is not an approximation, it's
what "no distribution-first pre-filter" costs, holding retrieval and scoring
logic fixed. Since `D ⊇ P`, `skip_stage1`'s `sum_U` optimum can only be `>=`
`two_stage`'s — never a free lunch, always a runtime-vs-quality trade,
reported both ways rather than assumed. `experiments/stage1_skip_ablation.py`
asserts this monotonicity never breaks on real data (see the CLI's own
"sum_U monotonicity check" printout).

**This ablation is only informative when Stage 1 has candidates to prune** —
if `alpha*k >= |D|`, the C5 clamp makes `two_stage`'s pool equal to `D`
anyway and the two conditions collapse to identical numbers. On santos's
default `--top-n 100`, retrieved pools are small enough (~10-20 candidates)
that this happens for most queries at `alpha=2`; use a **small `alpha`
(1)** and a **wide `--top-n`** to force real separation. Real numbers at
`k=10, alpha=1, top_n=500` (see `experiments/results/*_stage1_skip_ablation.csv`):

| benchmark | mean n scored (two_stage → skip_stage1) | mean scoring_time_s | mean sum_U | mean F_R |
|---|---|---|---|---|
| santos  | 10.0 → 14.6 (+46%) | 0.0095s → 0.0134s (+41%) | 86.98 → 94.44 | 0.525 → 0.486 |
| santos3 | 10.0 → 22.0 (+120%) | 0.0082s → 0.0170s (+108%) | 93.23 → 99.21 | 0.484 → 0.419 |

Two things worth noting: (1) `two_stage`'s scoring cost stays flat at
`alpha*k=10` regardless of benchmark size, while `skip_stage1`'s tracks
`|D|` and grows with the datalake (santos3's larger, planted-table-enriched
datalake gives it more candidates to retrieve) — the paper's efficiency
claim, made visible as a runtime number rather than an asymptotic argument.
(2) `skip_stage1` always wins on raw `sum_U` (guaranteed, superset) but
`two_stage` consistently lands closer to `F_star=0.3` — because Stage 1
explicitly optimizes `F` (unbounded `argmax`, C3) before Stage 2 ever runs,
`skip_stage1` has no such pre-filter and Stage 2 alone doesn't optimize `F`
beyond the `>= tau` floor.

## Flags common to every command

| Flag | Default | Meaning |
|---|---|---|
| `--seed` | 42 | RNG seed, for reproducibility |
| `--f-star` | 0.3 | target proportion `F*` |
| `--delta` | 0.15 | max allowed deviation `δ` (so `τ = F* − δ = 0.15`) |
| `--top-n` | 100 | ANN probe width (§7.1) |
| `--no-include-query` | off | drop `Q` from both stage objectives (C1) |
| `--unionability` | `pinned_match` | `pinned_match` (§6.2 spec) \| `starmie_verify` (pins the row, not the pair — see M3) \| `constant` (ablation double) |
| `--limit-queries` | none | truncate the query list — smoke-test aid |

`alpha-sweep`/`lp-precheck`/`all` additionally take `--alphas` (default
`1 2 3 5 10`) and `--ks` (default `5 10 20`). `retrieval-ablation` and
`groundtruth-eval` take a single fixed `--k`/`--alpha` (defaults `10`/`2.0`).

## Reading the output

Every row goes through `experiments/schema.py`'s `ResultRow` — one column
layout across all five experiments, so CSVs from different runs concatenate
cleanly (a column an experiment doesn't populate, e.g. `precision_r` outside
`groundtruth-eval`, is simply `None`/blank there). Load with pandas:

```python
import pandas as pd
df = pd.read_csv("experiments/results/santos_groundtruth_eval.csv")
```

`experiments/analysis.py` has the aggregation helpers the CLI itself uses
(`infeasibility_by_cause`, `quality_vs_cost_by_config`,
`check_efficiency_invariant`) if you want the same breakdowns on a fresh run.

## What's already been run

`experiments/results/` has committed output from prior runs — inspect these
without rerunning anything:

- `alpha_sweep.csv` — the full `α×k` sweep, 48 santos queries.
- `lp_precheck.csv` + `lp_precheck_synthetic.csv` — real santos pools are too
  small (~10 candidates) to exercise the LP pre-check's branch-and-bound
  savings, so there's a synthetic supplementary demo at pool sizes 100–3000.
  Headline finding: the pre-check is **consistently slower**, not faster (see
  `NOTES.md`'s Phase E entry for why — it follows directly from the LP/ILP
  equivalence proof for this constraint shape).
- `retrieval_ablation.csv` — `hnsw` vs `exact_scan` (indistinguishable, as
  expected from 0.998 mean recall); `inverted_index` vs `null` overlap (the
  overlap filter is load-bearing — dropping it roughly doubles `n_D` and
  raises `stage2_infeasible`).
- `santos_end_to_end.csv`, `santos{2,3,4}_end_to_end.csv` — the four-benchmark
  comparison across santos/santos2/santos3/santos4 (base vs. fairness-rebalanced
  queries, × 550 vs. 931-table datalakes). ⚠️ **Stale as of 2026-08-10**: these
  four predate the C4 certificate's removal and still carry the dropped
  `certify` / `F_max_k` / `certify_time_s` / `dinkelbach_iterations_certify`
  columns. No `experiments.cli` command regenerates them (they came from a
  Phase D one-off), so they were left in place rather than silently rewritten —
  don't mix them with the 2026-08-10 files above.
- `santos_groundtruth_eval.csv`, `santos3_groundtruth_eval.csv` — precision
  100% at every stage on both; recall falls off through the funnel (expected —
  DUTS optimizes a stricter, distribution-constrained problem, not coverage).
- `santos_stage1_skip_ablation.csv`, `santos3_stage1_skip_ablation.csv` —
  `two_stage` vs `skip_stage1` at `k=10, alpha=1, top_n=500`; see the
  `stage1-skip-ablation` section above for the numbers and how to read them.

## What's not built yet

- `JosieOverlap` — **will not be built.** `Fair_Table_Search_7.pdf` dropped
  JOSIE from the paper in favor of exactly the posting-list design
  `InvertedIndexOverlap` already implements (see `PLAN-full.md` I1).
- A `space="cosine"`-vs-`space="l2"` comparison beyond what
  `DISTANCE-METRICS.md` already covers.
