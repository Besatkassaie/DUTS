# The distance-function problem: Euclidean vs cosine

Single authoritative note on the metric inconsistency in the paper, the math
that reconciles it, what was verified empirically, and what the paper should
say. Supersedes the scattered fragments in `PLAN-full.md` I3 and
`PLAN-integration.md` Phase B — those now point here.

**Status:** resolved and verified (2026-08-09). Ships as `space='cosine'`.

---

## 1. The problem: three parts of the paper specify three different things

| Where | What it says | Metric |
|---|---|---|
| §7.1 (HNSW retrieval) | retain neighbours whose **Euclidean distance** falls below `θ_dis` | Euclidean |
| §6.2 (alignment) + `bounds.py:10` | retain edges whose **cosine similarity** is above `σ` (= 0.6) | cosine |
| Table 1 (`sim(A, V_D)`) | "as computed by the HNSW index" | *defers to whichever the index uses* |

Table 1 is the load-bearing one, and it is circular: it defines the similarity
that §7.3's per-table `argmax sim` tie-break depends on by pointing at the
index, while §7.1 and §6.2 disagree about what the index computes.

**Why this is not cosmetic.** §7.3 resolves a table with several qualifying
attributes by keeping the most semantically similar one. If retrieval ranks by
Euclidean distance over *unnormalized* vectors and alignment scores by cosine,
the tie-break orders candidates by one quantity and the downstream unionability
score uses another. They are not monotonically related in general, so §7.3's
"most similar attribute" is ill-defined — the pipeline can hand Stage 2 an
attribute that is not the one it would have chosen.

The vectors this project actually uses **are not normalized**: sampled L2 norm
≈ 9.75 on `cl_query_drop_col_tfidf_entity_column_0.pkl`. So this is a real
condition, not a hypothetical one. Pinned by
`tests/test_dutsx_retrieval.py::test_stored_vectors_are_not_l2_normalized`.

---

## 2. The reconciliation

For **L2-normalized** vectors `a, b` (`‖a‖ = ‖b‖ = 1`):

```
‖a − b‖²  =  ‖a‖² + ‖b‖² − 2·a·b  =  2 − 2·cos(a, b)
```

`‖a − b‖` is therefore a strictly decreasing function of `cos(a, b)`, so
thresholding on one is exactly equivalent to thresholding on the other:

```
‖a − b‖ < θ_dis   ⟺   cos(a, b) > σ,    with   θ_dis = sqrt(2 − 2σ)
```

At `σ = 0.6`: `θ_dis = sqrt(0.8) ≈ 0.894`.

Both metrics also induce the **same ranking**, which is what §7.3's `argmax
sim` actually needs — the tie-break is well-defined under either, *provided the
vectors are normalized*.

**Normalization is the entire load-bearing condition.** Without it the identity
fails and the two signals genuinely disagree.

---

## 3. What was verified, and how

Two independent checks, because the algebra and the ANN implementation can fail
differently.

**Exact / algebraic** —
`tests/test_dutsx_retrieval.py::test_l2_theta_dis_equivalent_to_cosine_sigma_exact_brute_force`.
Brute-force over real santos vectors: the `θ_dis` threshold set and the `σ`
threshold set are identical. **Max absolute error 2.24e-07, zero mismatched
threshold sets across 156 queries.** This validates the identity itself.

**Through two real indexes** —
`tests/test_dutsx_retrieval.py::test_l2_vs_cosine_hnsw_indexes_agree_closely`.
Two independently-built hnswlib indexes — `space='l2'` over explicitly
normalized vectors vs `space='cosine'` over raw vectors — queried the same way.
**Mean Jaccard 0.9953, min 0.9608 over 274 queries.** Below 1.0 purely because
HNSW is approximate; this isolates ANN noise from the algebra, which is why
both tests exist rather than just one.

---

## 4. What actually ships, and why it differs from the original plan

`dutsx/adapters/semantic.py::HnswRetriever` uses **`hnswlib.Index(space='cosine')`
over the raw, unnormalized vectors**, and thresholds on `sim = 1 − distance`.

**No explicit L2-normalization step, and no `θ_dis` conversion in the adapter.**
hnswlib's cosine space normalizes internally and returns `1 − cos_sim`
directly, so the conversion is unnecessary — it is only needed when indexing
with `space='l2'`.

This corrects `PLAN-full.md` I3, which prescribed "L2-normalize every attribute
embedding at index time" *and* "build hnswlib with `space='cosine'`". Those are
two different code paths, and the plan asserted both without noticing the
tension. Normalizing before handing vectors to a cosine-space index is
redundant work, not a safeguard.

Note the library is **vanilla upstream `hnswlib` 0.8.0**
(`github.com/yurymalkov/hnsw`). `PLAN-full.md`'s asset table previously claimed
a built `hnsw_fair/` fork was "Ready"; no such directory exists, and no fork is
needed — corrected there.

---

## 5. What the paper should say

The implementation is now consistent; **the paper is still self-inconsistent**
and needs an edit. Two options:

**Option A — normalize, keep both framings (minimal edit).** Add one sentence
to §7.1: *"Attribute embeddings are L2-normalized at index time, under which
Euclidean distance and cosine similarity are monotonically equivalent
(`‖a−b‖² = 2 − 2·cos(a,b)`); we therefore state thresholds interchangeably as
`θ_dis = sqrt(2 − 2σ)`."* Table 1's circular definition then resolves, and
§7.3's tie-break becomes well-defined.

**Option B — state cosine throughout (matches what ships).** Drop the Euclidean
framing from §7.1 and say the index is queried in cosine space at the same `σ`
used by §6.2. Fewer moving parts, one metric in the whole paper, and it
describes the actual implementation without needing the equivalence argument at
all.

**Recommendation: Option B**, with the identity from §2 kept as a one-line
footnote so a reader who expects the usual Euclidean-ANN framing sees why
cosine loses nothing. Option A is defensible if there is a reason to keep the
Euclidean wording for consistency with prior work, but it documents a
normalization step the code does not perform.

Either way, **Table 1's `sim(A, V_D)` should stop deferring to the index** and
state the metric directly.

---

## 6. Cross-references

- `PLAN-full.md` I3 — original correction; superseded here on the normalization step.
- `PLAN-integration.md` Phase B — acceptance criteria and measured numbers.
- `dutsx/adapters/semantic.py` — `HnswRetriever` (shipped), `ExactScanRetriever` (oracle).
- `tests/test_dutsx_retrieval.py` — the three tests named above.
- `NOTES.md` — Phase B entry.
