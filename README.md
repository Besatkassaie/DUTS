# DUTS — Distribution-Aware Unionable Table Search

Code for *Distribution-Aware Unionable Table Search* (Kassaie & Miller). `duts/` is the two-stage
optimization core (Stage 1: Dinkelbach pool selection; Stage 2: exact 0–1 ILP via HiGHS),
`dutsx/` the retrieval / synopsis / unionability adapters, `experiments/` the experiment harness.

## Setup

### 1. Python environment

```bash
conda env create -f environment.yml && conda activate duts     # Python 3.8, pinned versions
# or: pip install -r requirements.txt  (into a Python 3.8 environment)
```

`requirements.txt` installs the CUDA 12.1 build of PyTorch. **No GPU?** Install the CPU build first;
the pinned `torch==2.4.1` is then already satisfied and pip keeps it:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

PyTorch (and `transformers`, `tokenizers`, `scikit-learn`, `mlflow`, `xgboost`) is only needed to
*generate* embeddings. If you use precomputed embeddings, the core block of `requirements.txt`
(numpy, scipy, pandas, hnswlib, networkx, munkres) is enough to run DUTS and the baselines.

### 2. Starmie code

The Starmie baselines, the value-distribution synopsis format (`TableMetadata.py`) and embedding
generation (`sdd/`) come from a Starmie checkout. It is only read, never written.

> **TODO:** where to get the Starmie code — e.g.
> `git clone <STARMIE_REPO_URL> starmie_fair` (commit `<COMMIT>`).

Tell the program where it is, once per shell:

```bash
export STARMIE_FAIR_ROOT=/path/to/starmie_fair      # or pass --starmie-root on every run
```

⚠️ Without either, the default is the authors' own path (`/u6/bkassaie/starmie_fair`), which will
not exist on your machine; the prerequisite check then reports "Starmie checkout" as missing.

### 3. Benchmarks

> **TODO:** where to download the benchmarks (`santos3`, `tusSmall3`, `tusLarge3`) —
> `<BENCHMARK_DOWNLOAD_URL>` — and whether the download includes the precomputed embeddings
> (`vectors/`) and synopses (`indexes/metadata_combined.pkl`).

Each dataset is a directory laid out as:

```
<dataset>/
├── datalake/*.csv                              # datalake tables
├── query/*.csv                                 # query tables
├── <dataset>_benchmark_groundtruth.csv         # or <dataset>_small_benchmark_groundtruth.csv
├── vectors/cl_{datalake,query}_drop_col_tfidf_entity_column_0.pkl   # embeddings (can be generated)
└── indexes/metadata_combined.pkl               # synopsis (can be generated)
```

plus a query list `protected_attributes.csv` inside it, or `protected_attributes_<dataset>.csv` next
to it, with columns `q_name, protected_attribute_id, protected_value`. If the datasets live in
`$STARMIE_FAIR_ROOT/data/<dataset>`, the program suggests every path itself.

### 4. Model checkpoint (only to generate embeddings)

> **TODO:** where to get the trained Starmie checkpoint used for the reported results —
> `<CHECKPOINT_URL>` (`model_drop_col_tfidf_entity_column_0.pt`, trained on santos) — or how to
> train one with Starmie's `run_pretrain.py`.

When embeddings are missing, the program asks for this file (or pass `--checkpoint`). If it is at
`$STARMIE_FAIR_ROOT/results/santos/model_drop_col_tfidf_entity_column_0.pt` it is suggested
automatically.

## Running an experiment

```bash
python main.py
```

With no arguments the program asks for everything it needs and suggests a value in brackets for
each (Enter accepts it). Every parameter can also be passed on the command line; anything missing is
prompted for. `python main.py --help` lists all of them.

```bash
python main.py --system duts --dataset santos3 \
    --dataset-path $STARMIE_FAIR_ROOT/data/santos3 \
    --index-path artifacts/santos3/index \
    --embedding-path $STARMIE_FAIR_ROOT/data/santos3/vectors \
    --k 10 --alpha 5 --defaults
```

| | |
|---|---|
| `--system` | `duts`, `starmie` (top-k, no fairness), `starmie_exhaustive` (exhaustive swap), `starmie_nl` (NL swap) |
| `--dataset` | name, e.g. `santos3`, `tusSmall3`, `tusLarge3` |
| `--dataset-path` | directory with `datalake/` and `query/` (CSV tables) and the groundtruth CSV |
| `--index-path` | where the DUTS HNSW index (and, if built here, the synopsis) lives |
| `--embedding-path` | directory with `cl_datalake_…pkl` / `cl_query_…pkl` column embeddings |
| method parameters | `--k --f-star --delta --sigma`; DUTS: `--alpha --top-n --theta-cat`; baselines: `--n-columns --workers` |
| `--defaults` | use defaults for method parameters instead of prompting |
| `--non-interactive` | never prompt (fail on a missing required parameter) |
| `--limit-queries N` | first N queries only (smoke test) |
| `--starmie-root` | Starmie checkout (default: `$STARMIE_FAIR_ROOT`) |
| `--output-dir` | where result files go (default `experiments/results/main/`) |

Options used only when a prerequisite has to be built (asked for interactively otherwise):

| | |
|---|---|
| `--build-missing` | build missing prerequisites without asking for confirmation |
| `--checkpoint PATH` | trained Starmie model checkpoint (`.pt`) for generating embeddings |
| `--embed-batch-size N` | tables per inference batch when generating embeddings (default 1024; lower it if GPU memory runs out) |
| `--hnsw-m M` | HNSW graph degree for the DUTS index (default 32) |
| `--hnsw-ef-construction N` | HNSW build-time search width (default 200) |

The query list (`protected_attributes.csv` in the dataset directory, or
`../protected_attributes_<dataset>.csv`), the groundtruth and the synopsis are located
automatically and can be overridden with `--protected-csv`, `--groundtruth-csv` and
`--metadata-path`.

### Prerequisites are checked first

Before anything runs, every required input and artifact is checked and listed:

* **Input data** (tables, query list, groundtruth, Starmie checkout) cannot be generated; a missing
  one stops the run with an explanation of what to supply.
* **Derived artifacts** are built on request, then you are asked to rerun (the exact command is
  printed):
  * column **embeddings** — Starmie model inference over every table (asks for a trained checkpoint,
    `--checkpoint`; uses a GPU when available);
  * the **value-distribution synopsis** (`metadata_combined.pkl`, one histogram per column);
  * the **DUTS HNSW index** (asks for `M` / `ef_construction`); an index built from different
    embeddings or synopsis is detected as stale and rebuilt.

Building asks for confirmation; `--build-missing` skips the question (and is required when running
non-interactively).

Exit codes: `0` done · `2` bad configuration · `3` prerequisites built, rerun · `4` a prerequisite
is missing and was not built · `130` interrupted.

### Walkthrough: a first run

A first DUTS run on santos3, with the benchmark under `$STARMIE_FAIR_ROOT/data/santos3` and no DUTS
index built yet (output shortened):

```text
$ python main.py --defaults
Enter a value, or press Enter to accept the [suggestion].
  system/method to run (duts/starmie/starmie_exhaustive/starmie_nl): duts
  dataset (benchmark) name, e.g. santos3, tusSmall3, ...: santos3
  dataset directory holding datalake/ and query/ [/path/to/starmie_fair/data/santos3]:
  index directory (DUTS HNSW index; metadata store if built here) [/path/to/DUTS/artifacts/santos3/index]:
  embedding directory holding the Starmie column-vector pickles [/path/to/starmie_fair/data/santos3/vectors]:

prerequisites
  [ok] Starmie checkout                             /path/to/starmie_fair
  [ok] datalake tables                              .../santos3/datalake (999 CSV files)
  [ok] query tables                                 .../santos3/query (48 CSV files)
  [ok] query list (protected attributes)            .../protected_attributes_santos3.csv
  [ok] groundtruth                                  .../santos3_small_benchmark_groundtruth.csv
  [ok] column embeddings                            .../santos3/vectors/cl_datalake_...pkl + cl_query_...pkl
  [ok] value-distribution synopsis (MetadataStore)  .../santos3/indexes/metadata_combined.pkl
  [--] DUTS HNSW index                              missing: .../artifacts/santos3/index/santos3_duts_hnsw_m32_tc50_sig06.bin

The following prerequisites must be built before duts can run:
  - DUTS HNSW index: missing: ...
The DUTS HNSW index needs:
  HNSW graph degree M [32]:
  HNSW ef_construction [200]:

Build them now? [Y/n]
building HNSW (M=32, ef_construction=200) over 999 tables ...

Prerequisites are ready. Rerun the experiment with:
  python main.py --system duts --dataset santos3 --dataset-path ... --k 10 ... --output-dir ...
```

The program exits with code 3. Run the printed command; it has every parameter filled in, so it
does not prompt again:

```text
$ python main.py --system duts --dataset santos3 --dataset-path ... --output-dir ...
...
running duts on santos3 ...
setup 2.2s: 999 datalake tables indexed, 48 queries (48 skipped)

========================================================================
RESULTS  system=duts  dataset=santos3
========================================================================
  parameters        k=10  f_star=0.4  delta=0.1  sigma=0.6  alpha=5.0  top_n=1000  theta_cat=50  (tau=0.300)
  queries           48 evaluated, 48 skipped (not in query dir / no embedding)
  feasible          39 / 48  (81.2%)
    infeasible: insufficient_candidates      7
    infeasible: stage2_infeasible            2
  precision         0.8104   (mean over all queries; infeasible = 0)
  recall            0.3005   (ideal recall@k = 0.4736)
  ...
  avg per query     0.0260s

========================================================================
EXECUTION ENVIRONMENT
========================================================================
  python            3.8.5 ...
```

"48 skipped" is expected on santos3: the query list also names the 48 non-fair originals of the
queries, which are not in `query/`.

If embeddings or the synopsis are missing too, all of them are built in the same step: embeddings
first (you are asked for the checkpoint), then the synopsis, then the index. Embedding generation
is the slow part on large datasets; the rest takes seconds to minutes.

**Non-interactive (scripts, batch jobs):** pass every core parameter, add `--defaults
--non-interactive`, and add `--build-missing` to the first run so missing artifacts are built
instead of reported (exit code 4).

### Output

At the end of a run the terminal shows precision, recall (with the ideal recall@k), the number and
percentage of queries with a feasible result (and why the others failed), total runtime and average
runtime per query, followed by the execution environment (Python, OS, package versions, MILP solver,
Java, CUDA and GPUs, git revisions). Precision and recall are means over all queries, with a query
that has no feasible result counting 0; the same metrics over whatever was returned are shown on a
second line.

The per-query results (`<dataset>_<system>_k…_queries.csv`) and a JSON summary with the full
configuration and environment are written to `--output-dir` (default `experiments/results/main/`).

## Tests

```bash
python -m pytest -q
```

Further experiment scripts (parameter sweeps, ablations) are described in `experiments/README.md`.
