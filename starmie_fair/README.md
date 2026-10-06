# Starmie extensions used by DUTS

The fairness-aware Starmie code that DUTS needs, bundled so that cloning this repository is enough to
run DUTS and the Table 3 baselines. It extends Starmie (https://github.com/megagonlabs/starmie) and is
not part of the public Starmie repository.

| Module | Used for |
|---|---|
| `TableMetadata.py` | `MetadataStore`: the per-column value histograms (`metadata_combined.pkl`). DUTS needs it to read the synopses shipped with the benchmarks, which are pickled instances of these classes. |
| `HNSWSearcher_Fair.py` | Starmie search with the distribution attribute aligned; the `starmie`, `starmie_exhaustive` (Greedy-Swap) and `starmie_nl` (Preference-Swap) baselines |
| `exhaustive_swap.py`, `nl_swap.py`, `bnl_swap.py`, `preference.py`, `preference_config.json` | Swap-based distributional repair (paper §5) and preference-based candidate filtering |
| `bounds.py` | Constrained bipartite attribute alignment (modified from Starmie's `bounds.py`) |
| `utility.py`, `Custom_Heap.py` | Helpers |

Changes relative to the research checkout these were copied from: `HNSWSearcher_Fair` accepts
`index_path=None` and then keeps its HNSW index in memory instead of saving it to disk.

Starmie's own model code (`sdd/`) is **not** bundled. It is needed only to regenerate column
embeddings (the benchmarks ship them); for that, clone the public Starmie repository and pass it with
`--starmie-root`.
