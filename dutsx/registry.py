"""Config-driven adapter selection (PLAN-integration.md §5).

    {"adapters": {"synopsis": "metadata_store", ...}}

Swapping an implementation is a config string change plus one new adapter
class registered here, never a change to any driver code that consumes the
port.
"""
from typing import Any, Callable, Dict

from .adapters.overlap import InvertedIndexOverlap, NullOverlap
from .adapters.semantic import ExactScanRetriever, HnswRetriever
from .adapters.synopsis import CsvSynopsis, MetadataStoreSynopsis
from .adapters.unionability import (ConstantScorer, PinnedMatchScorer,
                                     StarmieVerifyScorer)

AdapterRegistry = Dict[str, Callable[..., Any]]

SYNOPSIS_REGISTRY: AdapterRegistry = {
    "metadata_store": MetadataStoreSynopsis,
    "csv": CsvSynopsis,
}

# Phase B. "hnsw" is the shipped retriever (space='cosine'); "exact_scan" is
# the brute-force recall oracle, not intended to be run at benchmark scale.
SEMANTIC_REGISTRY: AdapterRegistry = {
    "hnsw": HnswRetriever,
    "exact_scan": ExactScanRetriever,
}

# Phase B. "inverted_index" is the paper's own §7.2 design (v7: posting
# lists, exact boolean overlap -- not a JOSIE stand-in, see
# adapters/overlap.py); "null" ablates the overlap filter's contribution
# entirely.
OVERLAP_REGISTRY: AdapterRegistry = {
    "inverted_index": InvertedIndexOverlap,
    "null": NullOverlap,
}

# Phase C. "pinned_match" is the §6.2 spec (M3); "starmie_verify" pins only
# the row and is kept for comparability, not correctness -- see
# dutsx/adapters/unionability.py.
UNIONABILITY_REGISTRY: AdapterRegistry = {
    "pinned_match": PinnedMatchScorer,
    "starmie_verify": StarmieVerifyScorer,
    "constant": ConstantScorer,
}

_PORT_REGISTRIES: Dict[str, AdapterRegistry] = {
    "synopsis": SYNOPSIS_REGISTRY,
    "semantic": SEMANTIC_REGISTRY,
    "overlap": OVERLAP_REGISTRY,
    "unionability": UNIONABILITY_REGISTRY,
}


def build(port: str, name: str, **kwargs: Any) -> Any:
    """Instantiate the adapter registered as ``name`` for ``port``.

    ``kwargs`` are passed straight through to the adapter's constructor
    (e.g. ``pkl_path=`` for ``MetadataStoreSynopsis``, ``table_dirs=`` for
    ``CsvSynopsis``) -- the registry only resolves which class to build, it
    does not know or validate per-adapter constructor arguments.
    """
    try:
        registry = _PORT_REGISTRIES[port]
    except KeyError:
        raise KeyError(f"unknown port {port!r}; known ports: {sorted(_PORT_REGISTRIES)}")
    try:
        adapter_cls = registry[name]
    except KeyError:
        raise KeyError(
            f"unknown adapter {name!r} for port {port!r}; "
            f"known adapters: {sorted(registry)}"
        )
    return adapter_cls(**kwargs)


def build_from_config(cfg: Dict, port: str, **kwargs: Any) -> Any:
    """Look up ``cfg['adapters'][port]`` and build it -- the config shape in
    PLAN-integration.md §5."""
    name = cfg["adapters"][port]
    return build(port, name, **kwargs)
