"""Auto-select WDC query tables and protected values, biased toward legitimate pools.

WDC has no query/datalake split and no ``protected_attributes_<benchmark>.csv`` (unlike the
santos-family benchmarks), so query tasks have to be constructed algorithmically rather than
loaded from a curated file.

**The retrieval-yield problem this exists to address.** ``D`` (dutsx.runner.retrieve_unscored_
candidates) is built by intersecting semantic top-N (HNSW) with ``OverlapFilter.query(M)`` -- a
candidate must be BOTH schematically similar to the query AND literally contain a value from ``M``.
On santos-family data this compound condition is survivable because the datalake is curated to be
topically related to queries. On WDC, an uncurated, topically heterogeneous corpus, picking ``M``
uniformly at random from the query attribute's observed domain risks an empty or near-empty ``D``
for most queries -- the pipeline correctly reports ``insufficient_candidates`` and returns ``∅``,
but a query set dominated by that outcome never actually exercises Stage 1/Stage 2, defeating the
point of a scalability study.

The fix: rank a candidate attribute's observed values by their ``OverlapFilter`` posting-list size
(how many distinct tables contain that value, corpus-wide) and prefer values clearing
``min_posting_size`` before finalizing ``M``, rather than picking uniformly at random. To honestly
quantify how much this matters (not just assume it), also build a small unbiased side-sample for
comparison -- see ``select_queries_with_comparison``.
"""
import random
import statistics
from typing import Dict, List, Optional, Set, Tuple

from dutsx.runner import QueryTask, RunnerContext, retrieve_unscored_candidates


def _posting_size(overlap, value: str) -> int:
    """Number of DISTINCT TABLES (not (table, attr) pairs) containing ``value``,
    via the existing public ``OverlapFilter.query`` -- no adapter-specific
    method needed, works for any ``OverlapFilter`` implementation."""
    pairs = overlap.query({value})
    return len({table for table, _attr in pairs.keys()})


def _select_M(
    domain: List[str], overlap, m_size_range: Tuple[int, int], rng: random.Random,
    bias_by_overlap: bool, min_posting_size: int,
) -> Tuple[Set[str], bool]:
    """-> (M, met_threshold). ``met_threshold`` is False when biasing was
    requested but no candidate value cleared ``min_posting_size`` -- the
    single best-available value is used anyway (never returns an empty M),
    but the shortfall is reported rather than hidden."""
    m_size = rng.randint(*m_size_range)
    m_size = max(1, min(m_size, len(domain)))

    if not bias_by_overlap:
        return set(rng.sample(domain, m_size)), True

    scored = sorted(
        ((v, _posting_size(overlap, v)) for v in domain),
        key=lambda vp: vp[1], reverse=True,
    )
    qualifying = [v for v, size in scored if size >= min_posting_size]
    if len(qualifying) >= m_size:
        return set(qualifying[:m_size]), True
    # not enough values clear the threshold -- take what's available (best-first),
    # never return an empty M, but flag the shortfall
    return set(v for v, _ in scored[:m_size]), False


def _select_one(
    q_table: str, synopsis, overlap, rng: random.Random,
    m_size_range: Tuple[int, int], bias_by_overlap: bool, min_posting_size: int,
    k: int, alpha: float, F_star: float, delta: float, include_query: bool, top_n: int,
) -> Tuple[Optional[QueryTask], Optional[dict]]:
    cat_attrs = synopsis.categorical_attrs(q_table)
    if not cat_attrs:
        return None, None
    attr = rng.choice(cat_attrs)
    dist = synopsis.distribution(q_table, attr)
    domain = [v for v in dist if v != ""]
    if not domain:
        return None, None

    M, met_threshold = _select_M(domain, overlap, m_size_range, rng, bias_by_overlap, min_posting_size)
    task = QueryTask(
        q_table=q_table, attr=attr, M=M, F_star=F_star, delta=delta, k=k, alpha=alpha,
        include_query=include_query, top_n=top_n,
    )
    meta = {
        "q_table": q_table, "attr": attr, "domain_size": len(domain),
        "m_size": len(M), "met_posting_threshold": met_threshold,
    }
    return task, meta


def _yield_stats(sizes: List[int], k: int) -> dict:
    if not sizes:
        return {"n": 0, "median": None, "min": None, "max": None, "frac_below_k": None}
    return {
        "n": len(sizes),
        "median": statistics.median(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "frac_below_k": sum(1 for s in sizes if s < k) / len(sizes),
    }


def auto_select_queries(
    ctx: RunnerContext,
    table_ids: List[str],
    n_queries: int,
    seed: int = 42,
    m_size_range: Tuple[int, int] = (1, 3),
    k: int = 10,
    alpha: float = 2.0,
    F_star: float = 0.3,
    delta: float = 0.15,
    include_query: bool = True,
    top_n: int = 100,
    min_posting_size: Optional[int] = None,
    bias_by_overlap: bool = True,
    measure_yield: bool = True,
) -> Tuple[List[QueryTask], dict]:
    """Auto-select ``n_queries`` query tasks from ``table_ids`` (the tier's
    table universe). Requires ``ctx.query_vectors`` to already cover
    ``table_ids`` (real embeddings, Phase 3's output) when
    ``measure_yield=True``, since yield is measured via the real
    ``retrieve_unscored_candidates`` path, not an overlap-only proxy.

    Returns ``(tasks, selection_report)``.
    """
    min_posting_size = min_posting_size if min_posting_size is not None else 2 * top_n
    rng = random.Random(seed)

    synopsis, overlap = ctx.synopsis, ctx.overlap
    eligible = [t for t in table_ids if synopsis.categorical_attrs(t)]
    rng.shuffle(eligible)

    tasks: List[QueryTask] = []
    metas: List[dict] = []
    for q_table in eligible:
        if len(tasks) >= n_queries:
            break
        task, meta = _select_one(
            q_table, synopsis, overlap, rng, m_size_range, bias_by_overlap,
            min_posting_size, k, alpha, F_star, delta, include_query, top_n,
        )
        if task is None:
            continue
        tasks.append(task)
        metas.append(meta)

    d_sizes = []
    if measure_yield:
        for task in tasks:
            D, _N_Q, _n_Q, _telemetry = retrieve_unscored_candidates(task, ctx)
            d_sizes.append(len(D))
        for meta, size in zip(metas, d_sizes):
            meta["n_D"] = size

    report = {
        "n_table_universe": len(table_ids),
        "n_eligible": len(eligible),
        "eligibility_rate": len(eligible) / len(table_ids) if table_ids else 0.0,
        "n_queries_selected": len(tasks),
        "domain_size_distribution": [m["domain_size"] for m in metas],
        "m_size_distribution": [m["m_size"] for m in metas],
        "n_met_posting_threshold": sum(1 for m in metas if m.get("met_posting_threshold")),
        "bias_by_overlap": bias_by_overlap,
        "min_posting_size": min_posting_size,
        "seed": seed,
        "queries": metas,
        "yield": _yield_stats(d_sizes, k) if measure_yield else None,
    }
    return tasks, report


def select_queries_with_comparison(
    ctx: RunnerContext,
    table_ids: List[str],
    n_queries: int,
    seed: int = 42,
    m_size_range: Tuple[int, int] = (1, 3),
    min_posting_size: Optional[int] = None,
    k: int = 10,
    alpha: float = 2.0,
    F_star: float = 0.3,
    delta: float = 0.15,
    include_query: bool = True,
    top_n: int = 100,
) -> Tuple[List[QueryTask], dict]:
    """Build the main frequency-biased query set, and for the SAME
    (q_table, attr) selections, ALSO compute what unbiased (uniform-random)
    ``M`` selection would have yielded -- a paired comparison, not two
    independent samples.

    **Corrected during implementation** from an earlier, confounded design
    that drew the biased and unbiased samples from independently-random
    table subsets (different seeds). That version measured the SAME
    aggregate |D| distribution regardless of biasing (both medians tied at
    1.0 on a real tier_10k run) -- not because biasing doesn't work, but
    because which query TABLE gets selected dominates the aggregate
    statistic at this scale, masking the M-selection effect entirely. A
    row-by-row paired check (same table+attr, only M differs) showed
    biasing IS doing real work -- e.g. one pair picked M="6" (|D|=12) vs
    M="13 Rock Art Hot Spot" (|D|=3) for the identical query table/attr.
    This function reproduces that paired check directly, not the flawed
    aggregate-only version.
    """
    min_posting_size = min_posting_size if min_posting_size is not None else 2 * top_n
    rng = random.Random(seed)

    synopsis, overlap = ctx.synopsis, ctx.overlap
    eligible = [t for t in table_ids if synopsis.categorical_attrs(t)]
    rng.shuffle(eligible)

    tasks: List[QueryTask] = []
    metas: List[dict] = []
    biased_d, unbiased_d = [], []

    for q_table in eligible:
        if len(tasks) >= n_queries:
            break
        cat_attrs = synopsis.categorical_attrs(q_table)
        attr = rng.choice(cat_attrs)
        dist = synopsis.distribution(q_table, attr)
        domain = [v for v in dist if v != ""]
        if not domain:
            continue

        # same rng STATE forked for both draws, so a fixed |M| size and any
        # other randomness downstream matches between the two -- only the
        # bias_by_overlap flag differs.
        m_size = rng.randint(*m_size_range)
        rng_b = random.Random(rng.random())
        rng_u = random.Random(rng.random())
        M_biased, met = _select_M(domain, overlap, (m_size, m_size), rng_b, True, min_posting_size)
        M_unbiased, _ = _select_M(domain, overlap, (m_size, m_size), rng_u, False, min_posting_size)

        task_biased = QueryTask(
            q_table=q_table, attr=attr, M=M_biased, F_star=F_star, delta=delta, k=k, alpha=alpha,
            include_query=include_query, top_n=top_n,
        )
        task_unbiased = QueryTask(
            q_table=q_table, attr=attr, M=M_unbiased, F_star=F_star, delta=delta, k=k, alpha=alpha,
            include_query=include_query, top_n=top_n,
        )
        D_b, _, _, _ = retrieve_unscored_candidates(task_biased, ctx)
        D_u, _, _, _ = retrieve_unscored_candidates(task_unbiased, ctx)

        tasks.append(task_biased)  # only the biased task feeds the main timing sweep
        biased_d.append(len(D_b))
        unbiased_d.append(len(D_u))
        metas.append({
            "q_table": q_table, "attr": attr, "domain_size": len(domain),
            "m_size": len(M_biased), "met_posting_threshold": met,
            "n_D_biased": len(D_b), "n_D_unbiased": len(D_u),
        })

    report = {
        "n_table_universe": len(table_ids),
        "n_eligible": len(eligible),
        "eligibility_rate": len(eligible) / len(table_ids) if table_ids else 0.0,
        "n_queries_selected": len(tasks),
        "domain_size_distribution": [m["domain_size"] for m in metas],
        "m_size_distribution": [m["m_size"] for m in metas],
        "n_met_posting_threshold": sum(1 for m in metas if m.get("met_posting_threshold")),
        "min_posting_size": min_posting_size,
        "seed": seed,
        "queries": metas,
        "yield": _yield_stats(biased_d, k),
        "unbiased_side_sample": {
            "n_queries_selected": len(unbiased_d),
            "yield": _yield_stats(unbiased_d, k),
        },
        "paired_biased_minus_unbiased_mean": (
            sum(b - u for b, u in zip(biased_d, unbiased_d)) / len(biased_d)
            if biased_d else None
        ),
    }
    return tasks, report
