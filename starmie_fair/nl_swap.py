"""
Nested Loops (NL) fairification strategy.

Based on Chomicki (TODS 2003) Section 4.1 — the NL algorithm for evaluating
the winnow operator with configurable preference relations.

NL is correct for any preference relation — it does not require the relation
to be a strict partial order.
"""

import time
from bounds import verify_constrained
from utility import (
    round_fairness_f,
    round_fairness_delta,
    compute_fairness_delta,
    fairness_delta_exceeds,
    fairness_delta_within,
)


def _resolve_candidate(searcher, cand_name, cand_score, qScore, threshold, p_id, matched_columns_map):
    """
    Resolve a candidate table: if already verified (cand_score != -100), return as-is.
    If not verified (cand_score == -100), run verify_constrained to get the real score
    and matched columns.

    Returns:
        (score, matched_columns) or None if the candidate doesn't match the protected column.
    """
    if cand_score != -100:
        return cand_score, matched_columns_map.get(cand_name, [])

    if qScore is None:
        return None

    tScore = None
    for table in searcher.tables:
        if table[0] == cand_name:
            tScore = table[1]
            break
    if tScore is None:
        return None

    verified_score, cand_matched_columns = verify_constrained(qScore, tScore, threshold, p_id)

    matched_with = [col for row, col in cand_matched_columns if row == p_id]
    if not matched_with:
        return None

    return verified_score, cand_matched_columns


def _enrich_candidates(searcher, all_candidates, p_id, protected_value, matched_columns_map):
    """
    Enrich candidate tables with metadata attributes for preference-based dominance.

    For each candidate, looks up its metadata to compute:
    - unionability_score: the verified bipartite matching score (or 0.0 if not verified)
    - protected_count: number of protected records in the aligned column
    - non_protected_count: total records minus protected records
    - total_records: total number of records in the table
    - protected_proportion: protected_count / total_records

    Args:
        searcher: HNSWSearcher_Fair instance
        all_candidates: list of (cand_name, cand_score, source) tuples
        p_id: protected column id in query
        protected_value: the protected attribute value (e.g. "Female")
        matched_columns_map: dict mapping table_name to matched column pairs

    Returns:
        list of dicts with metadata attributes per candidate
    """
    enriched = []
    for cand_name, cand_score, source in all_candidates:
        attrs = {
            'name': cand_name,
            'score': cand_score,
            'source': source,
            'unionability_score': cand_score if cand_score != -100 else 0.0,
            'protected_count': 0,
            'non_protected_count': 0,
            'total_records': 0,
            'protected_proportion': 0.0,
        }

        metadata = searcher.metadata_store.get_metadata(cand_name) if hasattr(searcher, 'metadata_store') else None
        if metadata:
            aligned_col = p_id
            if matched_columns_map and isinstance(matched_columns_map, dict):
                mc = matched_columns_map.get(cand_name)
                if mc:
                    for qcol, tcol in mc:
                        if qcol == p_id:
                            aligned_col = tcol
                            break

            if isinstance(aligned_col, int) and aligned_col < len(metadata.column_names):
                col_name = metadata.column_names[aligned_col]
            else:
                col_name = aligned_col

            if metadata.is_categorical(col_name):
                prot = metadata.get_category_count(col_name, protected_value)
                total = metadata.num_records
                attrs['protected_count'] = prot
                attrs['total_records'] = total
                attrs['non_protected_count'] = total - prot
                attrs['protected_proportion'] = prot / total if total > 0 else 0.0

        enriched.append(attrs)
    return enriched


def nl_swap(searcher, scores, violate_list, verified_list, not_verified_list,
            query_table_name, p_id, protected_value, matched_columns_map=None,
            qScore=None, threshold=0.6):
    """
    Nested Loops (NL) fairification algorithm.

    For each violating table in top-K, compares it against ALL candidates using
    the NL approach: each candidate is checked against every other candidate to
    determine if it is dominated (per preference_config). Only non-dominated
    candidates (the "winnow" set) are considered for swaps.

    NL is correct for any preference relation — it does not require the relation
    to be a strict partial order.

    Complexity: O(|violators| * |candidates|^2) dominance checks in the worst case.

    Args:
        searcher: HNSWSearcher_Fair instance
        scores: list of (score, table_name) tuples - current top-K results
        violate_list: list of (table_name, unionability_score, delta) - violating tables
        verified_list: list of (table_name, unionability_score, delta) - verified candidates
        not_verified_list: list of (table_name, unionability_score, delta) - unverified candidates
        query_table_name: name of the query table
        p_id: protected column id
        protected_value: the protected attribute value
        matched_columns_map: dict mapping table_name to matched columns
        qScore: query score vectors (needed for verify_constrained)
        threshold: similarity threshold for column matching

    Returns:
        dict with 'sorted_results', 'delta', 'swaps_made', 'success', 'time_seconds',
             'dominance_checks', 'candidates_after_winnow'
    """
    start_time = time.time()

    current_scores = list(scores)
    current_table_names = [name for _, name in current_scores]

    if matched_columns_map is None:
        matched_columns_map = {}

    # Build the full candidate pool from verified + not_verified
    all_candidates = []
    for cand_name, cand_score, cand_delta in verified_list:
        all_candidates.append((cand_name, cand_score, 'verified'))
    for cand_name, cand_score, cand_delta in not_verified_list:
        all_candidates.append((cand_name, cand_score, 'not-verified'))

    # Enrich candidates with metadata attributes for preference comparison
    enriched = _enrich_candidates(searcher, all_candidates, p_id, protected_value, matched_columns_map)

    # Identify violating table names
    violating_names = set()
    for item in violate_list:
        if len(item) == 2:
            violating_names.add(item[1])
        else:
            violating_names.add(item[0])

    swaps_made = 0
    dominance_checks = 0

    # Compute initial delta
    current_F_result = searcher.compute_F(
        current_table_names, p_id, protected_value,
        matched_columns_map=matched_columns_map, include_query=True,
        query_table_name=query_table_name
    )
    current_F = round_fairness_f(current_F_result['protected_proportion'] if current_F_result else 0.0)
    current_delta = compute_fairness_delta(searcher.target_fairness, current_F)

    print(f"[NL] Starting. Initial delta: {current_delta:.4f}, Target: <= {searcher.delta:.4f}")
    print(f"[NL] Violating tables: {len(violating_names)}, Candidates: {len(all_candidates)}")
    print(f"[NL] Preference config: {searcher.preference_config}")

    # =========================================================================
    # NL winnow: for each candidate, check if dominated by any other
    # candidate. Only non-dominated candidates survive. (Chomicki NL)
    # =========================================================================
    active_indices = set(range(len(enriched)))
    active_indices = {i for i in active_indices if enriched[i]['name'] not in current_table_names}

    non_dominated = []
    dominated_candidate_names = set()
    for i in list(active_indices):
        dominated = False
        for j in active_indices:
            if i == j:
                continue
            dominance_checks += 1
            if searcher.preference_config.dominates(enriched[j], enriched[i]):
                dominated = True
                dominated_candidate_names.add(enriched[i]['name'])
                break
        if not dominated:
            non_dominated.append(i)
    dominated_removed_count = len(dominated_candidate_names)
    dominance_candidate_pool_size = len(active_indices)

    print(f"[NL] Winnowed {len(all_candidates)} candidates to {len(non_dominated)} non-dominated "
          f"({dominance_checks} dominance checks)")

    # =========================================================================
    # Swap loop: try non-dominated candidates for each violator
    # =========================================================================
    made_progress = True
    while made_progress and fairness_delta_exceeds(current_delta, searcher.delta):
        made_progress = False

        for score_val, table_name in list(current_scores):
            if table_name not in violating_names:
                continue
            if table_name not in current_table_names:
                continue

            best_cand = None
            best_F_value = -1

            for nd_idx in list(non_dominated):
                e = enriched[nd_idx]
                cand_name = e['name']
                cand_score = e['score']
                source = e['source']

                if cand_name in current_table_names:
                    continue

                resolved = _resolve_candidate(
                    searcher, cand_name, cand_score, qScore, threshold, p_id, matched_columns_map
                )
                if resolved is None:
                    continue
                real_score, cand_matched_cols = resolved

                temp_map = dict(matched_columns_map)
                if source == 'not-verified':
                    temp_map[cand_name] = cand_matched_cols

                new_table_names = [n for n in current_table_names if n != table_name] + [cand_name]
                cand_result = searcher.compute_F(
                    new_table_names, p_id, protected_value,
                    matched_columns_map=temp_map, include_query=True,
                    query_table_name=query_table_name
                )
                if cand_result is None:
                    continue

                cand_F_value = round_fairness_f(cand_result['protected_proportion'])
                new_delta = compute_fairness_delta(searcher.target_fairness, cand_F_value)
                if (
                    round_fairness_delta(new_delta) < round_fairness_delta(current_delta)
                    and cand_F_value > best_F_value
                ):
                    best_cand = {
                        'cand_name': cand_name,
                        'cand_score': real_score,
                        'best_F_value': cand_F_value,
                        'nd_idx': nd_idx,
                        'source': source,
                        'matched_columns': cand_matched_cols if source == 'not-verified' else None
                    }
                    best_F_value = cand_F_value

            if best_cand is not None:
                cand_name = best_cand['cand_name']
                cand_score = best_cand['cand_score']
                new_delta = compute_fairness_delta(searcher.target_fairness, best_cand['best_F_value'])

                print(f"[NL] Swapping '{table_name}' with '{cand_name}' from {best_cand['source']}")
                print(f"     Delta: {current_delta:.4f} -> {new_delta:.4f}")

                current_scores = [(s, n) for s, n in current_scores if n != table_name]
                current_scores.append((cand_score, cand_name))
                current_scores.sort(reverse=True)
                current_table_names = [name for _, name in current_scores]

                if best_cand['source'] == 'not-verified' and best_cand['matched_columns'] is not None:
                    matched_columns_map[cand_name] = best_cand['matched_columns']

                non_dominated.remove(best_cand['nd_idx'])

                violating_names.discard(table_name)
                current_F = best_cand['best_F_value']
                current_delta = new_delta
                swaps_made += 1
                made_progress = True

                if fairness_delta_within(current_delta, searcher.delta):
                    break

    elapsed_time = time.time() - start_time
    success = fairness_delta_within(current_delta, searcher.delta)
    print(f"[NL] Done. Swaps: {swaps_made}, Final delta: {current_delta:.4f}, "
          f"Success: {success}, Time: {elapsed_time:.4f}s")

    return {
        'sorted_results': current_scores,
        'delta': current_delta,
        'swaps_made': swaps_made,
        'success': success,
        'time_seconds': elapsed_time,
        'dominance_checks': dominance_checks,
        'candidates_after_winnow': len(non_dominated),
        'dominated_removed_count': dominated_removed_count,
        'dominance_candidate_pool_size': dominance_candidate_pool_size
    }
