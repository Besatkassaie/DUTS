"""
Blocked Nested Loops (BNL) fairification strategy.

Based on Chomicki (TODS 2003) Section 4.2 / Börzsönyi et al. (2001) — the BNL
algorithm for evaluating the winnow operator using a memory window with
configurable preference relations.

BNL requires the preference relation to be a strict partial order
(irreflexive, transitive).
"""

import time
from nl_swap import _resolve_candidate, _enrich_candidates
from utility import (
    round_fairness_f,
    round_fairness_delta,
    compute_fairness_delta,
    fairness_delta_exceeds,
    fairness_delta_within,
)


def bnl_swap(searcher, scores, violate_list, verified_list, not_verified_list,
             query_table_name, p_id, protected_value, matched_columns_map=None,
             qScore=None, threshold=0.6):
    """
    Blocked Nested Loops (BNL) fairification algorithm.

    Candidates are streamed through a window. For each new candidate:
      - If dominated by any window entry (per preference_config), discard it.
      - If it dominates window entries, remove those and add the new candidate.
      - If incomparable with all window entries, add to the window.

    The window acts as a single-pass winnow: after streaming all candidates,
    only non-dominated candidates remain. The best from the window is used for
    the swap, then the window is rebuilt for the next violator.

    Advantage over NL: The window prunes dominated candidates as they stream in,
    so fewer pairwise comparisons are needed in practice.

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
             'dominance_checks', 'window_sizes'
    """
    start_time = time.time()

    current_scores = list(scores)
    current_table_names = [name for _, name in current_scores]

    if matched_columns_map is None:
        matched_columns_map = {}

    # Build the full candidate pool
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
    window_sizes = []
    dominated_candidate_names = set()
    dominance_candidate_pool_size = len(
        [e for e in enriched if e['name'] not in current_table_names]
    )

    # Compute initial delta
    current_F_result = searcher.compute_F(
        current_table_names, p_id, protected_value,
        matched_columns_map=matched_columns_map, include_query=True,
        query_table_name=query_table_name
    )
    current_F = round_fairness_f(current_F_result['protected_proportion'] if current_F_result else 0.0)
    current_delta = compute_fairness_delta(searcher.target_fairness, current_F)

    print(f"[BNL] Starting. Initial delta: {current_delta:.4f}, Target: <= {searcher.delta:.4f}")
    print(f"[BNL] Violating tables: {len(violating_names)}, Candidates: {len(all_candidates)}")
    print(f"[BNL] Preference config: {searcher.preference_config}")

    # =========================================================================
    # Main BNL loop: iterate over violating tables
    # =========================================================================
    made_progress = True
    used_candidates = set()

    while made_progress and fairness_delta_exceeds(current_delta, searcher.delta):
        made_progress = False

        for score_val, table_name in list(current_scores):
            if table_name not in violating_names:
                continue
            if table_name not in current_table_names:
                continue

            # =================================================================
            # BNL window construction: stream candidates through window
            # using preference_config for dominance (Börzsönyi et al. 2001)
            # =================================================================
            # Window entries: (enriched_idx, enriched_attrs, real_score,
            #                  cand_matched_cols, cand_F_value)
            window = []

            for idx in range(len(enriched)):
                e = enriched[idx]
                cand_name = e['name']
                source = e['source']

                if cand_name in current_table_names or cand_name in used_candidates:
                    continue

                resolved = _resolve_candidate(
                    searcher, cand_name, e['score'], qScore, threshold, p_id, matched_columns_map
                )
                if resolved is None:
                    continue
                real_score, cand_matched_cols = resolved

                # Compute F value for swapping table_name -> cand_name
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

                cand_F = round_fairness_f(cand_result['protected_proportion'])
                cand_delta_val = compute_fairness_delta(searcher.target_fairness, cand_F)

                # Skip if doesn't improve over current state
                if round_fairness_delta(cand_delta_val) >= round_fairness_delta(current_delta):
                    continue

                # --- BNL window comparison using preference_config ---
                dominated_by_window = False
                entries_to_remove = []

                for w_idx, (w_e_idx, w_attrs, w_score, w_matched, w_F) in enumerate(window):
                    dominance_checks += 1
                    if searcher.preference_config.dominates(w_attrs, e):
                        dominated_by_window = True
                        break
                    dominance_checks += 1
                    if searcher.preference_config.dominates(e, w_attrs):
                        entries_to_remove.append(w_idx)

                if dominated_by_window:
                    dominated_candidate_names.add(cand_name)
                    continue

                # Remove dominated window entries
                for w_idx in sorted(entries_to_remove, reverse=True):
                    _, removed_attrs, _, _, _ = window[w_idx]
                    dominated_candidate_names.add(removed_attrs['name'])
                    window.pop(w_idx)

                # Add new candidate to window
                window.append((idx, e, real_score, cand_matched_cols, cand_F))

            window_sizes.append(len(window))

            # =================================================================
            # Pick the best candidate from window (highest F value)
            # =================================================================
            if not window:
                continue

            best_entry = max(window, key=lambda x: x[4])
            best_e_idx, best_attrs, best_cand_score, best_matched_cols, best_F_value = best_entry
            best_cand_name = best_attrs['name']
            best_source = best_attrs['source']
            new_delta = compute_fairness_delta(searcher.target_fairness, best_F_value)

            if round_fairness_delta(new_delta) < round_fairness_delta(current_delta):
                print(f"[BNL] Swapping '{table_name}' with '{best_cand_name}' from {best_source} "
                      f"(window size: {len(window)})")
                print(f"      Delta: {current_delta:.4f} -> {new_delta:.4f}")

                current_scores = [(s, n) for s, n in current_scores if n != table_name]
                current_scores.append((best_cand_score, best_cand_name))
                current_scores.sort(reverse=True)
                current_table_names = [name for _, name in current_scores]

                if best_source == 'not-verified' and best_matched_cols is not None:
                    matched_columns_map[best_cand_name] = best_matched_cols

                used_candidates.add(best_cand_name)
                violating_names.discard(table_name)
                current_F = best_F_value
                current_delta = new_delta
                swaps_made += 1
                made_progress = True

                if fairness_delta_within(current_delta, searcher.delta):
                    break

    elapsed_time = time.time() - start_time
    success = fairness_delta_within(current_delta, searcher.delta)
    print(f"[BNL] Done. Swaps: {swaps_made}, Final delta: {current_delta:.4f}, "
          f"Success: {success}, Time: {elapsed_time:.4f}s")
    if window_sizes:
        print(f"[BNL] Window sizes: {window_sizes} (avg: {sum(window_sizes)/len(window_sizes):.1f})")

    return {
        'sorted_results': current_scores,
        'delta': current_delta,
        'swaps_made': swaps_made,
        'success': success,
        'time_seconds': elapsed_time,
        'dominance_checks': dominance_checks,
        'window_sizes': window_sizes,
        'dominated_removed_count': len(dominated_candidate_names),
        'dominance_candidate_pool_size': dominance_candidate_pool_size
    }
