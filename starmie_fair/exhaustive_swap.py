"""
Exhaustive Swap fairification strategy.

Heap-based exhaustive swap that removes violating tables with smallest
unionability score first, finding the best replacement via linear scan.
"""

import time
from Custom_Heap import CustomHeap
from bounds import verify_constrained
from utility import (
    round_fairness_f,
    round_fairness_delta,
    compute_fairness_delta,
    fairness_delta_exceeds,
    fairness_delta_within,
)


def exhustive_swap(searcher, scores, violate_list, verified_list, not_verified_list,
                   query_table_name, p_id, protected_value, matched_columns_map=None,
                   qScore=None, threshold=0.6):
    """
    Exhaustive swap algorithm to fairify the results.

    Strategy:
    - Remove the violating table with the smallest unionability score
      (the one contributing least to unionability).
    - If two tables have the same unionability score, replace the one whose
      removal produces the largest improvement in fairness (largest delta_F).
    - A CustomHeap (min-heap) keyed by (score, -delta_F) is used to maintain
      this ordering, where delta_F(t) = F - F' (F' = fairness after removing t).
    - After each successful swap, the heap is rebuilt with updated delta_F values.
    - For each removal candidate, the best replacement is found by linear scan
      over verified_list (then not_verified_list), picking the candidate that
      maximizes the post-swap F value.

    Args:
        searcher: HNSWSearcher_Fair instance (provides compute_F, tables, etc.)
        scores: list of (score, table_name) tuples - current top-K results
        violate_list: list of (table_name, unionability_score, delta) - violating tables in top-K
        verified_list: list of (table_name, unionability_score, delta) - candidate tables to swap in
        not_verified_list: list of (table_name, unionability_score, delta) - unverified candidate tables
        query_table_name: name of the query table
        p_id: protected column id
        protected_value: the protected attribute value
        matched_columns_map: dict mapping table_name to list of (query_col_idx, table_col_idx) tuples
        qScore: query score vectors (needed for verify_constrained)
        threshold: similarity threshold for column matching

    Returns:
        dict with 'sorted_results', 'delta', 'swaps_made', 'success', 'time_seconds'
    """
    start_time = time.time()

    current_scores = list(scores)
    current_table_names = [name for score, name in current_scores]

    available_verified = list(verified_list)
    available_not_verified = list(not_verified_list)

    if matched_columns_map is None:
        matched_columns_map = {}

    swaps_made = 0

    # Compute current fairness F and delta
    current_F_result = searcher.compute_F(
        current_table_names, p_id, protected_value,
        matched_columns_map=matched_columns_map, include_query=True, query_table_name=query_table_name
    )
    current_F = round_fairness_f(current_F_result['protected_proportion'] if current_F_result else 0.0)
    current_delta = compute_fairness_delta(searcher.target_fairness, current_F)

    print(f"Starting exhaustive swap. Initial delta: {current_delta:.4f}, Target: <= {searcher.delta:.4f}")
    print(f"Violating tables: {len(violate_list)}, Verified candidates: {len(verified_list)}, Not verified: {len(not_verified_list)}")

    # Set of violating table names (only these are considered for removal)
    violating_names = set()
    for item in violate_list:
        if len(item) == 2:
            violating_names.add(item[1])
        else:
            violating_names.add(item[0])

    def build_removal_heap():
        """
        Build a min-heap of violating tables keyed by (unionability_score, -delta_F).
        """
        heap = CustomHeap(heap_type='min')
        for score, table_name in current_scores:
            if table_name not in violating_names:
                continue
            remaining = [n for n in current_table_names if n != table_name]
            f_prime_result = searcher.compute_F(
                remaining, p_id, protected_value,
                matched_columns_map=matched_columns_map, include_query=True, query_table_name=query_table_name
            )
            f_prime = round_fairness_f(f_prime_result['protected_proportion'] if f_prime_result else current_F)
            delta_f = round_fairness_f(current_F - f_prime)
            heap.push(score, -delta_f, 0.0, 0.0, value=table_name)
        return heap

    removal_heap = build_removal_heap()
    tried_removal = set()

    print(f"Removal heap size: {len(removal_heap)}")

    # Main swap loop
    while not removal_heap.is_empty() and fairness_delta_exceeds(current_delta, searcher.delta):
        item = removal_heap.pop()
        remove_table_name = item.value
        remove_score = item.k1
        remove_delta_f = -item.k2

        if remove_table_name in tried_removal:
            continue
        if remove_table_name not in current_table_names:
            continue

        print(f"\nConsidering removal: '{remove_table_name}' (score={remove_score:.4f}, delta_F={remove_delta_f:.4f})")

        swap_performed = False
        new_table_names_base = [n for n in current_table_names if n != remove_table_name]

        best_result = _find_best_replacement(
            searcher, new_table_names_base, available_verified, available_not_verified,
            query_table_name, p_id, protected_value, matched_columns_map, qScore, threshold
        )

        if best_result is not None:
            cand_name = best_result['cand_name']
            cand_score = best_result['cand_score']
            best_F_value = best_result['best_F_value']
            new_delta = compute_fairness_delta(searcher.target_fairness, best_F_value)
            source = best_result['source']

            print(f"    Best F from {source}: '{cand_name}' (F={best_F_value:.4f})")

            if round_fairness_delta(new_delta) < round_fairness_delta(current_delta):
                print(f"    Swapping '{remove_table_name}' with '{cand_name}' from {source} list")
                print(f"      Delta improved: {current_delta:.4f} -> {new_delta:.4f}")

                current_scores = [(s, n) for s, n in current_scores if n != remove_table_name]
                current_scores.append((cand_score, cand_name))
                current_scores.sort(reverse=True)
                current_table_names = [name for _, name in current_scores]
                violating_names.discard(remove_table_name)

                if source == 'verified':
                    available_verified.pop(best_result['idx'])
                else:
                    available_not_verified.pop(best_result['idx'])
                    matched_columns_map[cand_name] = best_result['matched_columns']

                current_F = best_F_value
                current_delta = new_delta
                swaps_made += 1
                swap_performed = True

                if fairness_delta_within(current_delta, searcher.delta):
                    elapsed_time = time.time() - start_time
                    print(f"\nTarget fairness achieved! Final delta: {current_delta:.4f}")
                    print(f"Fairification time: {elapsed_time:.4f} seconds")
                    return {
                        'sorted_results': current_scores,
                        'delta': current_delta,
                        'swaps_made': swaps_made,
                        'success': True,
                        'time_seconds': elapsed_time
                    }

                removal_heap = build_removal_heap()
                tried_removal.clear()
                continue
            else:
                print(f"    Candidate doesn't improve fairness (delta: {current_delta:.4f} -> {new_delta:.4f})")

        if not swap_performed:
            tried_removal.add(remove_table_name)
            print(f"  No beneficial swap found for '{remove_table_name}'")

    elapsed_time = time.time() - start_time
    print(f"\nExhaustive swap completed. Swaps made: {swaps_made}, Final delta: {current_delta:.4f}")
    print(f"Fairification time: {elapsed_time:.4f} seconds")

    return {
        'sorted_results': current_scores,
        'delta': current_delta,
        'swaps_made': swaps_made,
        'success': fairness_delta_within(current_delta, searcher.delta),
        'time_seconds': elapsed_time
    }


def _find_best_replacement(searcher, new_table_names_base, available_verified, available_not_verified,
                           query_table_name, p_id, protected_value, matched_columns_map, qScore, threshold):
    """
    Find the best replacement candidate that maximizes post-swap F value.
    Tries verified candidates first, then not-verified candidates.

    Returns:
        dict with 'cand_name', 'cand_score', 'best_F_value', 'idx', 'source',
        and optionally 'matched_columns' (for not-verified), or None if no candidate found.
    """
    best_F_value = -1
    best_result = None

    # Phase 1: Scan verified candidates
    for idx, (cand_name, cand_score, cand_delta) in enumerate(available_verified):
        new_table_names = new_table_names_base + [cand_name]
        cand_result = searcher.compute_F(
            new_table_names, p_id, protected_value,
            matched_columns_map=matched_columns_map, include_query=True, query_table_name=query_table_name
        )
        if cand_result is None:
            continue
        cand_F_value = cand_result['protected_proportion']
        if cand_F_value > best_F_value:
            best_F_value = cand_F_value
            best_result = {
                'cand_name': cand_name,
                'cand_score': cand_score,
                'best_F_value': cand_F_value,
                'idx': idx,
                'source': 'verified'
            }

    # Phase 2: Scan not-verified candidates (only if qScore available)
    if qScore is not None:
        for idx, (cand_name, _, cand_delta) in enumerate(available_not_verified):
            tScore = None
            for table in searcher.tables:
                if table[0] == cand_name:
                    tScore = table[1]
                    break
            if tScore is None:
                continue

            verified_score, cand_matched_columns = verify_constrained(qScore, tScore, threshold, p_id)

            matched_with = [col for row, col in cand_matched_columns if row == p_id]
            if not matched_with:
                continue

            temp_map = dict(matched_columns_map)
            temp_map[cand_name] = cand_matched_columns

            new_table_names = new_table_names_base + [cand_name]
            cand_result = searcher.compute_F(
                new_table_names, p_id, protected_value,
                matched_columns_map=temp_map, include_query=True, query_table_name=query_table_name
            )
            if cand_result is None:
                continue
            cand_F_value = cand_result['protected_proportion']
            if cand_F_value > best_F_value:
                best_F_value = cand_F_value
                best_result = {
                    'cand_name': cand_name,
                    'cand_score': verified_score,
                    'best_F_value': cand_F_value,
                    'idx': idx,
                    'source': 'not-verified',
                    'matched_columns': cand_matched_columns
                }

    return best_result


def exhustive_swap_legacy(searcher, scores, violate_list, verified_list, not_verified_list,
                          query_table_name, p_id, protected_value, matched_columns_map=None,
                          qScore=None, threshold=0.6):
    """Legacy exhaustive swap that iterates violate_list in arbitrary order (kept for comparison)."""
    start_time = time.time()
    current_scores = list(scores)
    current_table_names = [name for score, name in current_scores]
    available_verified = list(verified_list)
    available_not_verified = list(not_verified_list)
    swaps_made = 0
    current_delta, _ = searcher.compute_Delta(
        current_table_names, query_table_name, p_id,
        protected_value,
        include_query=True, matched_columns_map=matched_columns_map, target_fairness=searcher.target_fairness
    )
    for violate_table_name, violate_score, violate_delta in violate_list:

        if violate_table_name not in current_table_names:
            continue

        print(f"\nChecking violating table: {violate_table_name} (delta: {violate_delta:.4f})")

        swap_performed = False

        # Step 1: Try verified list first
        if len(available_verified) > 0:
            print(f"  Trying verified list ({len(available_verified)} candidates)...")

            best_F_candidate = None
            best_F_value = -1
            best_F_idx = -1
            best_new_table_names = None

            for idx, (cand_name, cand_score, cand_delta) in enumerate(available_verified):
                new_table_names = [name for name in current_table_names if name != violate_table_name]
                new_table_names.append(cand_name)

                cand_result = searcher.compute_F(
                    new_table_names, p_id, protected_value,
                    matched_columns_map=matched_columns_map, include_query=True, query_table_name=query_table_name
                )

                if cand_result is None:
                    continue

                cand_F_value = cand_result['protected_proportion']

                if cand_F_value > best_F_value:
                    best_F_candidate = (cand_name, cand_score, cand_delta)
                    best_F_value = cand_F_value
                    best_F_idx = idx
                    best_new_table_names = new_table_names

            if best_F_candidate is not None and best_new_table_names is not None:
                cand_name, cand_score, _ = best_F_candidate

                new_delta, _ = searcher.compute_Delta(
                    best_new_table_names, query_table_name, p_id,
                    protected_value,
                    include_query=True, matched_columns_map=matched_columns_map, target_fairness=searcher.target_fairness
                )

                print(f"    Best F from verified: '{cand_name}' (F={best_F_value:.4f})")

                if round_fairness_delta(new_delta) < round_fairness_delta(current_delta):
                    print(f"    Swapping '{violate_table_name}' with '{cand_name}' from verified list")
                    print(f"      Delta improved: {current_delta:.4f} -> {new_delta:.4f}")

                    violate_tuple = next((s, n) for s, n in current_scores if n == violate_table_name)
                    current_scores.remove(violate_tuple)
                    current_scores.append((cand_score, cand_name))
                    current_scores.sort(reverse=True)

                    current_table_names = [name for score, name in current_scores]
                    available_verified.pop(best_F_idx)
                    current_delta = new_delta
                    swaps_made += 1
                    swap_performed = True

                    if fairness_delta_within(current_delta, searcher.delta):
                        elapsed_time = time.time() - start_time
                        print(f"\nTarget fairness achieved! Final delta: {current_delta:.4f}")
                        print(f"Fairification time: {elapsed_time:.4f} seconds")
                        return {
                            'sorted_results': current_scores,
                            'delta': current_delta,
                            'swaps_made': swaps_made,
                            'success': True,
                            'time_seconds': elapsed_time
                        }
                else:
                    print(f"    Verified candidate doesn't improve fairness (delta: {current_delta:.4f} -> {new_delta:.4f})")

        # Step 2: Try not_verified list
        if not swap_performed and len(available_not_verified) > 0 and qScore is not None:
            print(f"  Trying not-verified list ({len(available_not_verified)} candidates)...")

            best_F_candidate = None
            best_F_value = -1
            best_F_idx = -1
            best_new_table_names = None
            best_cand_matched_columns = None

            for idx, (cand_name, _, cand_delta) in enumerate(available_not_verified):
                tScore = None
                for table in searcher.tables:
                    if table[0] == cand_name:
                        tScore = table[1]
                        break

                if tScore is None:
                    continue

                verified_score, cand_matched_columns = verify_constrained(qScore, tScore, threshold, p_id)

                matched_with = [col for row, col in cand_matched_columns if row == p_id]
                if not matched_with:
                    print(f"    Not-verified candidate '{cand_name}' doesn't match the protected column")
                    continue

                temp_map = dict(matched_columns_map) if matched_columns_map else {}
                temp_map[cand_name] = cand_matched_columns

                new_table_names = [name for name in current_table_names if name != violate_table_name]
                new_table_names.append(cand_name)

                cand_result = searcher.compute_F(
                    new_table_names, p_id, protected_value,
                    matched_columns_map=temp_map, include_query=True, query_table_name=query_table_name
                )

                if cand_result is None:
                    continue

                cand_F_value = cand_result['protected_proportion']

                if cand_F_value > best_F_value:
                    best_F_candidate = (cand_name, verified_score, cand_delta)
                    best_F_value = cand_F_value
                    best_F_idx = idx
                    best_new_table_names = new_table_names
                    best_cand_matched_columns = cand_matched_columns

            if best_F_candidate is not None and best_new_table_names is not None:
                cand_name, cand_score, _ = best_F_candidate

                if matched_columns_map is None:
                    matched_columns_map = {}
                matched_columns_map[cand_name] = best_cand_matched_columns

                new_delta, _ = searcher.compute_Delta(
                    best_new_table_names, query_table_name, p_id,
                    protected_value,
                    include_query=True, matched_columns_map=matched_columns_map, target_fairness=searcher.target_fairness
                )

                print(f"    Best F from not-verified: '{cand_name}' (F={best_F_value:.4f}, score={cand_score:.4f})")

                if round_fairness_delta(new_delta) < round_fairness_delta(current_delta):
                    print(f"    Swapping '{violate_table_name}' with '{cand_name}' from not-verified list")
                    print(f"      Delta improved: {current_delta:.4f} -> {new_delta:.4f}")

                    violate_tuple = next((s, n) for s, n in current_scores if n == violate_table_name)
                    current_scores.remove(violate_tuple)
                    current_scores.append((cand_score, cand_name))
                    current_scores.sort(reverse=True)

                    current_table_names = [name for score, name in current_scores]
                    available_not_verified.pop(best_F_idx)
                    current_delta = new_delta
                    swaps_made += 1
                    swap_performed = True

                    if fairness_delta_within(current_delta, searcher.delta):
                        elapsed_time = time.time() - start_time
                        print(f"\nTarget fairness achieved! Final delta: {current_delta:.4f}")
                        print(f"Fairification time: {elapsed_time:.4f} seconds")
                        return {
                            'sorted_results': current_scores,
                            'delta': current_delta,
                            'swaps_made': swaps_made,
                            'success': True,
                            'time_seconds': elapsed_time
                        }
                else:
                    print(f"    Not-verified candidate doesn't improve fairness (delta: {current_delta:.4f} -> {new_delta:.4f})")

        if not swap_performed:
            print(f"  No beneficial swap found for '{violate_table_name}' in either list")
    elapsed_time = time.time() - start_time
    return {
        'sorted_results': current_scores,
        'delta': current_delta,
        'swaps_made': swaps_made,
        'success': fairness_delta_within(current_delta, searcher.delta),
        'time_seconds': elapsed_time
    }
