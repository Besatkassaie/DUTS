"""
HNSWSearcher_Fair: Combines HNSW indexing with heap-based result management
This class uses HNSW for efficient candidate retrieval and applies heap-based 
top-K management with bounds for fair and efficient ranking.
"""

import numpy as np
import random
import pickle
import time
import heapq
import hnswlib
from Custom_Heap import CustomHeap


from munkres import Munkres, make_cost_matrix, DISALLOWED
from numpy.linalg import norm
from bounds import verify, upper_bound_bm, lower_bound_bm, get_edges, verify_matched_columns,verify_constrained
from typing import List, Tuple, Set, Optional
import os
from TableMetadata import MetadataStore
from utility import (
    Utility,
    round_fairness_f,
    round_fairness_delta,
    compute_fairness_delta,
    fairness_delta_exceeds,
    fairness_delta_within,
)
from preference import PreferenceConfig
from nl_swap import nl_swap
from bnl_swap import bnl_swap


class HNSWSearcher_Fair(object):
    def __init__(self,
                 table_path,
                 index_path,
                 query_path_raw,
                 table_path_raw,
                 scale, 
                 random_seed=42,
                 load_metadata=False,
                 metadata_dir=None,
                 delta=0.1,
                 target_fairness=0.5,
                 preference_config_path=None,
                 ):
        """
        Initialize HNSW searcher with heap-based result management

        Args:
            table_path: Path to pickled table data
            index_path: Path to save/load HNSW index
            query_path_raw: Path to raw query data
            table_path_raw: Path to raw table data
            scale: Percentage of tables to use (for scalability experiments)
            random_seed: Seed used for deterministic table sampling (default: 42)
            load_metadata: If True, try to load existing metadata from disk (default: False)
            metadata_dir: Directory to save/load metadata files (default: same as index_path)
            delta: Acceptable deviation from the target fairness score (default: 0.1)
            target_fairness: Target fairness score (default: 0.5)
            preference_config_path: JSON file for PreferenceConfig (NL/BNL dominance).
                If None, uses preference_config.json next to this module when present.
        """
      
        self.delta = delta
        self.target_fairness = target_fairness

        _pref_default = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preference_config.json')
        _pref = preference_config_path or _pref_default
        self.preference_config = PreferenceConfig(_pref if os.path.isfile(_pref) else None)
        
        tfile = open(table_path, "rb")
        tables = pickle.load(tfile)
        query_data = Utility.read_csv_files_to_dict(query_path_raw)     # this is a dictionary of query data
        self.query_data = query_data
        table_data = Utility.read_csv_files_to_dict(table_path_raw)     # this is a dictionary of table data
        self.table_data = table_data
        # For scalability experiments: load a percentage of tables
        self.random_seed = random_seed
        self.tables = random.Random(self.random_seed).sample(
            tables, int(scale * len(tables))
        )
        print("From %d total data-lake tables, scale down to %d tables" % (len(tables), len(self.tables)))
        print(f"Using deterministic random seed: {self.random_seed}")
        tfile.close()
        self.vec_dim = len(self.tables[1][1][0])
        # ========================================================================
        # STEP 2: Setup Metadata Paths
        # ========================================================================
        print("\n[2/4] Setting up metadata...")
        
        # Determine metadata directory
        if metadata_dir is None:
            metadata_dir = os.path.dirname(index_path) or '.'
        
        # Ensure metadata directory exists
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Define single combined metadata file path
        self.metadata_path = os.path.join(metadata_dir, 'metadata_combined.pkl')
        
        print(f"  Combined metadata: {self.metadata_path}")
        
        # ========================================================================
        # STEP 3: Build/Load Combined Metadata (datalake + query)
        # ========================================================================
        print("\n[3/4] Creating combined metadata store...")
        
        if load_metadata and os.path.exists(self.metadata_path):
            # Load existing combined metadata
            print(f"  Loading existing combined metadata...")
            try:
                self.metadata_store = MetadataStore.load(self.metadata_path)
                print(f"  ✓ Loaded: {self.metadata_store}")
            except Exception as e:
                print(f"  ⚠ Failed to load metadata: {e}")
                print(f"  Building new combined metadata...")
                self._build_combined_metadata()
        else:
            # Build new combined metadata
            print(f"  Building new combined metadata...")
            self._build_combined_metadata()
        
        # ========================================================================
        # STEP 4: Materialize Metadata (Save to Disk)
        # ========================================================================
        print("\n  Materializing metadata to disk...")
        
        # Save combined metadata
        if hasattr(self, 'metadata_store'):
            self.metadata_store.save(self.metadata_path)
        
        print(f"  ✓ Metadata materialized successfully")
        
        # Print metadata summary
        print("\n  Metadata Summary:")
        if hasattr(self, 'metadata_store'):
            stats = self.metadata_store.get_stats()
            print(f"    Combined: {stats['total_tables']} tables, "
                f"{stats['total_records']:,} records, "
                f"{stats['total_columns_with_distributions']} columns with distributions")






        # Build HNSW index
        index_start_time = time.time()
        self.index = hnswlib.Index(space='cosine', dim=self.vec_dim)
        self.all_columns, self.col_table_ids = self._preprocess_table_hnsw()
        self.index.init_index(max_elements=len(self.all_columns), ef_construction=100, M=32)
        self.index.set_ef(10)
        self.index.add_items(self.all_columns)
        if index_path:  # None = keep the index in memory only (DUTS main.py: no per-run disk copy)
            self.index.save_index(index_path)
        print("--- Indexing Time: %s seconds ---" % (time.time() - index_start_time))
        
 
     

    def _build_combined_metadata(self):
        """
        Helper method to build combined metadata for both datalake and query tables.
        Distributions are computed for ALL columns.
        """
        from TableMetadata import MetadataStore
        
        self.metadata_store = MetadataStore()
        
        # Build metadata from datalake tables first
        print(f"    Building metadata for {len(self.table_data)} datalake tables...")
        self.metadata_store.build_metadata_from_tables(self.table_data)
        
        # Then add query tables to the same metadata store
        print(f"    Adding metadata for {len(self.query_data)} query tables...")
        self.metadata_store.build_metadata_from_tables(self.query_data)
        
        print(f"    ✓ Combined metadata: {self.metadata_store}") 
           
   
   
    def topk(self, enc, query, K, N=50, threshold=0.6):
        """
        Original HNSW topk method (for comparison)
        
        Args:
            enc: Encoder type ('sato', 'sherlock', 'cl', etc.)
            query: Query table (name, column_vectors)
            K: Number of top results to return
            N: Number of nearest neighbor columns to retrieve from HNSW
            threshold: Similarity threshold for column matching
            
        Returns:
            Top-K scored tables and number of candidates evaluated
        """
        query_cols = []
        for col in query[1]:
            query_cols.append(col)
        candidates = self._find_candidates(query_cols, N)
        
        if enc == 'sato':
            scores = []
            querySherlock = query[1][:, :1187]
            querySato = query[1][0, 1187:]
            for table in candidates:
                sherlock = table[1][:, :1187]
                sato = table[1][0, 1187:]
                sScore = self._verify(querySherlock, sherlock, threshold)
                sherlockScore = (1/min(len(querySherlock), len(sherlock))) * sScore
                satoScore = self._cosine_sim(querySato, sato)
                score = sherlockScore + satoScore
                scores.append((score, table[0]))
        else:
            scores = [(self._verify(query[1], table[1], threshold), table[0]) for table in candidates]
        
        scores.sort(reverse=True)
        scoreLength = len(scores)
        return scores[:K], scoreLength
    
    def topk_fairified(self, enc, query, K, N=50, threshold=0.6, p_id=None, protected_value=None,
                       algorithm="exhustive_swap"):
        """
        HNSW top-K with fairification over verified swap candidates only.

        Args:
            enc: Encoder type ('sato', 'sherlock', 'cl', etc.)
            query: Query table (name, column_vectors)
            K: Number of top results to return
            N: Number of nearest neighbor columns to retrieve from HNSW
            threshold: Similarity threshold for column matching
            p_id: protected column id
            protected_value: the protected attribute value
            algorithm: fairification algorithm (exhustive_swap, nl_swap, bnl_swap)

        Returns:
            Fairification result dict and number of candidates evaluated
        """
        query_cols = []
        for col in query[1]:
            query_cols.append(col)
        candidates = self._find_candidates(query_cols, N)

        matched_columns_map = {}
        scores = []
        for table in candidates:
            table_name = table[0]
            score, matched_columns = verify_constrained(
                query[1], table[1], threshold, p_id
            )
            matched_columns_map[table_name] = matched_columns
            scores.append((score, table_name))

        scoreLength = len(scores)

        scores.sort(reverse=True)
        initial_topk = scores[:K]
        candidates_for_swap = scores[K:]

        # Tables in initial top-K that individually violate fairness (delta > self.delta).
        violate_list = []
        if p_id is not None:
            for unionability_score, table_name in initial_topk:
                matched_columns = matched_columns_map.get(table_name, [])
                matched_with = [col for row, col in matched_columns if row == p_id]
                if not matched_with:
                    continue
                delta_, _ = self.compute_Delta(
                    table_name,
                    query[0],
                    p_id,
                    protected_value,
                    include_query=False,
                    matched_columns_map=matched_columns_map,
                    target_fairness=self.target_fairness,
                )
                if fairness_delta_exceeds(delta_, self.delta):
                    violate_list.append((table_name, unionability_score, delta_))

        verified_list = []
        if p_id is not None:
            for unionability_score, table_name in candidates_for_swap:
                matched_columns = matched_columns_map.get(table_name, [])
                matched_with = [col for row, col in matched_columns if row == p_id]
                if not matched_with:
                    continue
                delta_, _ = self.compute_Delta(
                    table_name,
                    query[0],
                    p_id,
                    protected_value,
                    include_query=False,
                    matched_columns_map=matched_columns_map,
                    target_fairness=self.target_fairness,
                )
                verified_list.append((table_name, unionability_score, delta_))

        H = list(initial_topk)
        heapq.heapify(H)
        scores = self.fairify_verified_only(
            H, violate_list, verified_list, query, p_id, protected_value,
            matched_columns_map, threshold, algorithm,
        )
        return scores, scoreLength
    
    def topk_heap_bounds(self, enc, query, K, N=50, threshold=0.6, 
                         p_id=None, protected_value=None, algorithm="exhustive_swap"):
        """
        HNSW with heap-based result management and bounds pruning
        Uses HNSW to find candidates, then applies heap + bounds for fair ranking
        
        Args:
            enc: Encoder type ('sato', 'sherlock', 'cl', etc.)
            query: Query table (name, column_vectors)
            K: Number of top results to return
            N: Number of nearest neighbor columns to retrieve from HNSW
            threshold: Similarity threshold for column matching
            p_id: protected column id
            protected_value: the protected attribute value
            algorithm: fairification algorithm (exhustive_swap, nl_swap, bnl_swap)
        Returns:
            Top-K scored tables with heap-based fair ranking
        """
        # Step 1: Use HNSW to find candidate tables
        
        query_cols = []
        for col in query[1]:
            query_cols.append(col)
        candidates = self._find_candidates(query_cols, N)
        #print(f"HNSW found {len(candidates)} candidate tables")
        # those tables that are verified but not used in the top k
        #verified_heap = CustomHeap(heap_type='max')
        verified_list = []
        
        # those tables that are not verified but are candidates
        #not_verified_heap = CustomHeap(heap_type='max')
        not_verified_list = []
        # those tables that are verified and are in top k but violate the fairness constraint
        #violate_heap= CustomHeap(heap_type='min')
        violate_list = []
        # make the violate_list a heap which has the minimum unionablility score as root 
        heapq.heapify(violate_list)

 
        # Step 2: Apply heap-based ranking with bounds to candidates
        H = []
        heapq.heapify(H)
        
        verification_count = 0
        
        # Create a mapping from table_name to matched_columns
        # This stores the column mapping between query and each table
        matched_columns_map = {}  # {table_name: [(query_col, table_col), ...]}
        
        for table in candidates:
            table_name = table[0]
            table_data = self.table_data[table_name]
            tScore = table[1]
            qScore = query[1]
            
            # Add to heap to get len(H) = K
            if len(H) < K:
               # score_, matched_columns_ = verify_matched_columns(qScore, tScore, threshold)
                score, matched_columns = verify_constrained(qScore, tScore, threshold, p_id)
                # Store the matched_columns for this table in the map
                matched_columns_map[table_name] = matched_columns
                
                if p_id is not None:
                    matched_with = [col for row, col in matched_columns if row == p_id]
                    if matched_with:
                        #print(f"✓ Query column {p_id} matched with table column(s): {matched_with}")
                        heapq.heappush(H, (score, table[0]))
                        # check whether the table violates the fairness constraint and add to the violate_heap if it does
                        
                        delta_, result = self.compute_Delta(table_name, query[0], p_id, protected_value, include_query=False, matched_columns_map=matched_columns_map, target_fairness=self.target_fairness)
                     
                        unionability_score = score
                        if fairness_delta_exceeds(delta_, self.delta):
             
                           # violate_list.append((table_name, unionability_score,delta_))
                            heapq.heappush(violate_list, (unionability_score, table_name))
                        
                    else:
                        print(f"✗ Query column {p_id} NOT matched")
                    verification_count += 1
            else:
                # Heap is full, use bounds to decide if we should verify
                topScore = H[0]  # Minimum score in heap
                
                # Compute bounds
                edges, nodes1, nodes2 = get_edges(qScore, tScore, threshold)
                lb = lower_bound_bm(edges, nodes1, nodes2)
                ub = upper_bound_bm(edges, nodes1, nodes2)
                
                # If lower bound exceeds current minimum, definitely add
                if lb > topScore[0]:
                    score, matched_columns = verify_constrained(qScore, tScore, threshold, p_id)
                   # score_, matched_columns_ = verify_matched_columns(qScore, tScore, threshold)

                    # Store the matched_columns for this table in the map
                    matched_columns_map[table_name] = matched_columns
                    
                    if p_id is not None:
                        matched_with = [col for row, col in matched_columns if row == p_id]
                        if matched_with:
                            #print(f"✓ Query column {p_id} matched with table column(s): {matched_with}")
                            sc, tname=heapq.heappop(H)
                            # check if the table was violating the fairness constrants remove it from the violate_heap if it was
                            delat_,result=self.compute_Delta(tname, query[0], p_id, protected_value, include_query=False, matched_columns_map=matched_columns_map, target_fairness=self.target_fairness)
                               
                            if fairness_delta_exceeds(delat_, self.delta):
                                #violate_heap.remove_by_value(tname)
                                for i, (first, *rest) in enumerate(violate_list):  
                                    if first == tname:
                                        del violate_list[i]
                                        break
                            #add the poped one to another heap $T_{\text{ver}}$
                    
                            unionability_score = sc
                            verified_list.append((tname, unionability_score, delat_))
                            
                            heapq.heappush(H, (score, table_name))
                            # check whether the table violates the fairness constraint and add to the violate_list if it does
                            
                            delat_,result=self.compute_Delta(table_name, query[0], p_id, protected_value, include_query=False, matched_columns_map=matched_columns_map, target_fairness=self.target_fairness)
                            if fairness_delta_exceeds(delat_, self.delta):
                         
                                unionability_score = score
                              
                                violate_list.append((table_name, unionability_score,delat_))
                        else:
                            print(f"✗ Query column {p_id} NOT matched")
                            # $T_{not\_aligned\_with\_p}$
                    verification_count += 1
                # If upper bound might exceed, verify
                elif ub <= topScore[0]:
                   print("discard table")
                   # add to another heap to use in case we did not have other 
                   # options $T_{\text{drop}}:$ use -100 to show that this is not evaluated 
                   delat_,result=self.compute_Delta(table_name, query[0], p_id, protected_value, include_query=False, matched_columns_map=None, target_fairness=self.target_fairness)
                   
                   unionability_score = -100
                  
                   not_verified_list.append((table_name, unionability_score,delat_))
                elif verify(qScore, tScore, threshold) > topScore[0]:
                    score, matched_columns = verify_constrained(qScore, tScore, threshold, p_id)
                    #score_, matched_columns_ = verify_matched_columns(qScore, tScore, threshold)

                    # Store the matched_columns for this table in the map
                    matched_columns_map[table_name] = matched_columns
                    
                    if p_id is not None:
                        matched_with = [col for row, col in matched_columns if row == p_id]
                        if matched_with:
                            #print(f"✓ Query column {p_id} matched with table column(s): {matched_with}")
                            sc, tname=heapq.heappop(H)
                            
                        # check if the table was violating the fairness constrants remove it from the violate_heap if it was
                            delat_,result=self.compute_Delta(tname, query[0], p_id, protected_value,
                                                             include_query=False, matched_columns_map=matched_columns_map, target_fairness=self.target_fairness)
                               
                            if fairness_delta_exceeds(delat_, self.delta):
                                #removed_item=violate_heap.remove_by_value(tname)
                                for i, (first, *rest) in enumerate(violate_list):  
                                    if first == tname:
                                        del violate_list[i]
                                        break
                            #add the poped one to another heap $T_{\text{ver}}$
                       
                            verified_list.append((tname, unionability_score,delat_))
                            heapq.heappush(H, (score, table_name))
                            # check whether the table violates the fairness constraint and add to the violate_heap if it does
                            
                            delat_,result=self.compute_Delta(table_name, query[0], p_id, protected_value, include_query=False, matched_columns_map=matched_columns_map, target_fairness=self.target_fairness)
                            if fairness_delta_exceeds(delat_, self.delta):
                                unionability_score = score
                          
                                violate_list.append((table_name, unionability_score,delat_))
                                  
                        else:
                            print(f"✗ Query column {p_id} NOT matched")

        
        scores = self.fairify(H, violate_list, verified_list, not_verified_list, query, p_id, protected_value, matched_columns_map, threshold, algorithm)

        #print(f"Verification calls: {verification_count} out of {len(candidates)} candidates")
        
        return scores, verification_count
    
    def _preprocess_table_hnsw(self):
        """
        Flatten all columns from all tables for HNSW indexing
        """
        all_columns = []
        col_table_ids = []
        for idx, table in enumerate(self.tables):
            for col in table[1]:
                all_columns.append(col)
                col_table_ids.append(idx)
        return all_columns, col_table_ids
    
    def _find_candidates(self, query_cols, N):
        """
        Use HNSW to find candidate tables based on column similarity
        
        Args:
            query_cols: List of query column vectors
            N: Number of nearest neighbors to retrieve per query column
            
        Returns:
            List of candidate tables
        """
        table_subs = set()
        labels, _ = self.index.knn_query(query_cols, k=N)
        
        for result in labels:
            for idx in result:
                table_id = self.col_table_ids[idx]
                table_subs.add(table_id)
        
        candidates = []
        for tid in table_subs:
            candidates.append(self.tables[tid])
        
        return candidates
    
    def _cosine_sim(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        assert vec1.ndim == vec2.ndim
        return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))
    
    def _verify(self, table1, table2, threshold):
        """
        Compute similarity score using bipartite matching
        """
        score = 0.0
        nrow = len(table1)
        ncol = len(table2)
        graph = np.zeros(shape=(nrow, ncol), dtype=float)
        
        for i in range(nrow):
            for j in range(ncol):
                sim = self._cosine_sim(table1[i], table2[j])
                if sim > threshold:
                    graph[i, j] = sim
        
        max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
        m = Munkres()
        indexes = m.compute(max_graph)
        
        for row, col in indexes:
            score += graph[row, col]
        
        return score
    
    def _combine_sherlock_sato(self, score, qScore, tScore, satoScore):
        """
        Helper method for combining Sherlock and SATO scores
        """
        sherlockScore = (1/min(len(qScore), len(tScore))) * score
        full_satoScore = sherlockScore + satoScore
        return full_satoScore
   
   
   
    def _get_aligned_column_id(self, table_name, query_table_name, query_column_id, alignment):
        """
        Get the aligned column ID for a table based on query column alignment
        Automatically detects alignment format (list or dict)
        
        Args:
            table_name: Name of the table to get column for
            query_table_name: Name of the query table
            query_column_id: Query column identifier (int or str)
            alignment: Either:
                - List: [(query_name, query_col, table_name, table_col), ...]
                - Dict: {(query_name, query_col): [(table1, col1), (table2, col2), ...]}
        
        Returns:
            Column identifier for the table, or None if not found
        """
        if alignment is None:
            return None
        
        # Convert query column to index if it's a name
        if isinstance(query_column_id, str):
            if hasattr(self, 'metadata_store'):
                query_meta = self.metadata_store.get_metadata(query_table_name)
                if query_meta:
                    query_col_idx = query_meta.get_column_index(query_column_id)
                    if query_col_idx is None:
                        return None
                else:
                    return None
            else:
                return None
        else:
            query_col_idx = query_column_id
        
        # Detect format and process accordingly
        if isinstance(alignment, dict):
            # Dictionary format: {(query_name, query_col): [(table, col), ...]}
            alignment_key = (query_table_name, query_col_idx)
            
            if alignment_key not in alignment:
                return None
            
            table_mappings = alignment[alignment_key]
            for t_name, t_col_id in table_mappings:
                if t_name == table_name:
                    return t_col_id
        
        elif isinstance(alignment, list):
            # List format: [(query_name, query_col, table_name, table_col), ...]
            for entry in alignment:
                if len(entry) == 4:
                    q_name, q_col_id, t_name, t_col_id = entry
                    if (q_name == query_table_name and 
                        q_col_id == query_col_idx and 
                        t_name == table_name):
                        return t_col_id
        
        return None
        
    
    def compute_F(self, table_names, q_column_identifier, protected_value, 
                matched_columns_map=None, include_query=True, query_table_name=None):
        """
        Compute fairness statistics for given tables and protected attribute.
        
        Behavior based on number of tables:
        - If more than one table: compute delta for all tables PLUS query table
        - If only one table: compute delta for just that one table (ignore include_query)
        
        Args:
            table_names (list): List of table names to analyze
            q_column_identifier (int or str): Column index or name (e.g., 1 or "gender")
            protected_value (str): The protected attribute value (e.g., "Female", "Asian")
            matched_columns_map: Dict mapping table_name to list of (query_col_idx, table_col_idx) tuples
                                 OR list of tuples for single table (backward compatibility)
            include_query (bool): Whether to include query table in computation
                                  (only used when there are multiple tables)
            query_table_name (str, optional): Name of query table if include_query=True
        
        Returns:
            dict: Dictionary containing stats:
                {
                    'total_records_datalake': int,
                    'protected_records_datalake': int,
                    'total_records_query': int (if include_query),
                    'protected_records_query': int (if include_query),
                    'total_records_combined': int,
                    'protected_records_combined': int,
                    'protected_proportion': float,
                    'per_table_stats': [
                        {
                            'table_name': str,
                            'table_type': 'datalake' or 'query',
                            'total_records': int,
                            'protected_records': int,
                            'protected_proportion': float
                        },
                        ...
                    ],
                    'num_tables_analyzed': int
                }
            Returns None if metadata not available or column not categorical
        """
        # Check if metadata stores are available
        if not hasattr(self, 'metadata_store'):
            print("❌ Datalake metadata store not available")
            return None
        
        # Flatten table_names if it's a nested list (e.g., [[table1, table2]] -> [table1, table2])
        if table_names and isinstance(table_names[0], list):
            table_names = table_names[0]
        
        # Convert to list if single string
        if isinstance(table_names, str):
            table_names = [table_names]
        
        # Determine whether to include query based on number of tables
        # If only one table: just compute for that table (don't include query)
        # If more than one table: compute for all tables plus query table
        if len(table_names) == 1:
            include_query = False
        
        # Local counters for this call
        total_records_datalake = 0
        protected_records_datalake = 0
        total_records_query = 0
        protected_records_query = 0
        per_table_stats = []
        processed_tables = set()
        # Keep a safe default even when no datalake table is processed.
        column_identifier = q_column_identifier
        
        # ========================================================================
        # Process Datalake Tables
        # ========================================================================
        for table_name in table_names:
            # Skip already processed tables to avoid double-counting
            table_key = ('datalake', table_name)
            if table_key in processed_tables:
                continue
            
            # Get metadata for this table
            metadata = self.metadata_store.get_metadata(table_name)
            
            if not metadata:
                print(f"⚠️  Warning: Table '{table_name}' not found in datalake metadata")
                continue
            
            # Get aligned column ID for this table using matched_columns_map
            # matched_columns_map is either:
            # - A dict: {table_name: [(query_col, table_col), ...]}
            # - A list: [(query_col, table_col), ...] (for single table backward compatibility)
            aligned_column_id = None
            
            if matched_columns_map is not None:
                if isinstance(matched_columns_map, dict):
                    # Look up the matched_columns for this specific table
                    table_matched_columns = matched_columns_map.get(table_name)
                    if table_matched_columns is not None:
                        for query_col, table_col in table_matched_columns:
                            if query_col == q_column_identifier:
                                aligned_column_id = table_col
                                break
                elif isinstance(matched_columns_map, list):
                    # Backward compatibility: treat as single matched_columns list
                    for query_col, table_col in matched_columns_map:
                        if query_col == q_column_identifier:
                            aligned_column_id = table_col
                            break
            
            # Datalake rows count toward F only when the query protected column aligns.
            # Protected values always come from the query; never fall back to using
            # q_column_identifier as a direct datalake column index.
            if aligned_column_id is None:
                continue
            
            column_identifier = aligned_column_id
            
            # Resolve column name
            if isinstance(column_identifier, int):
                if column_identifier >= len(metadata.column_names):
                    print(f"⚠️  Warning: Column index {column_identifier} out of range for {table_name}")
                    continue
                column_name = metadata.column_names[column_identifier]
            else:
                column_name = column_identifier
                if column_name not in metadata.column_names:
                    print(f"⚠️  Warning: Column '{column_name}' not found in {table_name}")
                    continue
            
            # Check if column is categorical
            if not metadata.is_categorical(column_name):
                print(f"⚠️  Warning: Column '{column_name}' is not categorical in {table_name}")
                continue
            
            # Get counts
            table_total = metadata.num_records
            table_protected = metadata.get_category_count(column_name, protected_value)
            
            # Accumulate totals
            total_records_datalake += table_total
            protected_records_datalake += table_protected
            
            # Store per-table stats
            table_proportion = table_protected / table_total if table_total > 0 else 0.0
            per_table_stats.append({
                'table_name': table_name,
                'table_type': 'datalake',
                'total_records': table_total,
                'protected_records': table_protected,
                'protected_proportion': table_proportion
            })
            
            # Mark table as processed
            processed_tables.add(table_key)
        
        # ========================================================================
        # Process Query Table (if requested)
        # ========================================================================
        if include_query and query_table_name:
                # Skip if query table already processed
                query_key = ('query', query_table_name)
                if not hasattr(self, 'metadata_store'):
                    print("⚠️  Warning: Metadata store not available")
                else:
                    # Get query metadata from combined metadata store
                    query_metadata = self.metadata_store.get_metadata(query_table_name)
                    
                    if query_metadata:
                        # Resolve protected column on the *query* table (query schema), not the
                        # datalake aligned column from the loop above (column_identifier).
                        if isinstance(q_column_identifier, int):
                            if q_column_identifier < len(query_metadata.column_names):
                                query_column_name = query_metadata.column_names[q_column_identifier]
                            else:
                                print(f"⚠️  Warning: Column index {q_column_identifier} out of range for query "
                                      f"(query has {len(query_metadata.column_names)} columns)")
                                query_column_name = None
                        else:
                            query_column_name = q_column_identifier
                        
                        # Get counts if column is valid and categorical
                        if query_column_name and query_metadata.is_categorical(query_column_name):
                            query_total = query_metadata.num_records
                            query_protected = query_metadata.get_category_count(
                                query_column_name, protected_value
                            )
                            
                            # Accumulate query totals
                            total_records_query += query_total
                            protected_records_query += query_protected
                            
                            # Add to per-table stats
                            query_proportion = (query_protected / query_total 
                                            if query_total > 0 else 0.0)
                            per_table_stats.append({
                                'table_name': query_table_name,
                                'table_type': 'query',
                                'total_records': query_total,
                                'protected_records': query_protected,
                                'protected_proportion': query_proportion
                            })
                            
                            # Mark query table as processed
                            processed_tables.add(query_key)
                        else:
                            print(f"⚠️  Warning: Column not categorical or not found in query table")
                    else:
                        print(f"⚠️  Warning: Query table '{query_table_name}' not found in metadata")
        
        # ========================================================================
        # Compute Combined Statistics
        # ========================================================================
        total_records_combined = total_records_datalake + total_records_query
        protected_records_combined = protected_records_datalake + protected_records_query
        
        # Compute overall protected proportion (F_score based on all accumulated data)
        protected_proportion = round_fairness_f(
            protected_records_combined / total_records_combined
            if total_records_combined > 0 else 0.0
        )
        
        # Build result dictionary (reflects accumulated state)
        result = {
            'total_records_datalake': total_records_datalake,
            'protected_records_datalake': protected_records_datalake,
            'total_records_combined': total_records_combined,
            'protected_records_combined': protected_records_combined,
            'protected_proportion': protected_proportion,
            'per_table_stats': per_table_stats.copy(),  # Return a copy to prevent external modification
            'column_identifier': column_identifier,
            'protected_value': protected_value,
            'num_tables_analyzed': len(per_table_stats)
        }
        
        # Add query-specific fields if included
        if include_query:
            result['total_records_query'] = total_records_query
            result['protected_records_query'] = protected_records_query
        
        # Print summary
        #self._print_compute_F_summary(result)
        
        return result


    def _print_compute_F_summary(self, result):
        """
        Helper function to print compute_F results
        
        Args:
            result (dict): Result from compute_F
        """
        print("\n" + "="*70)
        print("FAIRNESS COMPUTATION SUMMARY")
        print("="*70)
        print(f"Column:           {result['column_identifier']}")
        print(f"Protected Value:  '{result['protected_value']}'")
        print(f"Tables Analyzed:  {result['num_tables_analyzed']}")
        
        print(f"\n--- Datalake Statistics ---")
        print(f"Total Records:       {result['total_records_datalake']:,}")
        print(f"Protected Records:   {result['protected_records_datalake']:,}")
        
        if 'total_records_query' in result:
            print(f"\n--- Query Statistics ---")
            print(f"Total Records:       {result['total_records_query']:,}")
            print(f"Protected Records:   {result['protected_records_query']:,}")
        
        print(f"\n--- Combined Statistics ---")
        print(f"Total Records:       {result['total_records_combined']:,}")
        print(f"Protected Records:   {result['protected_records_combined']:,}")
        print(f"Protected Proportion: {result['protected_proportion']:.2%}")
        
        # Visual bar
        bar_length = 50
        filled = int(result['protected_proportion'] * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"Distribution:        |{bar}|")
        
        print("="*70 + "\n")

    def compute_Delta(self, table_names, query_table_name, protected_column_id, protected_value, include_query=True,
                      matched_columns_map=None, target_fairness=0.5):
        """
        Compute the fairness deviation for a set of tables from target fairness score
        
        This method evaluates how much a set of retrieved tables deviates from
        a target fairness score with respect to protected/sensitive columns.
        
        Args:
            table_names (list or set): List/set of table names to evaluate
                                       (e.g., ['table1.csv', 'table2.csv'])
            query_table_name: Name of the query table
            protected_column_id: int - the protected column index in query
            protected_value: The protected attribute value
            include_query: Whether to include query table in computation
            matched_columns_map: Dict mapping table_name to list of (query_col_idx, table_col_idx) tuples
                                 OR list of tuples for single table (backward compatibility)
            target_fairness (float, optional): Target fairness score. Defaults to 0.5
        
        Returns:
            tuple: (delta, result) where delta = target_fairness - actual_fairness
        
        Notes:
            - Lower Delta values indicate better fairness alignment
            - Delta = 0 means perfect fairness alignment
    
        """   
        # compute the union of tables and query data considering the matched_columns_map
        result = self.compute_F([table_names], protected_column_id, 
                                protected_value,
                                matched_columns_map=matched_columns_map, 
                                include_query=include_query, 
                                query_table_name=query_table_name)
        
        F_table_union = round_fairness_f(result['protected_proportion'])
        return compute_fairness_delta(target_fairness, F_table_union), result
 


    def fairify_swap_example(scores, worst_violator_name, best_verified_tuple):
        """
        swap operation for fairification
        
        Args:
            scores: [(score, table_name), ...] sorted descending
            worst_violator_name: Name of table to remove
            best_verified_tuple: (score, table_name) to add
        """
        print(f"Before swap: {scores}")
        
        # Step 1: Remove worst violator
        scores = [(s, n) for s, n in scores if n != worst_violator_name]
        print(f"After removal: {scores}")
        
        # Step 2: Add best verified
        scores.append(best_verified_tuple)
        print(f"After addition: {scores}")
        
        # Step 3: Re-sort
        scores.sort(reverse=True)
        print(f"After re-sort: {scores}")
        
        return scores

        # Test
        # scores = [(0.92, "B"), (0.85, "A"), (0.78, "C"), (0.75, "D")]
        # new_scores = fairify_swap_example(scores, "C", (0.88, "E"))

        # Output:
        # Before swap: [(0.92, 'B'), (0.85, 'A'), (0.78, 'C'), (0.75, 'D')]
        # After removal: [(0.92, 'B'), (0.85, 'A'), (0.75, 'D')]
        # After addition: [(0.92, 'B'), (0.85, 'A'), (0.75, 'D'), (0.88, 'E')]
        # After re-sort: [(0.92, 'B'), (0.88, 'E'), (0.85, 'A'), (0.75, 'D')]




    def fairify_verified_only(self, H, H_violate, Tables_verified, query, p_id, protected_value,
                              matched_columns_map=None, threshold=0.6, algorithm="exhustive_swap"):
        """
        Fairify using only verified swap candidates (no not_verified_list).

        Same return format as fairify(); swap algorithms only search Tables_verified.
        """
        return self._fairify(
            H, H_violate, Tables_verified, query, p_id, protected_value,
            matched_columns_map=matched_columns_map, threshold=threshold, algorithm=algorithm,
            not_verified_list=[],
        )

    def fairify(self, H, H_violate, Tables_verified, Tables_not_verified,
                query, p_id, protected_value, matched_columns_map=None, threshold=0.6, algorithm="exhustive_swap"):
        """
        Fairify the ranking by reordering results to minimize fairness violation
        
        Args:
            H (list): Main heap - list of (score, table_name) tuples
            H_violate (list): Violation list - list of (table_name, unionability_score,delta) tuples
            Tables_verified (list): List of (table_name, unionability_score,delta) tuples
            Tables_not_verified (list): List of (table_name, unionability_score,delta) tuples1: not verified tables 
            query: tuple of (query_table_name, query_score_vectors)
            p_id: protected column id
            protected_value: the protected attribute value
            matched_columns_map: dict mapping table_name to list of (query_col_idx, table_col_idx) tuples
            threshold: similarity threshold for column matching
            algorithm: fairification algorithm: exhustive_swap, nl_swap, or bnl_swap
        
        Returns:
            dict: Always returns a dictionary with consistent format:
                {
                    'sorted_results': [(score, table_name), ...],  # Fair ranking
                    'delta': float,                                 # Fairness deviation
                    'swaps_made': int,                             # Number of swaps performed
                    'success': True
                }
        """
        return self._fairify(
            H, H_violate, Tables_verified, query, p_id, protected_value,
            matched_columns_map=matched_columns_map, threshold=threshold, algorithm=algorithm,
            not_verified_list=Tables_not_verified,
        )

    def _fairify(self, H, H_violate, Tables_verified, query, p_id, protected_value,
                 matched_columns_map=None, threshold=0.6, algorithm="exhustive_swap",
                 not_verified_list=None):
        query_table_name = query[0]
        qScore = query[1]
        if not_verified_list is None:
            not_verified_list = []

        # 1. Extract the top k results from the main heap H
        scores = []
        while len(H) > 0:
            scores.append(heapq.heappop(H))
        scores.sort(reverse=True)

        # Extract table names (second element)
        table_names = [name for score, name in scores]
        # compute the fairness of the result if violated the deviation is greater than the delta then we need to fairify the results
        current_delta, result_current = self.compute_Delta(table_names, query_table_name, p_id, protected_value,
                              include_query=True, matched_columns_map=matched_columns_map, target_fairness=self.target_fairness)

        if fairness_delta_exceeds(current_delta, self.delta):
            print("Initial Top K violates the fairness constraint")
            if algorithm == "exhustive_swap":
                swap_result = self.exhustive_swap(scores, H_violate, Tables_verified, not_verified_list,
                                                  query_table_name, p_id, protected_value, matched_columns_map,
                                                  qScore, threshold)
            elif algorithm == "nl_swap":
                swap_result = nl_swap(self, scores, H_violate, Tables_verified, not_verified_list,
                                      query_table_name, p_id, protected_value, matched_columns_map,
                                      qScore, threshold)
            elif algorithm == "bnl_swap":
                swap_result = bnl_swap(self, scores, H_violate, Tables_verified, not_verified_list,
                                       query_table_name, p_id, protected_value, matched_columns_map,
                                       qScore, threshold)
            else:
                raise ValueError(f"Unknown fairification algorithm: {algorithm}")
            # Flag that this query's initial top-K exceeded the fairness threshold
            # and therefore required (i.e. invoked) the fairification step.
            swap_result['needs_fairification'] = True
            swap_result['initial_delta'] = current_delta
            swap_result.setdefault(
                'initial_fairness',
                result_current['protected_proportion'] if result_current else (self.target_fairness - current_delta),
            )
            return swap_result

        # Fairness constraint already satisfied - return consistent format
        print("Initial Top K satisfies the fairness constraint")
        initial_fairness = (
            result_current['protected_proportion'] if result_current else (self.target_fairness - current_delta)
        )
        return {
            'sorted_results': scores,
            'delta': current_delta,
            'swaps_made': 0,
            'success': True,
            'time_seconds': 0.0,
            'needs_fairification': False,
            'initial_delta': current_delta,
            'initial_fairness': initial_fairness,
        }
    
    def exhustive_swap(self, scores, violate_list,
                       verified_list, not_verified_list, 
                       query_table_name, p_id, protected_value, 
                       matched_columns_map=None,
                       qScore=None, 
                       threshold=0.6):
        """
        Exhaustive swap algorithm to fairify the results.

        Strategy:
        - Remove the violating table whose removal produces the largest
          improvement in fairness (largest delta_F).
        - If two tables have the same delta_F, remove the one with the smallest
          unionability score (least contribution to unionability).
        - A CustomHeap (max-heap) keyed by (delta_F, -score) is used to maintain
          this ordering, where delta_F(t) = F' - F (F' = fairness after removing t).
        - After each successful swap, the heap is rebuilt with updated delta_F values.
        - For each removal candidate, the best replacement is found by linear scan
          over verified_list (then not_verified_list), picking the candidate that
          maximizes the post-swap F value.

        Args:
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
            dict with 'sorted_results', 'delta', 'swaps_made', 'success', 'time_seconds', 'initial_fairness'
        """
        print("Exhaustive swap started")
        start_time = time.time()

        current_scores = list(scores)
        current_table_names = [name for score, name in current_scores]

        available_verified = list(verified_list)
        available_not_verified = list(not_verified_list)

        if matched_columns_map is None:
            matched_columns_map = {}

        swaps_made = 0

        # Compute current fairness F and delta
        current_F_result = self.compute_F(
            current_table_names, p_id, protected_value,
            matched_columns_map=matched_columns_map, include_query=True, query_table_name=query_table_name
        )
        current_F = round_fairness_f(current_F_result['protected_proportion'] if current_F_result else 0.0)
        current_delta = compute_fairness_delta(self.target_fairness, current_F)
        initial_fairness = current_F

        print(f"Starting exhaustive swap. Initial fairness (F from score input): {initial_fairness:.4f}, "
              f"Initial delta: {current_delta:.4f}, Target: <= {self.delta:.4f}")
        print(f"Violating tables: {len(violate_list)}, Verified candidates: {len(verified_list)}, Not verified: {len(not_verified_list)}")

        # Set of violating table names (only these are considered for removal)
        # violate_list can be either:
        #   - 2-tuples: (unionability_score, table_name) from topk_heap_bounds heapq
        #   - 3-tuples: (table_name, unionability_score, delta) from legacy callers
        violating_names = set()
        for item in violate_list:
            if len(item) == 2:
                violating_names.add(item[1])  # (score, name)
            else:
                violating_names.add(item[0])  # (name, score, delta)

        def build_removal_heap():
            """
            Build a max-heap of violating tables keyed by (delta_F, -unionability_score).
            - k1 = delta_F: largest delta_F popped first (most fairness gain)
            - k2 = -score: on tie, smallest unionability score popped first
            - delta_F(t) = F' - F, where F' is fairness after removing t from top-K
            """
            heap = CustomHeap(heap_type='max')
            for score, table_name in current_scores:
                if table_name not in violating_names:
                    continue
                remaining = [n for n in current_table_names if n != table_name]
                f_prime_result = self.compute_F(
                    remaining, p_id, protected_value,
                    matched_columns_map=matched_columns_map, include_query=True, query_table_name=query_table_name
                )
                f_prime = round_fairness_f(
                    f_prime_result['protected_proportion'] if f_prime_result else current_F
                )
                delta_f = round_fairness_f(f_prime - current_F)
                heap.push(delta_f, -score, 0.0, 0.0, value=table_name)
            return heap

        removal_heap = build_removal_heap()
        tried_removal = set()

        print(f"Removal heap size: {len(removal_heap)}")

        # ========================================================================
        # Convert heap to sorted list for non-destructive iteration
        # ========================================================================
        def get_sorted_removal_candidates():
            """Extract all items from heap into a sorted list (by priority order)."""
            candidates = []
            temp_heap = build_removal_heap()  # Build a fresh copy
            while not temp_heap.is_empty():
                item = temp_heap.pop()
                candidates.append({
                    'table_name': item.value,
                    'score': -item.k2,
                    'delta_f': item.k1
                })
            return candidates

        removal_candidates = get_sorted_removal_candidates()
        candidate_idx = 0

        # ========================================================================
        # Main swap loop: iterate over removal candidates without modifying heap
        # ========================================================================
        while candidate_idx < len(removal_candidates) and fairness_delta_exceeds(current_delta, self.delta):
            candidate = removal_candidates[candidate_idx]
            remove_table_name = candidate['table_name']
            remove_score = candidate['score']
            remove_delta_f = candidate['delta_f']

            # Skip invalid items
            if remove_table_name in tried_removal:
                candidate_idx += 1
                continue
            if remove_table_name not in current_table_names:
                candidate_idx += 1
                continue

            print(f"\nConsidering removal: '{remove_table_name}' (score={remove_score:.4f}, delta_F={remove_delta_f:.4f})")

            swap_performed = False
            new_table_names_base = [n for n in current_table_names if n != remove_table_name]

            # Step 1: Find best replacement from verified list first
            best_verified = self._find_best_from_verified(
                new_table_names_base, available_verified,
                query_table_name, p_id, protected_value, matched_columns_map
            )

            # Check if verified candidate improves delta
            if best_verified is not None:
                verified_new_delta = compute_fairness_delta(
                    self.target_fairness, best_verified['best_F_value']
                )
                print(f"    Best from verified: '{best_verified['cand_name']}' (F={best_verified['best_F_value']:.4f}, would give delta={verified_new_delta:.4f})")
                
                if round_fairness_delta(verified_new_delta) < round_fairness_delta(current_delta):
                    # Verified candidate improves delta - use it
                    best_result = best_verified
                    best_result['source'] = 'verified'
                    print(f"    -> Verified candidate improves delta, using it")
                else:
                    print(f"    -> Verified candidate doesn't improve delta (current={current_delta:.4f})")
                    best_verified = None  # Mark as not usable

            # Step 2: If verified didn't improve delta, try not-verified list
            if best_verified is None:
                best_not_verified = self._find_best_from_not_verified(
                    new_table_names_base, available_not_verified,
                    query_table_name, p_id, protected_value, matched_columns_map, qScore, threshold
                )
                
                if best_not_verified is not None:
                    not_verified_new_delta = compute_fairness_delta(
                        self.target_fairness, best_not_verified['best_F_value']
                    )
                    print(f"    Best from not-verified: '{best_not_verified['cand_name']}' (F={best_not_verified['best_F_value']:.4f}, would give delta={not_verified_new_delta:.4f})")
                    
                    if round_fairness_delta(not_verified_new_delta) < round_fairness_delta(current_delta):
                        # Not-verified candidate improves delta - use it
                        best_result = best_not_verified
                        best_result['source'] = 'not-verified'
                        print(f"    -> Not-verified candidate improves delta, using it")
                    else:
                        print(f"    -> Not-verified candidate doesn't improve delta (current={current_delta:.4f})")
                        best_result = None
                else:
                    best_result = None
            else:
                best_result = best_verified
                best_result['source'] = 'verified'

            # Step 3: Perform swap if we found a beneficial candidate
            if best_result is not None:
                cand_name = best_result['cand_name']
                cand_score = best_result['cand_score']
                best_F_value = best_result['best_F_value']
                new_delta = compute_fairness_delta(self.target_fairness, best_F_value)
                source = best_result['source']

                print(f"    Swapping '{remove_table_name}' with '{cand_name}' from {source} list")
                print(f"      Delta improved: {current_delta:.4f} -> {new_delta:.4f}")

                # Perform swap
                current_scores = [(s, n) for s, n in current_scores if n != remove_table_name]
                current_scores.append((cand_score, cand_name))
                current_scores.sort(reverse=True)
                current_table_names = [name for _, name in current_scores]
                violating_names.discard(remove_table_name)

                # Remove used candidate from its source list
                if source == 'verified':
                    available_verified.pop(best_result['idx'])
                else:
                    available_not_verified.pop(best_result['idx'])
                    matched_columns_map[cand_name] = best_result['matched_columns']

                current_F = round_fairness_f(best_F_value)
                current_delta = new_delta
                swaps_made += 1
                swap_performed = True

                if fairness_delta_within(current_delta, self.delta):
                    elapsed_time = time.time() - start_time
                    print(f"\nTarget fairness achieved! Final delta: {current_delta:.4f}")
                    print(f"Fairification time: {elapsed_time:.4f} seconds")
                    return {
                        'sorted_results': current_scores,
                        'delta': current_delta,
                        'swaps_made': swaps_made,
                        'success': True,
                        'time_seconds': elapsed_time,
                        'initial_fairness': initial_fairness,
                    }

                # Move to next candidate (don't restart from beginning)
                candidate_idx += 1
                continue

            # No beneficial swap found - just move to next candidate (no popping)
            if not swap_performed:
                tried_removal.add(remove_table_name)
                candidate_idx += 1  # Move to next without modifying anything
                print(f"  No beneficial swap found for '{remove_table_name}'")

        # ========================================================================
        # Final result
        # ========================================================================
        elapsed_time = time.time() - start_time
        print(f"\nExhaustive swap completed. Swaps made: {swaps_made}, Final delta: {current_delta:.4f}")
        print(f"Fairification time: {elapsed_time:.4f} seconds")

        # If target fairness not achieved, return empty list
        if fairness_delta_exceeds(current_delta, self.delta):
            print(f"Target fairness NOT achieved. Returning empty list.")
            return {
                'sorted_results': [],
                'delta': current_delta,
                'swaps_made': swaps_made,
                'success': False,
                'time_seconds': elapsed_time,
                'initial_fairness': initial_fairness,
            }

        return {
            'sorted_results': current_scores,
            'delta': current_delta,
            'swaps_made': swaps_made,
            'success': True,
            'time_seconds': elapsed_time,
            'initial_fairness': initial_fairness,
        }

    def _find_best_replacement(self, new_table_names_base, available_verified, available_not_verified,
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
            cand_result = self.compute_F(
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
                for table in self.tables:
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
                cand_result = self.compute_F(
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

    def _find_best_from_verified(self, new_table_names_base, available_verified,
                                  query_table_name, p_id, protected_value, matched_columns_map):
        """
        Find the best replacement candidate from verified list only.
        Maximizes post-swap F; on tie, prefers higher unionability score.
        
        Returns:
            dict with 'cand_name', 'cand_score', 'best_F_value', 'idx', or None if no candidate found.
        """
        best_F_value = -1
        best_result = None

        for idx, (cand_name, cand_score, cand_delta) in enumerate(available_verified):
            new_table_names = new_table_names_base + [cand_name]
            cand_result = self.compute_F(
                new_table_names, p_id, protected_value,
                matched_columns_map=matched_columns_map, include_query=True, query_table_name=query_table_name
            )
            if cand_result is None:
                continue
            cand_F_value = cand_result['protected_proportion']
            if best_result is None or (cand_F_value, cand_score) > (best_F_value, best_result['cand_score']):
                best_F_value = cand_F_value
                best_result = {
                    'cand_name': cand_name,
                    'cand_score': cand_score,
                    'best_F_value': cand_F_value,
                    'idx': idx
                }

        return best_result

    def _find_best_from_not_verified(self, new_table_names_base, available_not_verified,
                                      query_table_name, p_id, protected_value, matched_columns_map, 
                                      qScore, threshold):
        """
        Find the best replacement candidate from not-verified list only.
        Maximizes post-swap F; on tie, prefers higher unionability score.
        
        Returns:
            dict with 'cand_name', 'cand_score', 'best_F_value', 'idx', 'matched_columns', 
            or None if no candidate found.
        """
        if qScore is None:
            return None
            
        best_F_value = -1
        best_result = None

        for idx, (cand_name, _, cand_delta) in enumerate(available_not_verified):
            tScore = None
            for table in self.tables:
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
            cand_result = self.compute_F(
                new_table_names, p_id, protected_value,
                matched_columns_map=temp_map, include_query=True, query_table_name=query_table_name
            )
            if cand_result is None:
                continue
            cand_F_value = cand_result['protected_proportion']
            if best_result is None or (cand_F_value, verified_score) > (best_F_value, best_result['cand_score']):
                best_F_value = cand_F_value
                best_result = {
                    'cand_name': cand_name,
                    'cand_score': verified_score,
                    'best_F_value': cand_F_value,
                    'idx': idx,
                    'matched_columns': cand_matched_columns
                }

        return best_result

    # Keep the old iteration-based method for reference/comparison
    def _exhustive_swap_legacy(self, scores, violate_list, verified_list, not_verified_list, query_table_name, p_id, protected_value, matched_columns_map=None, qScore=None, threshold=0.6):
        """Legacy exhaustive swap that iterates violate_list in arbitrary order (kept for comparison)."""
        start_time = time.time()
        current_scores = list(scores)
        current_table_names = [name for score, name in current_scores]
        available_verified = list(verified_list)
        available_not_verified = list(not_verified_list)
        swaps_made = 0
        current_delta, _ = self.compute_Delta(
            current_table_names, query_table_name, p_id,
            protected_value,
            include_query=True, matched_columns_map=matched_columns_map, target_fairness=self.target_fairness
        )
        for violate_table_name, violate_score, violate_delta in violate_list:
            
            # Check if this violating table is still in current results
            if violate_table_name not in current_table_names:
                continue  # Already swapped out
            
            print(f"\nChecking violating table: {violate_table_name} (delta: {violate_delta:.4f})")
            
            swap_performed = False
            
            # -----------------------------------------------------------------
            # Step 1: Try verified list first
            # -----------------------------------------------------------------
            if len(available_verified) > 0:
                print(f"  Trying verified list ({len(available_verified)} candidates)...")
                
                # Find the candidate that gives the BIGGEST F value after swapping
                best_F_candidate = None
                best_F_value = -1
                best_F_idx = -1
                best_new_table_names = None
                
                for idx, (cand_name, cand_score, cand_delta) in enumerate(available_verified):
                    # Create hypothetical new table list with this swap
                    new_table_names = [name for name in current_table_names if name != violate_table_name]
                    new_table_names.append(cand_name)
                    
                    # Compute F value for the NEW table list (after hypothetical swap)
                    cand_result = self.compute_F(
                        new_table_names, p_id, protected_value,
                        matched_columns_map=matched_columns_map, include_query=True, query_table_name=query_table_name
                    )
                    
                    if cand_result is None:
                        continue
                        
                    cand_F_value = cand_result['protected_proportion']
                    
                    # Pick the candidate that gives the highest F value after swapping
                    if cand_F_value > best_F_value:
                        best_F_candidate = (cand_name, cand_score, cand_delta)
                        best_F_value = cand_F_value
                        best_F_idx = idx
                        best_new_table_names = new_table_names
                
                # Check if swapping with this best candidate improves overall fairness
                if best_F_candidate is not None and best_new_table_names is not None:
                    cand_name, cand_score, _ = best_F_candidate
                    
                    # Compute delta with this swap
                    new_delta, _ = self.compute_Delta(
                        best_new_table_names, query_table_name, p_id,
                        protected_value,
                        include_query=True, matched_columns_map=matched_columns_map, target_fairness=self.target_fairness
                    )
                    
                    print(f"    Best F from verified: '{cand_name}' (F={best_F_value:.4f})")
                    
                    # Check if this swap improves overall fairness
                    if round_fairness_delta(new_delta) < round_fairness_delta(current_delta):
                        print(f"    ✓ Swapping '{violate_table_name}' with '{cand_name}' from verified list")
                        print(f"      Delta improved: {current_delta:.4f} -> {new_delta:.4f}")
                        
                        # Update current scores
                        violate_tuple = next((s, n) for s, n in current_scores if n == violate_table_name)
                        current_scores.remove(violate_tuple)
                        current_scores.append((cand_score, cand_name))
                        current_scores.sort(reverse=True)
                        
                        # Update table names
                        current_table_names = [name for score, name in current_scores]
                        
                        # Remove used candidate from available list
                        available_verified.pop(best_F_idx)
                        
                        # Update current delta
                        current_delta = new_delta
                        swaps_made += 1
                        swap_performed = True
                        
                        # Check if we've achieved target fairness
                        if fairness_delta_within(current_delta, self.delta):
                            elapsed_time = time.time() - start_time
                            print(f"\n✓ Target fairness achieved! Final delta: {current_delta:.4f}")
                            print(f"⏱ Fairification time: {elapsed_time:.4f} seconds")
                            return {
                                'sorted_results': current_scores,
                                'delta': current_delta,
                                'swaps_made': swaps_made,
                                'success': True,
                                'time_seconds': elapsed_time
                            }
                    else:
                        print(f"    ✗ Verified candidate doesn't improve fairness (delta: {current_delta:.4f} -> {new_delta:.4f})")
            
            # -----------------------------------------------------------------
            # Step 2: If verified list didn't help, try not_verified list
            # -----------------------------------------------------------------
            if not swap_performed and len(available_not_verified) > 0 and qScore is not None:
                print(f"  Trying not-verified list ({len(available_not_verified)} candidates)...")
                
                # Find the candidate that gives the BIGGEST F value after swapping
                best_F_candidate = None
                best_F_value = -1
                best_F_idx = -1
                best_new_table_names = None
                best_cand_matched_columns = None
                
                for idx, (cand_name, _, cand_delta) in enumerate(available_not_verified):
                    # Get tScore for this candidate table
                    tScore = None
                    for table in self.tables:
                        if table[0] == cand_name:
                            tScore = table[1]
                            break
                    
                    if tScore is None:
                        continue
                    
                    # Call verify_constrained to obtain the unionability score and matched_columns
                    verified_score, cand_matched_columns = verify_constrained(qScore, tScore, threshold, p_id)
                    
                    # Check if protected column matches
                    matched_with = [col for row, col in cand_matched_columns if row == p_id]
                    if not matched_with:
                        print(f"    ✗ Not-verified candidate '{cand_name}' doesn't match the protected column")
                        continue
                    
                    # Create a temporary map with this candidate's matched_columns added
                    temp_map = dict(matched_columns_map) if matched_columns_map else {}
                    temp_map[cand_name] = cand_matched_columns
                    
                    # Create hypothetical new table list with this swap
                    new_table_names = [name for name in current_table_names if name != violate_table_name]
                    new_table_names.append(cand_name)
                    
                    # Compute F value for the NEW table list (after hypothetical swap)
                    cand_result = self.compute_F(
                        new_table_names, p_id, protected_value,
                        matched_columns_map=temp_map, include_query=True, query_table_name=query_table_name
                    )
                    
                    if cand_result is None:
                        continue
                    
                    cand_F_value = cand_result['protected_proportion']
                    
                    # Pick the candidate that gives the highest F value after swapping
                    if cand_F_value > best_F_value:
                        best_F_candidate = (cand_name, verified_score, cand_delta)
                        best_F_value = cand_F_value
                        best_F_idx = idx
                        best_new_table_names = new_table_names
                        best_cand_matched_columns = cand_matched_columns
                
                # Check if swapping with this best candidate improves overall fairness
                if best_F_candidate is not None and best_new_table_names is not None:
                    cand_name, cand_score, _ = best_F_candidate
                    
                    # Add the best candidate's matched_columns to the map
                    if matched_columns_map is None:
                        matched_columns_map = {}
                    matched_columns_map[cand_name] = best_cand_matched_columns
                    
                    # Compute delta with this swap using updated matched_columns_map
                    new_delta, _ = self.compute_Delta(
                        best_new_table_names, query_table_name, p_id,
                        protected_value,
                        include_query=True, matched_columns_map=matched_columns_map, target_fairness=self.target_fairness
                    )
                    
                    print(f"    Best F from not-verified: '{cand_name}' (F={best_F_value:.4f}, score={cand_score:.4f})")
                    
                    # Check if this swap improves overall fairness
                    if round_fairness_delta(new_delta) < round_fairness_delta(current_delta):
                        print(f"    ✓ Swapping '{violate_table_name}' with '{cand_name}' from not-verified list")
                        print(f"      Delta improved: {current_delta:.4f} -> {new_delta:.4f}")
                        
                        # Update current scores
                        violate_tuple = next((s, n) for s, n in current_scores if n == violate_table_name)
                        current_scores.remove(violate_tuple)
                        current_scores.append((cand_score, cand_name))
                        current_scores.sort(reverse=True)
                        
                        # Update table names
                        current_table_names = [name for score, name in current_scores]
                        
                        # Remove used candidate from available list
                        available_not_verified.pop(best_F_idx)
                        
                        # Update current delta
                        current_delta = new_delta
                        swaps_made += 1
                        swap_performed = True
                        
                        # Check if we've achieved target fairness
                        if fairness_delta_within(current_delta, self.delta):
                            elapsed_time = time.time() - start_time
                            print(f"\n✓ Target fairness achieved! Final delta: {current_delta:.4f}")
                            print(f"⏱ Fairification time: {elapsed_time:.4f} seconds")
                            return {
                                'sorted_results': current_scores,
                                'delta': current_delta,
                                'swaps_made': swaps_made,
                                'success': True,
                                'time_seconds': elapsed_time
                            }
                    else:
                        print(f"    ✗ Not-verified candidate doesn't improve fairness (delta: {current_delta:.4f} -> {new_delta:.4f})")
            
            if not swap_performed:
                print(f"  No beneficial swap found for '{violate_table_name}' in either list")
        elapsed_time = time.time() - start_time
        return {
            'sorted_results': current_scores,
            'delta': current_delta,
            'swaps_made': swaps_made,
            'success': fairness_delta_within(current_delta, self.delta),
            'time_seconds': elapsed_time
        }

   