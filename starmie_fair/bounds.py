
import numpy as np
import random
import os

from munkres import Munkres, make_cost_matrix, DISALLOWED
from numpy.linalg import norm


def cosine_sim(vec1, vec2):
    assert vec1.ndim == vec2.ndim
    return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))


def verify(table1, table2, threshold=0.6):
    score = 0.0
    nrow = len(table1)
    ncol = len(table2)
    graph = np.zeros(shape=(nrow,ncol),dtype=float)
    for i in range(nrow):
        for j in range(ncol):
            sim = cosine_sim(table1[i],table2[j])
            if sim > threshold:
                graph[i,j] = sim
    max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
    m = Munkres()
    indexes = m.compute(max_graph)
    for row,col in indexes:
        score += graph[row,col]
    return score

#Besat added this version of verify to make sure that we have the mathced columns back
def verify_matched_columns(table1, table2, threshold=0.6):
    score = 0.0
    matched_columns = []    
    nrow = len(table1)
    ncol = len(table2)
    graph = np.zeros(shape=(nrow,ncol),dtype=float)
    for i in range(nrow):
        for j in range(ncol):
            sim = cosine_sim(table1[i],table2[j])
            if sim > threshold:
                graph[i,j] = sim
    max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
    m = Munkres()
    indexes = m.compute(max_graph)
    for row,col in indexes:
        score += graph[row,col]
    return score, indexes



def verify_constrained(table1, table2, threshold=0.6, mandatory_idx=None):
    """
    Compute a maximum-weight bipartite matching between rows of table1 and table2
    using cosine similarity, with an optional constraint that a given row
    (mandatory_idx in table1) must be matched.

    If mandatory_idx is given and all original weights in that row are zero
    (i.e., no edge above threshold), the function returns 0.0 immediately.
    """

    nrow = len(table1)
    ncol = len(table2)

    # Build similarity matrix (weights)
    graph = np.zeros((nrow, ncol), dtype=float)
    for i in range(nrow):
        for j in range(ncol):
            sim = cosine_sim(table1[i], table2[j])
            if sim > threshold:
                graph[i, j] = sim

    # Keep the original weights
    orig_graph = graph.copy()

    # --- NEW: if mandatory_idx has only zero weights, matching is impossible ---
    if mandatory_idx is not None:
        if not (0 <= mandatory_idx < nrow):
            raise ValueError(f"mandatory_idx={mandatory_idx} is out of range for table1 size {nrow}")

        # If all original weights in that row are zero, no valid edge exists
        if np.all(orig_graph[mandatory_idx, :] == 0.0):
            # "match is not possible"
            return 0.0, []

        # Otherwise, apply the big bonus trick
        S = np.abs(graph).sum()
        B = S + 1.0
        graph[mandatory_idx, :] += B

    # Prepare cost matrix for Hungarian algorithm
    max_weight = graph.max() if graph.size > 0 else 0.0

    def to_cost(cost):
        if cost == DISALLOWED:
            return DISALLOWED
        return max_weight - cost

    cost_matrix = make_cost_matrix(graph, to_cost)

    # Run Hungarian algorithm
    m = Munkres()
    indexes = m.compute(cost_matrix)

    # Check that mandatory_idx is actually matched
    if mandatory_idx is not None:
        matched_rows = {r for (r, c) in indexes}
        if mandatory_idx not in matched_rows:
            # No matching including mandatory_idx beats the best one without it
            return 0.0, []

    # Sum the ORIGINAL similarities (without bonus)
    score = 0.0
    for row, col in indexes:
        score += orig_graph[row, col]

    return score, indexes
def upper_bound_bm(edges, nodes1, nodes2):
    '''
        Calculate the upper bound of the bipartite matching
        Input:
        table1/table2: two tables each of which is with a set of column vectors
        threshold: the minimum cosine similarity to include an edge 
        Output:
        The upper bound of the bipartite matching score (no smaller than true score)
    '''
    score = 0.0
    for e in edges:
        score += e[0]
        nodes1.discard(e[1])
        nodes2.discard(e[2])
        if len(nodes1) == 0 or len(nodes2) == 0:
            return score
    return score

def lower_bound_bm(edges, nodes1, nodes2):
    '''
    Output the lower bound of the bipartite matching score (no larger than true score)
    '''
    score = 0.0
    for e in edges:
        if e[1] in nodes1 and e[2] in nodes2:
            score += e[0]
            nodes1.discard(e[1])
            nodes2.discard(e[2])
        if len(nodes1) == 0 or len(nodes2) == 0:
            return score
    return score


def get_edges(table1, table2, threshold):
    '''
    Generate the similarity graph used by lower bounds and upper bounds
    Args:
        table1 (numpy array): the vectors of the query (# rows: # columns in a table, #cols: dimension of embedding)
        table2 (numpy array): similar to table1, set of column vectors of the data lake table
        threshold (float): minimum cosine similarity to include an edge
    Return:
        list of edges and sets of nodes used in lower and upper bounds calculations
    '''
    nrow = len(table1)
    ncol = len(table2)
    edges = []
    nodes1 = set()
    nodes2 = set()
    for i in range(nrow):
        for j in range(ncol):
            sim = cosine_sim(table1[i],table2[j])
            if sim > threshold:
                edges.append((sim,i,j))
                nodes1.add(i)
                nodes2.add(j)
    edges.sort(reverse=True)
    return edges, nodes1, nodes2