"""
 MST and DFS algo
"""

import math

def DFS(T, visited):
    """
    recursively DFS algo, each return is back to the last level( parent),
    and we use a list named visited to store processed nodes, so that we do not need to delete duplicates

    Args:
        T: Adjacency matrix of a tree (size of n x n)
        visited: Sequence of visited nodes (list of integers)

    Returns:
        DFS order
    """
    # order ← visited
    order = visited.copy()

    if not visited:
        return order

    # current ← visited[-1] (last element)
    current = visited[-1]
    n = len(T)

    # Get vertices not in visited using set for O(1) lookup
    visited_set = set(visited)

    # for neighbor ∈ ℕ \ visited do
    for neighbor in range(n):
        if neighbor not in visited_set:
            # Not leaf
            if T[current][neighbor] != math.inf:
                # order ← order + neighbor
                order.append(neighbor)

                order = DFS(T, order)

    return order


def MST(G, n):
    """
    Build Minimum Spanning Tree using Prim's algo discussed in class

    Args:
        G: Adjacency matrix
        n: Number of vertices

    Returns:
        tau: Adjacency matrix for the built MST
    """
    tau = [[math.inf] * n for _ in range(n)]

    # Set diagonal to 0
    for i in range(n):
        tau[i][i] = 0

    # T ← {0}
    T = set([0])
    all_vertices = set(range(n))

    # while not all nodes processed
    while T != all_vertices:
        # i, j ← 0, 0 (initialize with invalid indices)
        i, j = -1, -1
        min_weight = math.inf

        for s in T:
            # forall t ∈ V \ T do
            for t in all_vertices - T:
                if G[s][t] < min_weight:
                    min_weight = G[s][t]
                    i, j = s, t

        # Undirected graph, so symmetric adj matrix
        tau[i][j] = G[i][j]
        tau[j][i] = G[i][j]

        # Add as processed
        T.add(j)

    return tau

