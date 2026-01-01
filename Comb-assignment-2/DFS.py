# file: DFS.py
"""
Depth-First Search (DFS) module for tree traversal.
Simple, focused implementation following the pseudocode.
"""

import math


def DFS(T, visited):
    """
    Depth-First Search on a tree using recursive strategy.

    Args:
        T: Adjacency matrix of a tree (n x n)
        visited: Sequence of visited nodes (list of integers)

    Returns:
        order: Sequence of nodes visited in DFS order
    """
    # order ← visited
    order = visited.copy()

    # If visited is empty, return empty list
    if not visited:
        return order

    # current ← visited[-1] (last element)
    current = visited[-1]
    n = len(T)

    # Get vertices not in visited
    # Create set for O(1) lookup
    visited_set = set(visited)

    # for neighbor ∈ ℕ \ visited do
    for neighbor in range(n):
        if neighbor not in visited_set:
            # if T[current][neighbor] ≠ ∞ then
            if T[current][neighbor] != math.inf:
                # order ← order + neighbor
                order.append(neighbor)

                # order ← DFS(T, order)
                order = DFS(T, order)

    return order


if __name__ == "__main__":
    """
    Simple test when run directly.
    """
    # Test with a small tree
    test_tree = [
        [0, 1, math.inf],
        [1, 0, 1],
        [math.inf, 1, 0]
    ]

    print("Testing DFS function...")
    order = DFS(test_tree, [0])
    print(f"DFS order: {order}")
    print(f"Expected: [0, 1, 2]")