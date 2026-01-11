# Author: Yiling Liu
import math
from graph_reader import read_graph


def MST(G, n):
    """
    Compute Minimum Spanning Tree using Prim's algorithm.

    Args:
        G: Adjacency matrix of undirected weighted graph
        n: Number of vertices

    Returns:
        tau: Adjacency matrix of the MST
    """
    # tau ← new matrix n × n
    tau = [[math.inf] * n for _ in range(n)]

    # Set diagonal to 0
    for i in range(n):
        tau[i][i] = 0

    # T ← {0}
    T = set([0])
    all_vertices = set(range(n))

    # while T ≠ V do
    while T != all_vertices:
        # i, j ← 0, 0 (initialize with invalid indices)
        i, j = -1, -1
        min_weight = math.inf

        # forall s ∈ T do
        for s in T:
            # forall t ∈ V \ T do
            for t in all_vertices - T:
                if G[s][t] < min_weight:
                    min_weight = G[s][t]
                    i, j = s, t

        # τ[i][j], τ[j][i] ← G[i][j], G[i][j]
        tau[i][j] = G[i][j]
        tau[j][i] = G[i][j]

        # T ← T ∪ {j}
        T.add(j)

    return tau


def print_mst(tau, n):
    """
    Print MST edges and regarding weights, help for test

    Args:
        tau: Adjacency matrix of MST
        n: Number of vertices
    """
    print(f"Minimum Spanning Tree (n={n}):")
    print("Edge\tWeight")
    total_weight = 0

    # Only print upper triangle to avoid duplicates
    for i in range(n):
        for j in range(i + 1, n):
            if tau[i][j] != math.inf and tau[i][j] != 0:
                print(f"{i + 1}-{j + 1}\t{tau[i][j]}")
                total_weight += tau[i][j]

    print(f"Total weight: {total_weight}")
    print()


def main():
    """
    Main function testing read graphs and construct MST
    """
    if len(sys.argv) != 2:
        print("Usage: python mst.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    # Read graphs from file
    graphs = read_graph(input_file)

    if not graphs:
        print("No valid graphs found in the file.")
        return

    print(f"Found {len(graphs)} graph(s) in the file.\n")

    # Process each graph
    for graph_idx, (n, graph) in enumerate(graphs, 1):
        print(f"--- Graph {graph_idx} (n={n}) ---")

        # Compute MST
        try:
            mst_matrix = MST(graph, n)
            print_mst(mst_matrix, n)
        except Exception as e:
            print(f"Error computing MST for graph {graph_idx}: {e}\n")
            continue


if __name__ == "__main__":
    import sys

    main()