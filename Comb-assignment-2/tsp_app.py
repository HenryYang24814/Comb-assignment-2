import sys
import math
from graph_reader import read_graph
from mst import MST
from DFS import DFS


def tsp_app(G, n):
    """
    Computes the TSP approximation order for a single graph
    Returns only the node visitation sequence (0-indexed)
    """
    # 1. Compute the MST using Prim's algorithm
    mst_matrix = MST(G, n)

    # 2. Perform DFS on the MST to get the pre-order traversal
    # Starting at vertex 0
    tsp_order = DFS(mst_matrix, [0])

    return tsp_order


def calculate_tour_weight(G, order):
    """
    Calculates total weight of the cycle, including return to start.
    """
    total_dist = 0
    cycle = order + [order[0]]
    for i in range(len(cycle) - 1):
        u, v = cycle[i], cycle[i + 1]
        weight = G[u][v]
        if weight == math.inf:
            return None  # Indicates a broken tour (missing edges)
        total_dist += weight
    return total_dist


def main():
    if len(sys.argv) != 2:
        print("Usage: python tsp_approximation.py <input_file>")
        sys.exit(1)

    # Use graph_reader to get adjacency matrices
    graphs = read_graph(sys.argv[1])

    if not graphs:
        print("No valid graphs found.")
        return

    for idx, (n, G) in enumerate(graphs, 1):
        print(f"--- Graph {idx} (n={n}) ---")
        try:
            # Get the raw order from the clean function
            order = tsp_app(G, n)

            # Format results for display (1-based indexing)
            display_order = [v + 1 for v in order]
            display_cycle = display_order + [display_order[0]]

            # Calculate weight
            weight = calculate_tour_weight(G, order)

            print(f"Approximation Order: {display_order}")
            print(f"Full TSP Cycle:      {display_cycle}")
            if weight is not None:
                print(f"Total Tour Weight:   {weight}")
            else:
                print("Total Tour Weight:   Incomplete (Missing Edges)")
            print("-" * 30)

        except Exception as e:
            print(f"Error processing graph {idx}: {e}")


if __name__ == "__main__":
    main()