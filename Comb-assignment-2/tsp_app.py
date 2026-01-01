import sys
import math
from graph_reader import read_graph
from mst import MST
from DFS import DFS


def solve_tsp_approx(input_file):
    # 1. Read graphs from the file
    graphs = read_graph(input_file)

    if not graphs:
        print("No valid graphs found.")
        return

    for idx, (n, G) in enumerate(graphs, 1):
        print(f"--- Processing Graph {idx} (n={n}) ---")

        try:
            # 2. Compute the MST using Prim's algorithm
            # This returns the adjacency matrix of the MST
            mst_matrix = MST(G, n)

            # 3. Perform DFS on the MST to get the pre-order traversal
            # We start at vertex 0 (represented as [0] in the visited list)
            tsp_order = DFS(mst_matrix, [0])

            # 4. To represent a full tour, we add the start node at the end
            # (Optional, but standard for TSP cycles)
            tsp_cycle = tsp_order + [tsp_order[0]]

            # Convert to 1-based indexing for display (standard for these exercises)
            display_order = [v + 1 for v in tsp_order]
            display_cycle = [v + 1 for v in tsp_cycle]

            print(f"Approximation Order: {display_order}")
            print(f"Full TSP Cycle:      {display_cycle}")

            # Calculate total distance of the tour
            total_dist = 0
            for i in range(len(tsp_cycle) - 1):
                u, v = tsp_cycle[i], tsp_cycle[i + 1]
                weight = G[u][v]
                if weight == math.inf:
                    print(f"Warning: Edge ({u + 1}, {v + 1}) does not exist in original graph!")
                else:
                    total_dist += weight

            print(f"Total Tour Weight:   {total_dist}")
            print("-" * 30)

        except Exception as e:
            print(f"Error processing graph {idx}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tsp_approximation.py <input_file>")
        sys.exit(1)

    solve_tsp_approx(sys.argv[1])