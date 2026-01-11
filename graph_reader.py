import math


def read_graph(filename):
    """
    Convert line representation of graph into adjacency matrix
    Almost same as assignment1, in A1, it can read a file with several graphs
    so it can also process the files with single graph

    Returns:
        - n: number of vertices
        - graph: adjacency matrix with size n * n
    """
    graphs = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            n = int(parts[0])  # number of vertices

            # check the number of weights
            weights = parts[1:]
            expected_weights = n * (n - 1) // 2

            if len(weights) != expected_weights:
                print(f"Invalid graph: expected {expected_weights} weights, but got {len(weights)}")
                continue

            # Initialize
            graph = [[math.inf] * n for _ in range(n)]
            for i in range(n):
                graph[i][i] = 0 # diagonal all zero, each node to itself is 0 in simple graphs

            # Fill the upper triangle and reflex to make it symmetric
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    w_str = weights[idx].lower()
                    weight = math.inf if w_str == 'inf' else float(w_str)

                    graph[i][j] = weight
                    graph[j][i] = weight
                    idx += 1

            graphs.append((n, graph))

    return graphs