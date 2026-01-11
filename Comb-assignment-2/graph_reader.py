import math
import sys


def read_graph(filename):
    """
    Convert line representation of graph into adjacency matrix

    Args:
         several lines of number sequences, each line denote a graph

    Returns:
        List of tuples (n, graph) where:
        - n: number of vertices
        - graph: 2D adjacency matrix
    """
    graphs = []

    try:
        # Try reading with UTF-8 encoding first, then fallback to latin-1
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(filename, 'r', encoding='latin-1') as f:
                lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue

            parts = line.split()
            if not parts:
                continue

            # Parse number of vertices
            try:
                n = int(parts[0])
            except ValueError:
                print(f"Error: Line {line_num}: Invalid vertex count '{parts[0]}', skipping graph")
                continue

            # Validate: n must be at least 1
            if n < 1:
                print(f"Error: Line {line_num}: Number of vertices must be ≥ 1, got {n}, skipping graph")
                continue

            weights = parts[1:]
            expected_weights = n * (n - 1) // 2

            if len(weights) != expected_weights:
                print(
                    f"Error: Line {line_num}: Expected {expected_weights} weights for {n} vertices, "
                    f"but got {len(weights)}. Skipping graph.")
                continue  # Skip this invalid graph entirely

            # Create adjacency matrix filled with infinity
            graph = [[math.inf] * n for _ in range(n)]

            # Set diagonal to 0 (distance to self)
            for i in range(n):
                graph[i][i] = 0

            # Fill the upper triangle from the input data
            idx = 0
            valid_graph = True  # Flag to track if graph is valid

            for i in range(n):
                for j in range(i + 1, n):
                    weight_str = weights[idx]

                    # Only accept 'inf' as the special marker for no edge
                    if weight_str.lower() == 'inf':
                        weight = math.inf
                    else:
                        # Try to convert to float
                        try:
                            weight = float(weight_str)
                        except ValueError:
                            print(f"Error: Line {line_num}: Invalid weight '{weight_str}'. "
                                  f"Must be 'inf' or a number. Skipping graph.")
                            valid_graph = False
                            break

                        # Check for negative weight
                        if weight < 0:
                            print(f"Error: Line {line_num}: Negative weight {weight} at edge ({i + 1},{j + 1}). "
                                  f"Dijkstra requires non-negative weights. Skipping graph.")
                            valid_graph = False
                            break

                    graph[i][j] = weight
                    graph[j][i] = weight  # Undirected graph so symmetric
                    idx += 1

                if not valid_graph:
                    break  # Break out of outer loop too

            if valid_graph:
                graphs.append((n, graph))

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    return graphs