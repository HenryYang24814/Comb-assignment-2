import time
import sys
import matplotlib.pyplot as plt
# Import functions from tsp.py (including the new mst_bound)
from tsp import backtracking, minout, two_minout, mst_bound, brute_force_tsp, tsp_approximation, inf
# Import the file reader
from graph_reader import read_graph


def run_test(name, func, *args, **kwargs):
    """Helper to run a solver and measure time."""
    start = time.perf_counter()
    result_cost, result_path = func(*args, **kwargs)
    end = time.perf_counter()

    # Format the path for display (A->B->C...)
    if result_path and isinstance(result_path, list):
        path_str = "->".join([str(i) for i in result_path]) # Using numbers as user context implies
    else:
        path_str = "No path found"

    return result_cost, path_str, end - start

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <graph_file.txt>")
        return

    input_file = sys.argv[1]
    graphs = read_graph(input_file)

    if not graphs:
        print("No valid graphs found in the input file.")
        return

    for idx, (n, G) in enumerate(graphs, 1):
        print(f"\n--- Graph {idx} (Nodes: {n}) ---")

        # 1. Approximation
        cost_ap, tour_ap, time_ap = run_test("Approx", tsp_approximation, G, n)
        print(f"Approximation: Cost {cost_ap}, Time {time_ap:.6f}s, Path: {tour_ap}")

        # 2. Backtracking with minout bound
        if n <= 14:
            cost_mo, tour_mo, time_mo = run_test("BT + minout", backtracking, G, bounding=minout)
            print(f"BT + minout:   Cost {cost_mo}, Time {time_mo:.6f}s, Path: {tour_mo}")
        else:
            print(f"BT + minout:   Skipped, (n too large)")
        # 3. Backtracking with two_minout bound
        if n <= 17:
            cost_tm, tour_tm, time_tm = run_test("BT + min2", backtracking, G, bounding=two_minout)
            print(f"BT + min2:     Cost {cost_tm}, Time {time_tm:.6f}s, Path: {tour_tm}")
        else:
            print(f"BT + min2:     Skipped, (n too large)")
        
        # 4. Backtracking with MST bound
        cost_mst, tour_mst, time_mst = run_test("BT + mst_bound", backtracking, G, bounding=mst_bound)
        print(f"BT + min2:   Cost {cost_mst} ,Time {time_mst:.6f}s, Path: {tour_mst}")

        # 5. Brute Force (Only for small graphs)
        if n <= 10:
            cost_bf, tour_bf, time_bf = run_test("Brute Force", brute_force_tsp, G)
            print(f"Brute Force:   Cost {cost_bf}, Time {time_bf:.6f}s, Path: {tour_bf}")
        else:
            print("Brute Force:   Skipped (n too large)")
        if cost_ap == inf or cost_mst == 0:
            error = inf
        else:
            error = cost_ap / cost_mst    
        print(f"Approximation Error: {error}")
        print("\n" + "="*65)
if __name__ == "__main__":
    main()