import time
from tsp import backtracking, minout, two_minout, inf
def main():
    # Define the 5x5 adjacency matrix as shown in the teacher's example
    graph = [
        [inf, 1, 3, 5, 8],
        [1, inf, 4, 2, 9],
        [3, 4, inf, 7, 2],
        [5, 2, 7, inf, 2],
        [8, 9, 2, 2, inf]
    ]

    # Print the table header to match the expected output format
    print(f"{'alg':<10} | {'cost':<5} | {'tour':<25} | {'elapsed time':<15}")
    print("-" * 65)

    # --- Test Algorithm: minout ---
    start_minout = time.time()
    # Execute backtracking with minout bounding function
    res_minout = backtracking(graph=graph, path=[0], shortest=inf, best_path=[], bounding=minout)
    end_minout = time.time()
    # Format the tour as letters (A, B, C...)
    tour_minout = "->".join([chr(65 + i) for i in res_minout[1]])
    print(f"{'minout':<10} | {res_minout[0]:<5} | {tour_minout:<25} | {end_minout - start_minout:.4f}s")

    # --- Test Algorithm: mintwo (two_minout) ---
    start_min2 = time.time()
    # Execute backtracking with two_minout bounding function
    res_min2 = backtracking(graph=graph, path=[0], shortest=inf, best_path=[], bounding=two_minout)
    end_min2 = time.time()
    tour_min2 = "->".join([chr(65 + i) for i in res_min2[1]])
    print(f"{'mintwo':<10} | {res_min2[0]:<5} | {tour_min2:<25} | {end_min2 - start_min2:.4f}s")

    # --- Final Result Output (As per the teacher's screenshot) ---
    # According to the teacher's expectation in Figure 2, print the exact string format
    print("\n" + "="*50)
    result = backtracking(graph=graph, path=[0], shortest=inf, best_path=[], bounding=minout)
    print(f"TSP({graph}) = {result}")

if __name__ == "__main__":
    main()

"""from math import inf
import tsp

def main():

    # graph = [
    #         [inf, 1, 3, 2],
    #         [1, inf, 1, 2],
    #         [3, 1, inf, 5],
    #         [2, 2, 5, inf]
    #     ]

    graph = [
            [0, 2,   3,   8],
            [2, 0,   inf, 7],
            [3, inf, 0,   1],
            [8, 7,   1,   0]
        ]

    result = tsp.backtracking(graph=graph, bounding = tsp.minout)

    print(f'TSP({graph}) = {result}')

if __name__ == "__main__":
    main()"""