import time
# 从 tsp.py 导入所有需要的函数和常量
from tsp import backtracking, minout, two_minout, brute_force_tsp, inf

def main():

    graph = [
        [inf, 1, 3, 5, 8],
        [1, inf, 4, 2, 9],
        [3, 4, inf, 7, 2],
        [5, 2, 7, inf, 2],
        [8, 9, 2, 2, inf]
    ]


    print(f"{'alg':<10} | {'cost':<5} | {'tour':<25} | {'elapsed time':<15}")
    print("-" * 65)

    # ---  bf  ---
    start_bf = time.time()
    res_bf = brute_force_tsp(graph)
    end_bf = time.time()
    tour_bf = "->".join([chr(65 + i) for i in res_bf[1]])
    print(f"{'brute':<10} | {res_bf[0]:<5} | {tour_bf:<25} | {end_bf - start_bf:.4f}s")

    # --- minout ---
    start_minout = time.time()
    res_minout = backtracking(graph=graph, path=[0], shortest=inf, best_path=[], bounding=minout)
    end_minout = time.time()
    tour_minout = "->".join([chr(65 + i) for i in res_minout[1]])
    print(f"{'minout':<10} | {res_minout[0]:<5} | {tour_minout:<25} | {end_minout - start_minout:.4f}s")

    # --- mintwo ---
    start_min2 = time.time()
    res_min2 = backtracking(graph=graph, path=[0], shortest=inf, best_path=[], bounding=two_minout)
    end_min2 = time.time()
    tour_min2 = "->".join([chr(65 + i) for i in res_min2[1]])
    print(f"{'mintwo':<10} | {res_min2[0]:<5} | {tour_min2:<25} | {end_min2 - start_min2:.4f}s")

    # --- print ---
    print("\n" + "="*50)
    # 再次运行以获取结果对象（或直接复用 res_minout）
    result = backtracking(graph=graph, path=[0], shortest=inf, best_path=[], bounding=minout)
    print(f"TSP({graph}) = {result}")



# 13x13 
    graph_13 = [
        [inf, 12, 10, 19, 8, 18, 2, 13, 5, 14, 1, 11, 7],
        [12, inf, 15, 6, 9, 3, 17, 4, 20, 16, 8, 2, 11],
        [10, 15, inf, 14, 2, 18, 6, 11, 9, 5, 13, 7, 4],
        [19, 6, 14, inf, 12, 1, 15, 10, 8, 3, 17, 9, 2],
        [8, 9, 2, 12, inf, 11, 7, 14, 6, 19, 4, 15, 10],
        [18, 3, 18, 1, 11, inf, 12, 5, 17, 8, 9, 14, 6],
        [2, 17, 6, 15, 7, 12, inf, 19, 4, 11, 10, 8, 13],
        [13, 4, 11, 10, 14, 5, 19, inf, 12, 6, 2, 18, 9],
        [5, 20, 9, 8, 6, 17, 4, 12, inf, 15, 7, 11, 3],
        [14, 16, 5, 3, 19, 8, 11, 6, 15, inf, 12, 10, 1],
        [1, 8, 13, 17, 4, 9, 10, 2, 7, 12, inf, 19, 5],
        [11, 2, 7, 9, 15, 14, 8, 18, 11, 10, 19, inf, 6],
        [7, 11, 4, 2, 10, 6, 13, 9, 3, 1, 5, 6, inf]
    ]

    print("\n" + "-" * 20 + " 13 Vertices Test " + "-" * 20)

   #======13*13
    start_bt = time.time()
    res_bt = backtracking(graph=graph_13, path=[0], shortest=inf, best_path=[], bounding=two_minout)
    end_bt = time.time()
    tour_bt = "->".join([chr(65 + i) for i in res_bt[1]])
    print(f"{'BT+min2':<10} | {res_bt[0]:<5} | {tour_bt:<25} | {end_bt - start_bt:.4f}s")

    start_minout = time.time()
    res_minout = backtracking(graph=graph_13, path=[0], shortest=inf, best_path=[], bounding=minout)
    end_minout = time.time()
    tour_minout = "->".join([chr(65 + i) for i in res_minout[1]])
    print(f"{'minout':<10} | {res_minout[0]:<5} | {tour_minout:<35} | {end_minout - start_minout:.4f}s")

    # brute force
    run_brute = True #
    if run_brute:
        print("Running Brute Force for 13 points... this might take a LONG time.")
        start_bf = time.time()
        res_bf = brute_force_tsp(graph_13)
        end_bf = time.time()
        print(f"{'Brute':<10} | {res_bf[0]:<5} | {'Done':<25} | {end_bf - start_bf:.4f}s")


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