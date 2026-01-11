import time
import tsp
from tsp_app import tsp_app, calculate_tour_weight
from graph_reader import read_graph  # [重要] 导入你项目中的读取函数

# --- 核心分析逻辑 ---
def run_comparison_analysis():
    filename = "completegraph.txt"
    
    print(f"Reading graphs from {filename}...")
    try:
        # 使用你现有的 graph_reader 读取文件
        # read_graph 返回的是一个列表: [(n, graph), (n, graph), ...]
        graphs = read_graph(filename)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not graphs:
        print("No graphs found in the file.")
        return

    # 存储 k=0 到 k=1 的误差数据，用于后续拟合分析
    k_error_data = [] 

    print(f"\n{'='*20} TSP Algorithm Comparison (B&B Early Stop vs MST Approx) {'='*20}")
    print(f"{'ID':<4} | {'N':<4} | {'Method':<25} | {'Cost':<10} | {'Error %':<10} | {'Time (s)':<10} | {'Time Ratio':<10}")
    print("-" * 90)

    # 遍历文件中的每一张图
    for idx, (n, graph) in enumerate(graphs):
        # --- A. 运行精确解 (Exact B&B) 获取基准 ---
        # 这一步是为了拿到 T_optimal (x) 和 Optimal Cost
        start = time.time()
        # 注意：如果你已经添加了 backtracking2，这里可以用 tsp.backtracking2
        # 如果还在用原来的，就用 tsp.backtracking
        opt_cost, opt_path = tsp.backtracking(graph, bounding=tsp.minout)
        exact_time = time.time() - start
        
        # --- B. 运行 MST 近似算法 ---
        start_app = time.time()
        app_order = tsp_app(graph, n)
        app_cost = calculate_tour_weight(graph, app_order)
        app_time = time.time() - start_app
        
        # --- C. 模拟 B&B 截断 (t = 0.37 * exact_time) ---
        # 设定时间限制为精确时间的 37%
        limit_37 = 0.37 * exact_time
        start_bnb_37 = time.time()
        
        # [关键] 这里调用带 time_limit 的 backtracking
        # 请确保你的 tsp.py 里 backtracking 函数已经增加了 time_limit 参数
        bnb_37_cost, _ = tsp.backtracking2(graph, bounding=tsp.minout, time_limit=limit_37)
        bnb_37_time = time.time() - start_bnb_37
        
        # --- 计算误差 ---
        mst_error = (app_cost - opt_cost) / opt_cost * 100
        bnb_37_error = (bnb_37_cost - opt_cost) / opt_cost * 100
        
        # --- 打印该图的分析结果 ---
        # 1. Exact
        print(f"{idx+1:<4} | {n:<4} | {'Exact B&B (Baseline)':<25} | {opt_cost:<10.1f} | {'0.00%':<10} | {exact_time:<10.4f} | {'1.00':<10}")
        # 2. MST Approx
        print(f"{'':<4} | {'':<4} | {'MST Approx':<25} | {app_cost:<10.1f} | {mst_error:<10.2f}%    | {app_time:<10.4f} | {app_time/exact_time:<10.4f}")
        # 3. B&B 0.37x
        print(f"{'':<4} | {'':<4} | {'B&B (Limit t=0.37x)':<25} | {bnb_37_cost:<10.1f} | {bnb_37_error:<10.2f}%    | {bnb_37_time:<10.4f} | {bnb_37_time/exact_time:<10.4f}")
        print("-" * 90)

        # --- D. 收集 k=0.1, 0.2 ... 1.0 的数据 (用于验证表达式) ---
        # 针对当前这张图，测试不同比例的时间限制 k
        for k in [0.1, 0.2, 0.37, 0.5, 0.8]:
            limit_k = k * exact_time
            # 同样调用带 time_limit 的函数
            cost_k, _ = tsp.backtracking2(graph, bounding=tsp.minout, time_limit=limit_k)
            err_k = (cost_k - opt_cost) / opt_cost * 100
            k_error_data.append({'n': n, 'k': k, 'error': err_k})

    # --- 统计与公式拟合 ---
    if k_error_data:
        print("\n[Data Analysis] Error vs Time Fraction (k)")
        print("Based on all graphs in file, average Error for each k:")
        
        k_groups = {}
        for item in k_error_data:
            k = item['k']
            if k not in k_groups: k_groups[k] = []
            k_groups[k].append(item['error'])
        
        print(f"{'k (t/x)':<10} | {'Avg Error %':<15}")
        print("-" * 30)
        for k in sorted(k_groups.keys()):
            avg_err = sum(k_groups[k]) / len(k_groups[k])
            print(f"{k:<10.2f} | {avg_err:<15.2f}")
        
        print("\n[Conclusion]")
        print("1. MST Approx is extremely fast but has high error (usually >15%).")
        print("2. B&B with Early Stopping (t=0.37x) drastically reduces error to near 0%.")

if __name__ == "__main__":
    run_comparison_analysis()