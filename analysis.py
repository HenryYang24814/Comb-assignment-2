import time
import sys
import matplotlib.pyplot as plt
import tsp
from tsp import minout
from graph_reader import read_graph

# --- 1. 硬编码基准数据 (避免重复耗时计算) ---
# 格式: { N: (Exact_Time_Seconds, Optimal_Cost) }
BASELINE_DATA = {
    12: (2.884, 289),
    13: (39.94, 334),
    14: (182.32, 344),
    15: (347.0, 295),
}

def run_analysis_with_plot():
    # 默认 k=0.37，也可以通过命令行 python analysis.py 0.5 传入
    if len(sys.argv) > 1:
        try:
            target_k = float(sys.argv[1])
        except ValueError:
            target_k = 0.37
    else:
        target_k = 0.37

    filename = "completegraph.txt"
    print(f"Reading graphs from {filename}...")
    graphs = read_graph(filename)
    
    if not graphs:
        print("No graphs found.")
        return

    # 用于绘图的数据存储
    # 结构: { n: {'k_values': [], 'errors': []} }
    plot_data = {}

    print(f"\n{'='*30} Truncated B&B Analysis (Max N=15) {'='*30}")
    print(f"Target k (red line) = {target_k}")
    print(f"{'N':<4} | {'k':<5} | {'Limit(s)':<10} | {'Cost':<8} | {'Opt Cost':<8} | {'Error %':<10} | {'Status'}")
    print("-" * 85)

    # 定义我们要测试的一系列 k 值 (用于画曲线)
    # 我们会把用户输入的 target_k 也加入进去一起算
    test_k_values = sorted(list(set([0.05, 0.1, 0.2, 0.3, 0.37, 0.5, 0.75, 1.0, target_k])))

    for n, graph in graphs:
        # 1. 过滤：只处理 N=12 到 15 的图
        if n not in BASELINE_DATA:
            continue
        
        # 初始化该 N 的绘图数据
        plot_data[n] = {'k_values': [], 'errors': []}
        
        # 获取基准数据
        exact_time, opt_cost = BASELINE_DATA[n]

        # 2. 遍历不同的 k 值进行测试
        for k in test_k_values:
            time_limit = k * exact_time
            
            # 调用 tsp.py 中的 backtracking2 (带时间限制)
            # 注意：这里我们只关心 cost，不需要 tour
            start_run = time.time()
            res_cost, _ = tsp.backtracking2(
                graph, 
                bounding=minout, 
                time_limit=time_limit
            )
            actual_run_time = time.time() - start_run

            # 3. 计算误差
            if res_cost == float('inf'):
                error = 100.0 # 没找到解
                status = "No Path"
            else:
                error = (res_cost - opt_cost) / opt_cost * 100
                status = "OK"

            # 存入绘图数据
            plot_data[n]['k_values'].append(k)
            plot_data[n]['errors'].append(error)

            # 仅打印用户指定的 target_k 的详细日志，避免刷屏，或者打印所有也行
            # 这里选择打印 target_k 的结果以供表格展示
            if k == target_k:
                 print(f"{n:<4} | {k:<5.2f} | {time_limit:<10.4f} | {res_cost:<8.1f} | {opt_cost:<8.1f} | {error:<9.2f}% | {status}")

    # --- 4. 绘图逻辑 ---
    print("\nGenerating plot...")
    plt.figure(figsize=(10, 6))
    
    # 定义不同 N 的颜色，区分度高一点
    colors = {12: 'blue', 13: 'orange', 14: 'green', 15: 'purple'}
    
    for n, data in plot_data.items():
        plt.plot(
            data['k_values'], 
            data['errors'], 
            marker='o', 
            label=f'N={n} (Opt Time={BASELINE_DATA[n][0]}s)',
            color=colors.get(n, 'black')
        )

    # 画红色虚线 (User Target k)
    plt.axvline(x=target_k, color='red', linestyle='--', linewidth=2, label=f'Chosen k={target_k}')

    plt.title(f'TSP Error Rate vs Time Limit Fraction (k)\n(Baseline: Exact B&B using minout)')
    plt.xlabel('k (Time Fraction: t = k * ExactTime)')
    plt.ylabel('Error Relative to Optimal (%)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(bottom=-1) # Y轴从-1开始，稍微留点空隙
    
    # 保存图片并显示
    plt.savefig('tsp_k_analysis.png')
    print("Plot saved as 'tsp_k_analysis.png'")
    plt.show()

if __name__ == "__main__":
    run_analysis_with_plot()