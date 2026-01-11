import matplotlib.pyplot as plt
import tsp
import time
from graph_reader import read_graph
from tsp import minout

# Baseline optimal costs and total times from previous runs
BASELINE_DATA = {
    12: (2.884, 289),
    13: (39.94, 334),
    14: (182.32, 344),
    15: (347.0, 295)
}

def plot_convergence():
    filename = "completegraph.txt"
    print(f"Reading graphs from {filename}...")
    graphs = read_graph(filename)
    
    if not graphs:
        print("No graphs found.")
        return

    plt.figure(figsize=(14, 8))
    
    colors = {12: 'tab:blue', 13: 'tab:orange', 14: 'tab:green', 15: 'tab:red'}
    
    # Iterate through graphs and run B&B with history logging
    for idx, (n, graph) in enumerate(graphs):
        if n not in BASELINE_DATA:
            continue
            
        print(f"Running B&B for Graph N={n}...")
        
        # Setup for recording history
        history = [] # Will store (time, cost)
        
        # We use the total time from baseline to limit the run (or just run it fully if feasible)
        # To get the full curve up to 1.0, we should run it for at least the estimated time.
        # But since we want to plot *Search Progress Ratio*, we need the actual total time of THIS run.
        # So we run it fully (or with a large enough limit that is effectively full for these N)
        # N=15 takes ~6 mins. We can run it.
        
        total_time_estimate, optimal_cost = BASELINE_DATA[n]
        
        # Run algorithm
        start_time = time.time()
        final_cost, _ = tsp.backtracking2(
            graph, 
            bounding=minout, 
            history=history,
            time_limit=total_time_estimate * 1.2 # Give it a bit more buffer to finish naturally if estimate is off
        )
        actual_total_time = time.time() - start_time
        
        # Print raw history data
        print(f"--- Raw History Data for N={n} ---")
        print(f"Final Cost: {final_cost}")
        print(f"Total Time: {actual_total_time:.4f}s")
        print("Update History (Time, Cost):")
        for t, c in history:
            print(f"  t={t:.4f}s, cost={c}")
        print("-" * 30)

        # If the run finished early (found optimum and explored all), actual_total_time is the real 100%
        # If it hit time limit, then that's our 100% reference for the plot.
        
        # Process history data for plotting
        # History contains (t, cost). We need (t/T_total, cost/Opt_cost)
        
        # Add initial point (t=0, cost=initial_found_cost or just start from first update)
        # And add final point
        if not history:
            # If no history recorded (very fast or no updates), just plot start and end
            history = [(0, final_cost), (actual_total_time, final_cost)]
        else:
            # Ensure the last point extends to the end of the run
            last_t, last_cost = history[-1]
            if last_t < actual_total_time:
                history.append((actual_total_time, last_cost))
                
        # Prepare X and Y arrays for step plot
        x_values = []
        y_values = []
        
        # Add a starting point at t=0 with the first found cost (or a high value if unknown, but history[0] is best guess)
        # Actually, let's just plot the history points.
        
        # To make it a "step" plot that looks like the image (dropping down):
        # We can use plt.step or manually construct points.
        # The image has markers at the "update" points.
        
        for t, cost in history:
            ratio_t = t / actual_total_time
            ratio_cost = cost / optimal_cost
            
            # Clip x to max 1.0
            if ratio_t > 1.0: ratio_t = 1.0
            
            x_values.append(ratio_t)
            y_values.append(ratio_cost)
            
        # Plot
        # 'where="post"' means the value remains constant until the next change? 
        # Actually, when we find a new cost at time t, the cost drops INSTANTLY at time t.
        # So from previous_t to t, the cost was previous_cost.
        # At t, it becomes new_cost.
        # So 'pre' step is appropriate? Or just connect points if they are dense.
        # The image shows vertical drops.
        
        # Let's construct a step-like sequence for plotting with markers at corners
        plot_x = []
        plot_y = []
        
        if history:
            # Start: t=0, cost = first_cost (or infinite/high?)
            # Usually the first solution is found quickly. Let's assume history[0] is the first solution.
            # Before history[0], we don't have a solution?
            # We can start plotting from the first found solution.
            
            prev_x = history[0][0] / actual_total_time
            prev_y = history[0][1] / optimal_cost
            
            plot_x.append(prev_x)
            plot_y.append(prev_y)
            
            for i in range(1, len(history)):
                curr_t, curr_cost = history[i]
                curr_x = curr_t / actual_total_time
                curr_y = curr_cost / optimal_cost
                
                if curr_x > 1.0: curr_x = 1.0
                
                # Horizontal line from prev_x to curr_x with height prev_y
                plot_x.append(curr_x)
                plot_y.append(prev_y)
                
                # Vertical drop to curr_y
                plot_x.append(curr_x)
                plot_y.append(curr_y)
                
                prev_x = curr_x
                prev_y = curr_y
                
        # Filter markers: we want markers only at the "new path found" events (the corners)
        # These correspond to the original (x, y) pairs from history
        marker_x = [t/actual_total_time for t, c in history]
        marker_y = [c/optimal_cost for t, c in history]
        
        # Main line
        plt.plot(plot_x, plot_y, color=colors[n], linewidth=2, label=f"Graph {idx+1} (N={n})")
        
        # Markers
        plt.scatter(marker_x, marker_y, color=colors[n], s=30, alpha=0.6, edgecolors='white')

    # Add reference line for Optimal (y=1.0)
    plt.axhline(y=1.0, color='gray', linestyle='--', label='Optimal Threshold')
    
    plt.title("TSP Search Convergence (MST Bound)", fontsize=16)
    plt.xlabel("Search Progress Ratio (Current Time / Total Time)", fontsize=12)
    plt.ylabel("Error Ratio (Current Cost / Optimal)", fontsize=12)
    plt.xlim(0, 1.05)
    # plt.ylim(bottom=1.0) # Adjust as needed
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    
    output_file = 'tsp_convergence_plot.png'
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_convergence()
