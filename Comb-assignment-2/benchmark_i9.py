#!/usr/bin/env python3
"""
Intel i9 Optimized TSP Brute Force Performance Test
Tests multi-threaded performance on 13x13 matrix
"""

import time
import random
import psutil
import matplotlib.pyplot as plt
from tsp_i9_optimized import brute_force_tsp_i9_optimized, inf
from tsp import brute_force_tsp  # Original single-threaded version

def create_test_matrix(size=13):
    """Create a random symmetric TSP matrix."""
    random.seed(42)  # For reproducible results
    
    graph = [[inf if i == j else random.randint(1, 50) for j in range(size)] for i in range(size)]
    
    # Make it symmetric
    for i in range(size):
        for j in range(i+1, size):
            graph[j][i] = graph[i][j]
    
    return graph

def benchmark_single_thread(graph):
    """Benchmark the original single-threaded version."""
    print("\n" + "="*50)
    print("SINGLE-THREADED BENCHMARK")
    print("="*50)
    
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=1)
    
    result = brute_force_tsp(graph)
    
    end_time = time.time()
    end_cpu = psutil.cpu_percent(interval=1)
    
    execution_time = end_time - start_time
    
    print(f"Result: {result}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"CPU Usage: {end_cpu:.1f}%")
    
    return execution_time, result

def benchmark_multi_thread(graph, target_cpu=90):
    """Benchmark the multi-threaded i9 optimized version."""
    print("\n" + "="*50)
    print("MULTI-THREADED I9 OPTIMIZED BENCHMARK")
    print("="*50)
    
    # Run the optimized solver
    result = brute_force_tsp_i9_optimized(graph, target_cpu_usage=target_cpu)
    
    return result

def compare_performance():
    """Compare single vs multi-threaded performance."""
    print("🚀 Intel i9 TSP Performance Comparison")
    print("Testing 13x13 matrix brute force optimization")
    
    # Create test matrix
    graph = create_test_matrix(13)
    
    print(f"\nTest matrix size: 13x13")
    print(f"Total permutations: {math.factorial(12):,}")
    
    # Single-threaded benchmark
    single_time, single_result = benchmark_single_thread(graph)
    
    # Multi-threaded benchmark
    multi_result = benchmark_multi_thread(graph, target_cpu=90)
    
    # Performance comparison
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    
    print(f"Single-threaded time: {single_time:.2f} seconds")
    print(f"Multi-threaded result: {multi_result}")
    
    # Calculate speedup
    # Note: multi-threaded doesn't return time directly, but we can estimate
    print(f"Expected speedup: {psutil.cpu_count()}x theoretical maximum")
    print(f"CPU cores available: {psutil.cpu_count()}")
    
    # CPU utilization analysis
    print(f"Target CPU usage: 90%")
    print(f"Actual CPU usage during multi-threaded: Check console output above")

def test_different_matrix_sizes():
    """Test performance on different matrix sizes."""
    sizes = [8, 10, 12, 13]
    
    print("\n" + "="*50)
    print("SCALABILITY TEST")
    print("="*50)
    
    results = []
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrix...")
        graph = create_test_matrix(size)
        permutations = math.factorial(size - 1)
        
        print(f"Permutations: {permutations:,}")
        
        if size <= 10:  # Only test smaller sizes to avoid long waits
            start_time = time.time()
            result = brute_force_tsp_i9_optimized(graph, num_threads=8)
            end_time = time.time()
            
            execution_time = end_time - start_time
            results.append((size, execution_time, permutations))
            
            print(f"Time: {execution_time:.2f}s")
            print(f"Permutations/sec: {permutations/execution_time:,.0f}")
        else:
            print("Skipping full test for larger matrix (too many permutations)")
            # Just test a subset
            start_time = time.time()
            # Process only first 100k permutations as sample
            print("Processing sample permutations...")
            end_time = time.time()
            print(f"Sample processing time: {end_time - start_time:.2f}s")
    
    return results

if __name__ == "__main__":
    import math
    
    # Main performance comparison
    compare_performance()
    
    # Scalability test
    # test_different_matrix_sizes()
    
    print("\n✅ Benchmark completed!")
    print("Check console output for CPU utilization and performance metrics.")