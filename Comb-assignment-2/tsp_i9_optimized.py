import math
import time
import concurrent.futures
import multiprocessing
import numpy as np
from numba import jit, prange
import psutil

inf = math.inf

@jit(nopython=True, parallel=True, cache=True)
def calculate_path_distance_numba(path, graph):
    """Optimized path distance calculation using Numba."""
    total = 0.0
    for i in range(len(path) - 1):
        total += graph[path[i], path[i + 1]]
    return total

@jit(nopython=True, cache=True)
def is_valid_path_numba(path, graph):
    """Check if path is valid (no infinite edges)."""
    for i in range(len(path) - 1):
        if graph[path[i], path[i + 1]] >= 1e10:  # inf in numpy
            return False
    return True

def generate_permutations_chunk_optimized(n, start_rank, end_rank):
    """Generate permutations chunk using Trotter-Johnson algorithm."""
    from permutation import trotter_johnson_unrank
    
    chunk_data = []
    for rank in range(start_rank, end_rank):
        perm = trotter_johnson_unrank(n - 1, rank)
        path = [0] + perm + [0]
        chunk_data.append(path)
    
    return chunk_data

def process_chunk_optimized(args):
    """Optimized chunk processing."""
    chunk_paths, graph_np = args
    min_distance = 1e10  # inf equivalent
    best_path = None
    
    # Convert to numpy for faster processing
    for path in chunk_paths:
        if is_valid_path_numba(np.array(path), graph_np):
            path_distance = calculate_path_distance_numba(np.array(path), graph_np)
            if path_distance < min_distance:
                min_distance = path_distance
                best_path = path.copy()
    
    return min_distance, best_path

def monitor_cpu_usage():
    """Monitor CPU usage in real-time."""
    return psutil.cpu_percent(interval=0.1, percpu=True)

def brute_force_tsp_i9_optimized(graph, num_threads=None, target_cpu_usage=90):
    """
    Ultra-optimized brute force TSP for Intel i9 processors.
    Target: 90% CPU utilization for 13x13 matrix.
    """
    if num_threads is None:
        # Use all physical cores for i9
        num_threads = multiprocessing.cpu_count()
    
    n = len(graph)
    total_permutations = math.factorial(n - 1)
    
    print(f"🚀 Intel i9 Optimized Brute Force TSP Solver")
    print(f"📊 Matrix size: {n}x{n}")
    print(f"🔧 Total permutations: {total_permutations:,}")
    print(f"⚡ Using {num_threads} threads")
    print(f"🎯 Target CPU usage: {target_cpu_usage}%")
    
    # Convert graph to numpy array for faster processing
    graph_np = np.array(graph, dtype=np.float64)
    graph_np[graph_np == inf] = 1e10  # Replace inf with large number
    
    # Calculate optimal chunk size for i9 cache optimization
    # L3 cache size consideration for i9: typically 16-24MB
    chunk_size = max(5000, total_permutations // (num_threads * 20))
    
    print(f"📦 Chunk size: {chunk_size}")
    
    start_time = time.time()
    cpu_samples = []
    
    # Generate chunks with progress monitoring
    chunks = []
    print("\n🔄 Generating permutation chunks...")
    
    for start in range(0, total_permutations, chunk_size):
        end = min(start + chunk_size, total_permutations)
        chunk = generate_permutations_chunk_optimized(n, start, end)
        chunks.append((chunk, graph_np))
        
        # Sample CPU usage
        if start % (chunk_size * 10) == 0:
            cpu_usage = monitor_cpu_usage()
            avg_cpu = sum(cpu_usage) / len(cpu_usage)
            cpu_samples.append(avg_cpu)
            print(f"   Progress: {start:,}/{total_permutations:,} | CPU: {avg_cpu:.1f}%")
    
    print(f"\n🎯 Processing {len(chunks)} chunks in parallel...")
    
    # Initialize performance tracking
    best_distance = 1e10
    best_path = None
    processed_chunks = 0
    total_chunks = len(chunks)
    
    # Process with optimized thread pool for i9
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_threads,
        mp_context=multiprocessing.get_context('spawn')  # Better for Windows
    ) as executor:
        
        # Submit chunks in batches to prevent memory overflow
        batch_size = min(100, max(10, total_chunks // num_threads))
        
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            
            # Submit batch
            futures = [executor.submit(process_chunk_optimized, chunk) for chunk in batch_chunks]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_distance, chunk_path = future.result(timeout=300)  # 5 min timeout
                    
                    if chunk_distance < best_distance:
                        best_distance = chunk_distance
                        best_path = chunk_path
                        
                        # Real-time progress and CPU monitoring
                        cpu_usage = monitor_cpu_usage()
                        avg_cpu = sum(cpu_usage) / len(cpu_usage)
                        
                        print(f"   ✨ New best: {best_distance} | "
                              f"CPU: {avg_cpu:.1f}% | "
                              f"Progress: {processed_chunks}/{total_chunks}")
                    
                    processed_chunks += 1
                    
                except Exception as e:
                    print(f"⚠️  Error processing chunk: {e}")
                    processed_chunks += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final performance summary
    if cpu_samples:
        avg_cpu_usage = sum(cpu_samples) / len(cpu_samples)
        max_cpu_usage = max(cpu_samples)
        min_cpu_usage = min(cpu_samples)
    else:
        avg_cpu_usage = max_cpu_usage = min_cpu_usage = 0
    
    print(f"\n🏁 COMPLETED!")
    print(f"   Best distance: {best_distance}")
    print(f"   Best path: {best_path}")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Permutations/second: {total_permutations/total_time:,.0f}")
    print(f"   CPU Usage - Avg: {avg_cpu_usage:.1f}%, Max: {max_cpu_usage:.1f}%, Min: {min_cpu_usage:.1f}%")
    
    # Performance optimization suggestions
    if avg_cpu_usage < target_cpu_usage * 0.8:
        print(f"\n💡 Suggestions to improve CPU utilization:")
        print(f"   - Increase chunk size (current: {chunk_size})")
        print(f"   - Reduce batch size (current: {batch_size})")
        print(f"   - Check for I/O bottlenecks")
    elif avg_cpu_usage > target_cpu_usage:
        print(f"\n✅ Excellent! CPU utilization exceeded target.")
    
    return best_distance, best_path

def test_13x13_matrix():
    """Test with a 13x13 matrix as requested."""
    print("\n" + "="*60)
    print("TESTING 13x13 MATRIX - INTEL i9 OPTIMIZED")
    print("="*60)
    
    # Create a random 13x13 matrix
    import random
    random.seed(42)
    
    n = 13
    graph = [[inf if i == j else random.randint(1, 50) for j in range(n)] for i in range(n)]
    
    # Make it symmetric for TSP
    for i in range(n):
        for j in range(i+1, n):
            graph[j][i] = graph[i][j]
    
    print(f"Generated {n}x{n} symmetric matrix")
    print("Matrix preview (first 5x5):")
    for i in range(min(5, n)):
        print(graph[i][:min(5, n)])
    
    # Run the optimized solver
    result = brute_force_tsp_i9_optimized(graph, target_cpu_usage=90)
    
    return result

if __name__ == "__main__":
    # Small test first
    small_graph = [
        [inf, 1, 3, 5, 8],
        [1, inf, 4, 2, 9],
        [3, 4, inf, 7, 2],
        [5, 2, 7, inf, 2],
        [8, 9, 2, 2, inf]
    ]
    
    print("Testing with small 5x5 matrix...")
    result = brute_force_tsp_i9_optimized(small_graph)
    
    # Uncomment to test 13x13 matrix
    # test_13x13_matrix()