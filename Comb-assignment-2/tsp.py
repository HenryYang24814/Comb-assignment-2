#回溯放这里，第二题的三个bound算法放这里，第一题的bfs放这里
import math
inf = math.inf 


def distance(path, graph):
    """Return the total distance of the given path."""
    dist_val = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        dist_val = dist_val + graph[u][v]
    return dist_val

def iscycle(path, graph):
    size = len(graph)
    if len(path) == size:
        # Check if we can return from the last city to the start (city 0)
        if graph[path[-1]][path[0]] != inf:
            return True
    return False

def minout(path, graph):
    size = len(graph)
    r = distance(path, graph)
    unvisited = set(range(size)) - set(path[:-1])
    for target in unvisited:
        r = r + min(graph[target])
    return r

def two_min(path):
    sorted_path = sorted(path)
    r = sorted_path[0] + sorted_path[1]
    return r


def two_minout(path,graph):
    size = len(graph)
    r = distance(path, graph)
    total = 0
    unvisited = set(range(size)) - set(path[:-1])
    for target in unvisited:
        total = total + two_min(graph[target])

    r = r + total/2

    return r


def backtracking(graph, path, shortest, best_path, bounding=minout):
    """
    Recursive function to find the shortest Hamiltonian cycle.
    """
    size = len(graph)
    
    # Initialize result with a full cycle if possible
    result = path + [path[0]] if iscycle(path, graph) else []
    
    # Calculate candidates (targets)
    targets = []
    for target in (set(range(size)) - set(path)):
        if (graph[path[-1]][target]) != inf:
            targets.append(target)
            
    for target in targets:
        # Pruning based on the lower bound
        if bounding(path + [target], graph) < shortest:
            # Recurse: Note we pass shortest and best_path forward
            current_shortest, tour = backtracking(graph, path + [target], shortest, best_path, bounding)
            
            if tour:
                cost = distance(tour, graph)
                if cost < shortest:
                    shortest = cost
                    result = tour
                    
    # Return both the cost and the path to match your test.py expectation
    return shortest, result