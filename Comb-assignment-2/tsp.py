

import math
inf = math.inf
from permutation import trotter_johnson_unrank


def distance(path, graph):
    """Return the total distance of the given path."""
    dist_val = 0
    for i in range(len(path) - 1):
        j = path[i]
        k = path[i+1]
        dist_val = dist_val + graph[j][k]
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

def factorial(n):
    """Return the factorial of n."""
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)



def brute_force_tsp(graph):
    size = len(graph) - 1
    min_weight = inf
    best_path = []

    for i in range(factorial(size)):
        temp_list = trotter_johnson_unrank(size, i)#trotter_johnson_unrank(size, rank)
        #path = [0] + temp_list + [0]
        path = [0] + temp_list
        if iscycle(path, graph):
            path = path + [0]
            weight = distance(path, graph)
            if weight < min_weight:
                min_weight = weight
                best_path = path

    return min_weight, best_path






def backtracking(graph, path=[0], shortest=inf, best_path=[], bounding=minout):
    """
    Recursive function to find the shortest Hamiltonian cycle.
    bounding belongs to {minout, two_min, mst_bound}
    path = [0] by default
    """
    size = len(graph)

    #Basis
    if len(path) == size:
        # Use your existing iscycle function to check the return path to city 0
        if iscycle(path, graph):
            # Construct the complete cycle path
            full_path = path + [path[0]]
            # Calculate the total weight of this cycle
            cost = distance(full_path, graph)
            # Update the global shortest distance and best tour found so far
            if cost < shortest:
                return cost, full_path
        return shortest, best_path
    # Calculate candidates (targets) with their bounds
    targets = []
    unvisited = set(range(size)) - set(path)
    for target in unvisited:
        if graph[path[-1]][target] != inf:
            # Calculate bound for the potential next step
            bound = bounding(path + [target], graph)
            targets += [(bound, target)]

    # Can use heap here
    targets = sorted(targets)

    for (bound, target) in targets:
        #prunching
        if bound < shortest:
            current_shortest, tour = backtracking(graph, path + [target], shortest, best_path, bounding)

            if tour and current_shortest < shortest:
                shortest = current_shortest

                # Update best_path for this scope
                best_path = tour

    # Return both the cost and the path to match your test.py expectation
    return shortest, best_path