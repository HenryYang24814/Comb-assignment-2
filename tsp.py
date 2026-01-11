import math
inf = math.inf
from permutation import trotter_johnson_unrank
from graph import MST, DFS

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
    Recursively to find the shortest Hamiltonian cycle.
    bounding belongs to {minout, two_min, mst_bound}
    path = [0] as default start
    """
    size = len(graph)

    #Basis
    if len(path) == size:
        # check the return path to city 0
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

    # use heap here
    targets = sorted(targets)

    for (bound, target) in targets:
        #prunching
        if bound < shortest:
            current_shortest, tour = backtracking(graph, path + [target], shortest, best_path, bounding)

            if tour and current_shortest < shortest:
                shortest = current_shortest

                # Update best_path for this scope
                best_path = tour

    return shortest, best_path


def tsp_approximation(graph, n):
    """
    According to the class perseudo code
    """
    # 1. Compute MST
    mst_matrix = MST(graph, n)

    # 2. Get the DFS order
    tsp_order = DFS(mst_matrix, [0])

    # 3. Add the edge connected last and first node to build the hamilton cycle and get its total dist
    tsp_cycle = tsp_order + [tsp_order[0]]
    total_dist = distance(tsp_cycle, graph)

    return total_dist, tsp_cycle

def pathdistance(path, graph):
    """The distance of a partial {path} in {graph}"""
    r = 0
    for i in range(len(path) - 1):
        weight = graph[path[i]][path[i + 1]]
        if weight == inf:
            return inf
        else:
            r = r + weight
    return r


def mst_cost(U, graph):
    """
    The cost of MST of the unvisited vertices {U}
    if U disconnected, return inf
    """
    U = list(U)

    # if all vertices visited
    if len(U) <= 1:
        return 0

    # initialization
    tree = set([U[0]])
    best = {}
    for v in U[1:]:
        best[v] = graph[U[0]][v]
    tot = 0

    while len(tree) < len(U):
        # loop until picking all the unvisited vertices with the minimum cost
        v = None
        w = inf
        for x in U:
            if x not in tree:
                wx = best.get(x, inf)
                if wx < w:
                    w = wx
                    v = x

        # if U is disconnected, return inf
        if v is None or w == inf:
            return inf

        tree.add(v)
        tot = tot + w

        # renew the edges from v
        for x in U:
            if x not in tree:
                wx = graph[v][x]
                if wx < best.get(x, inf):
                    best[x] = wx

    return tot


def mst_bound(path, graph):
    """
    Return the MST lower bound for partial {path} in {graph}
        by the formula:
            Lower_bound = cost(visited_vertex) + MST(U) + min(last->U) + min(U->start)
            ( U - set of unvisited vertices )
    """
    size = len(graph)
    start = path[0]
    last = path[-1]

    cost_visited = pathdistance(path, graph)  # cost(visited_vertex)

    U = set(range(size)) - set(path)

    if not U:
        # all visited: must close the cycle
        if graph[last][start] == inf:
            return inf
        else:
            return cost_visited + graph[last][start]

    last_U = min([graph[last][u] for u in U])  # min(last->U)
    U_start = min([graph[u][start] for u in U])  # min(U->start)
    MST_U = mst_cost(U, graph)  # MST(U)


    if last_U == inf or U_start == inf or cost_visited == inf or MST_U == inf:
        return inf

    return cost_visited + MST_U + last_U + U_start
