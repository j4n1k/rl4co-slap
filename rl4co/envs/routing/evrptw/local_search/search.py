"""
Giant tour representation:
o1 -> 2 -> 3 -> 3 -> 5 -> 6 -> d1 -> o2 -> 8 -> 7 -> 9 -> d2
o = Origins
d = Destinations
1 ... N = Customers / Requests
"""
import networkx as nx
import random

customers = [i for i in range(12)]

depots = [i for i in range(12, 16)]

n_vehicles = len(depots) // 2
depots_start = depots[:n_vehicles]
depots_end = depots[n_vehicles:]


def feasible(route: list, extension: int):
    if route[-1] in depots_start and extension not in customers and extension not in depots_end:
        return False
    elif route[-1] in depots_end and extension not in depots_start:
        return False
    elif route[-1] in depots_end and extension in depots_start:
        return True
    elif route[-1] in depots_start and extension in customers:
        return True
    elif route[-1] in customers and extension in customers:
        return True
    elif route[-1] in customers and extension in depots_end and not set(depots_start).issubset(set(route)) and customers or set(customers).issubset(set(route)):
        return True
    else:
        return False


def gen_solution(customers, depots_start, depots_end):
    s = customers + depots_start + depots_end
    solution = []
    for i in s:
        if i in depots_start:
            solution.append(i)
            idx = s.index(i)
            s.pop(idx)
            break
    while s:
        extension = random.choice(s)
        if feasible(solution, extension):
            solution.append(extension)
            idx = s.index(extension)
            s.pop(idx)
    return solution


gen_solution(customers, depots_start, depots_end)
print()