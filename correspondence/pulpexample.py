import numpy as np
from scipy.spatial import cKDTree
from pulp import *

# Generate random point clouds for mesh A and mesh B
num_points = 150000
num_dimensions = 3
points_a = np.random.rand(num_points, num_dimensions)
points_b = np.random.rand(num_points, num_dimensions)

# Compute distances between point clouds using cKDTree
tree_a = cKDTree(points_a)
distances, indices = tree_a.query(points_b, k=2)

# Formulate MILP using PuLP
def solve_milp(distances):
    num_points_a, num_points_b = distances.shape

    # Create a binary variable for each pair of points
    variables = {(i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary) for i in range(num_points_a) for j in range(num_points_b)}

    # Create the optimization problem
    prob = LpProblem("Minimize Distance", LpMinimize)

    # Objective: minimize total distance
    prob += lpSum(distances[i, j] * variables[(i, j)] for i in range(num_points_a) for j in range(num_points_b))

    # Constraints: each vertex in mesh A is matched with exactly one vertex in mesh B
    for i in range(num_points_a):
        prob += lpSum(variables[(i, j)] for j in range(num_points_b)) == 1

    # Constraint: the sum of pairs equals the number of vertices in mesh A
    prob += lpSum(variables[(i, j)] for i in range(num_points_a) for j in range(num_points_b)) == num_points_a

    # Solve the problem
    prob.solve()

    # Extract solution
    solution = {(i, j): value(variables[(i, j)]) for i in range(num_points_a) for j in range(num_points_b)}

    return solution

# Solve MILP
solution = solve_milp(distances)

# Process solution: Get matching pairs
matching_pairs = [(i, j) for (i, j), value in solution.items() if value == 1]
#print("Matching pairs:", matching_pairs)
print("len Matching pairs:", len(matching_pairs))
