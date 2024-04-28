import numpy as np
from scipy.spatial import cKDTree
from ortools.linear_solver import pywraplp

# Generate random point clouds for mesh A and mesh B
num_points = 150
num_dimensions = 3
points_a = np.random.rand(num_points, num_dimensions)
points_b = np.random.rand(num_points, num_dimensions)

# Compute distances between point clouds using cKDTree
tree_a = cKDTree(points_a)
distances, indices = tree_a.query(points_b, k=2)

# Create a linear solver instance
solver = pywraplp.Solver.CreateSolver('SCIP')

# Create binary variables for matching pairs
x = {}
for i in range(num_points):
    for j in range(num_points):
        x[i, j] = solver.BoolVar(f'x_{i}_{j}')

# Add constraints: each vertex in mesh A is matched with exactly one vertex in mesh B
for i in range(num_points):
    solver.Add(sum(x[i, j] for j in range(num_points)) == 1)

# Add constraint: the sum of pairs equals the number of vertices in mesh A
solver.Add(sum(x[i, j] for i in range(num_points) for j in range(num_points)) == num_points)

# Set objective: minimize total distance
objective_expr = solver.Sum(distances[i, indices[i, 0]] * x[i, indices[i, 0]] for i in range(num_points))
solver.Minimize(objective_expr)

# Solve the problem
status = solver.Solve()

# Process solution: Get matching pairs
matching_pairs = [(i, indices[i, 0]) for i in range(num_points) if x[i, indices[i, 0]].solution_value() == 1]
print("len Matching pairs:", len(matching_pairs))
