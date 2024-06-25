import numpy as np
from scipy.spatial import cKDTree

# Create a set of vertices with some identical points
vertices = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0],
    [1.0, 1.0, 1.0],  # Identical to the second vertex
    [4.0, 4.0, 4.0]
])

# Create a KD-Tree from these vertices
tree = cKDTree(vertices)

# Query the KD-Tree for nearest neighbors
distances, indices = tree.query(vertices, k=1)

# Print the results
print("Distances:")
print(distances)
print("Indices:")
print(indices)

# Check if identical vertices are considered nearest neighbors
identical_pairs = [(i, indices[i]) for i in range(len(vertices)) if i != indices[i] and np.all(vertices[i] == vertices[indices[i]])]

print("\nIdentical Pairs:")
print(identical_pairs)
