import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
import trimesh
import vtk
import sys

# Ensure vtk logging verbosity is off
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)

def triangles_intersect(triangle1, vertices, faces):
    face = np.array([3, 0, 1, 2])
    labelmap = {x: idx for idx, x in enumerate(np.unique(faces))}
    @np.vectorize
    def relabel(x):
        y = 0
        if x in labelmap:
            y = labelmap[x]
        return y
    faces = relabel(faces)
    new_column = np.full((faces.shape[0], 1), 3)
    faces = np.hstack((new_column, faces))

    surface1 = pv.PolyData(triangle1, face)
    surface2 = pv.PolyData(vertices, faces.flatten())

    return surface1.collision(surface2)[1] > 0

def mesh2triangles(mesh):
    return mesh.vertices[mesh.faces]

def mesh2tricenters(mesh, triangles=None):
    if triangles is None:
        triangles = mesh2triangles(mesh)
    centers = np.mean(triangles, axis=1)
    return centers

def count_collisions(mesh1, mesh2, k=5):
    triangles2 = mesh2triangles(mesh2)
    centers2 = mesh2tricenters(mesh2, triangles=triangles2)
    triangles1 = mesh2triangles(mesh1)
    centers1 = mesh2tricenters(mesh1, triangles=triangles1)

    tree = cKDTree(centers2)

    collision_count = 0
    for idx, triangle in enumerate(centers1):
        dists, indices = tree.query(triangle.reshape(1, -1), k=k)
        collision = triangles_intersect(triangles1[idx, :, :],
                                        mesh2.vertices[np.sort(np.unique(mesh2.faces[indices[0]].flatten()))],
                                        mesh2.faces[indices[0]])
        if collision:
            collision_count += 1

    return collision_count

def detachedtriangles(mesh, triangle_id, other_ids):
    mask = np.any(np.isin(mesh.faces[other_ids], mesh.faces[triangle_id]), axis=1)
    faces = mesh.faces[other_ids][~mask]
    return faces

def count_self_collisions(mesh, k=5):
    faces = mesh.faces
    triangles = mesh2triangles(mesh)
    centers = mesh2tricenters(mesh, triangles=triangles)
    tree = cKDTree(centers)

    collision_count = 0
    for idx, triangle in enumerate(centers):
        dists, indices = tree.query(triangle.reshape(1, -1), k=k)
        faces = detachedtriangles(mesh, idx, indices[0][1:])
        if faces.size == 0:
            print('k is too small')
            continue
        collision = triangles_intersect(triangles[idx, :, :],
                                        mesh.vertices[np.sort(np.unique(faces.flatten()))],
                                        faces)
        if collision:
            collision_count += 1

    return collision_count

# Dummy data for testing
mesh1_vertices = np.random.rand(100, 3)  # Random vertices for mesh1
mesh2_vertices = np.random.rand(100, 3)  # Random vertices for mesh2
faces1 = np.array([np.random.choice(100, 3, replace=False) for _ in range(50)])  # Random faces for mesh1
faces2 = np.array([np.random.choice(100, 3, replace=False) for _ in range(50)])  # Random faces for mesh2

# Create Trimesh objects
mesh1 = trimesh.Trimesh(vertices=mesh1_vertices, faces=faces1)
mesh2 = trimesh.Trimesh(vertices=mesh2_vertices, faces=faces2)

# Check self-intersections in mesh1
self_collision_count = count_self_collisions(mesh1, k=50)
print(f'Number of self-intersections in mesh1: {self_collision_count}')

# Check collisions between mesh1 and mesh2
collision_count = count_collisions(mesh1, mesh2, k=50)
print(f'Number of collisions between mesh1 and mesh2: {collision_count}')
