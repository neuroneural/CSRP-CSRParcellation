import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import jaccard_score
from scipy.spatial import cKDTree
import vtk

# Turn off VTK logging verbosity
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)

# Function to compute Chamfer Distance
def compute_chamfer_distance(mesh1, mesh2):
    points1 = mesh1.vertices
    points2 = mesh2.vertices
    dist1 = np.mean(np.min(np.linalg.norm(points1[:, None] - points2[None, :], axis=-1), axis=-1))
    dist2 = np.mean(np.min(np.linalg.norm(points2[:, None] - points1[None, :], axis=-1), axis=-1))
    chamfer_dist = dist1 + dist2
    return chamfer_dist

# Function to compute Hausdorff Distance
def compute_hausdorff_distance(mesh1, mesh2):
    points1 = mesh1.vertices
    points2 = mesh2.vertices
    d1, _, _ = directed_hausdorff(points1, points2)
    d2, _, _ = directed_hausdorff(points2, points1)
    hausdorff_dist = max(d1, d2)
    return hausdorff_dist

# Function to compute Self-Intersections using PyVista (Sergey's function)
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

def detachedtriangles(mesh, triangle_id, other_ids):
    mask = np.any(np.isin(mesh.faces[other_ids], mesh.faces[triangle_id]), axis=1)
    faces = mesh.faces[other_ids][~mask]
    return faces

# Function to calculate Dice Score for Parcellation
def calculate_dice_score(labels1, labels2):
    dice_scores = {}
    unique_labels = np.unique(labels1)
    
    for label in unique_labels:
        intersection = np.sum((labels1 == label) & (labels2 == label))
        size1 = np.sum(labels1 == label)
        size2 = np.sum(labels2 == label)
        dice_score = 2 * intersection / (size1 + size2)
        dice_scores[label] = dice_score
    
    return dice_scores

# Dummy data for testing
mesh1_vertices = np.random.rand(100, 3)  # Random vertices for mesh1
mesh2_vertices = np.random.rand(100, 3)  # Random vertices for mesh2
faces = np.array([np.random.choice(100, 3, replace=False) for _ in range(50)])  # Random faces

# Create Trimesh objects
mesh1 = trimesh.Trimesh(vertices=mesh1_vertices, faces=faces)
mesh2 = trimesh.Trimesh(vertices=mesh2_vertices, faces=faces)

# Testing the functions on dummy data
chamfer_dist = compute_chamfer_distance(mesh1, mesh2)
hausdorff_dist = compute_hausdorff_distance(mesh1, mesh2)
self_collision_count = count_self_collisions(mesh1, k=5)

# Dummy data for Dice score
labels1 = np.random.randint(0, 5, 100)  # Random labels
labels2 = np.random.randint(0, 5, 100)  # Random labels
dice_scores = calculate_dice_score(labels1, labels2)

print(f'Chamfer Distance: {chamfer_dist}')
print(f'Hausdorff Distance: {hausdorff_dist}')
print(f'Self-Intersections (SIF): {self_collision_count}')
print(f'Dice Scores: {dice_scores}')
