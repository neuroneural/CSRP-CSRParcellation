import os
import nibabel as nib
import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
import vtk

# Turn off VTK logging verbosity
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)

# Function to compute Chamfer Distance using KD-Tree
def compute_chamfer_distance(mesh1, mesh2):
    points1 = mesh1.vertices
    points2 = mesh2.vertices
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    dist1, _ = tree1.query(points2, k=1)
    dist2, _ = tree2.query(points1, k=1)
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    return chamfer_dist

# Function to compute Hausdorff Distance using KD-Tree
def compute_hausdorff_distance(mesh1, mesh2):
    points1 = mesh1.vertices
    points2 = mesh2.vertices
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    dist1, _ = tree1.query(points2, k=1)
    dist2, _ = tree2.query(points1, k=1)
    hausdorff_dist = max(np.max(dist1), np.max(dist2))
    return hausdorff_dist

# Sergey's functions for self-intersection detection
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

# Function to calculate Dice Score
def calculate_dice_score(labels1, labels2):
    dice_scores = {}
    unique_labels = np.unique(labels1)
    
    for label in unique_labels:
        intersection = np.sum((labels1 == label) & (labels2 == label))
        size1 = np.sum(labels1 == label)
        size2 = np.sum(labels2 == label)
        dice_score = 2 * intersection / (size1 + size2) if (size1 + size2) > 0 else 0.0
        dice_scores[label] = dice_score
    
    return dice_scores

# Adjusted function to ensure labels match vertex counts
def load_freesurfer_surface_and_labels(surface_path, label_path):
    coords, faces = nib.freesurfer.read_geometry(surface_path)
    labels, _, _ = nib.freesurfer.read_annot(label_path)
    assert len(labels) == coords.shape[0], f"Label and vertex count mismatch: labels {len(labels)}, vertices {coords.shape[0]}"
    mesh = trimesh.Trimesh(vertices=coords, faces=faces)
    assert mesh.vertices.shape == coords.shape,f"{mesh.vertices.shape} - {coords.shape}\n{surface_path}\n{label_path}"
    return mesh, labels

# Function to calculate Dice Score for Parcellation using KD-Tree mapping
def calculate_dice_score_with_mapping(labels1, labels2, vertices1, vertices2):
    assert len(labels1) == vertices1.shape[0], f"Mismatch between labels1 and vertices1: labels1 {len(labels1)}, vertices1 {vertices1.shape[0]}"
    assert len(labels2) == vertices2.shape[0], f"Mismatch between labels2 and vertices2: labels2 {len(labels2)}, vertices2 {vertices2.shape[0]}"
    
    if len(labels1) > len(labels2):
        labels1, labels2 = labels2, labels1
        vertices1, vertices2 = vertices2, vertices1
    
    tree = cKDTree(vertices1)
    distances, indices = tree.query(vertices2, k=1)
    
    assert len(indices) == vertices2.shape[0]
    
    # Debugging the mapping process
    print(f'Min distance: {np.min(distances)}, Max distance: {np.max(distances)}, Mean distance: {np.mean(distances)}')

    # Check if indices are unique when vertices1 == vertices2
    if np.array_equal(vertices1, vertices2):
        unique_indices = np.unique(indices)
        if len(unique_indices) != len(indices):
            print(f"Warning: Non-unique indices found when vertices1 == vertices2. Unique indices: {len(unique_indices)}, Total indices: {len(indices)}")
        else:
            print("All indices are unique when vertices1 == vertices2.")
    
    mapped_labels2 = np.zeros_like(labels2)
    for index, assignmentIndex in enumerate(indices):
        mapped_labels2[index] = labels1[assignmentIndex]
    
    assert labels2.shape == mapped_labels2.shape
    assert np.array_equal(vertices1, vertices2) == np.array_equal(labels2, mapped_labels2)
    return calculate_dice_score(labels2, mapped_labels2)

# Paths to the required data
freesurfer_subject_path = '/data/users2/washbee/speedrun/cortexode-data-rp/test/201818'
fastsurfer_subject_path = '/data/users2/washbee/fastsurfer-output/test/201818/201818'

# Load FreeSurfer pial surfaces and labels
try:
    fs_lh_pial, fs_lh_labels = load_freesurfer_surface_and_labels(
        os.path.join(freesurfer_subject_path, 'surf', 'lh.pial'),
        os.path.join(freesurfer_subject_path, 'label', 'lh.aparc.DKTatlas40.annot')
    )
except ValueError as e:
    print(f"Error loading FreeSurfer left hemisphere: {e}")

try:
    fs_rh_pial, fs_rh_labels = load_freesurfer_surface_and_labels(
        os.path.join(freesurfer_subject_path, 'surf', 'rh.pial'),
        os.path.join(freesurfer_subject_path, 'label', 'rh.aparc.DKTatlas40.annot')
    )
except ValueError as e:
    print(f"Error loading FreeSurfer right hemisphere: {e}")

# Load FastSurfer pial surfaces and labels
try:
    fast_lh_pial, fast_lh_labels = load_freesurfer_surface_and_labels(
        os.path.join(fastsurfer_subject_path, 'surf', 'lh.pial'),
        os.path.join(fastsurfer_subject_path, 'label', 'lh.aparc.DKTatlas.mapped.annot')
    )
except ValueError as e:
    print(f"Error loading FastSurfer left hemisphere: {e}")

try:
    fast_rh_pial, fast_rh_labels = load_freesurfer_surface_and_labels(
        os.path.join(fastsurfer_subject_path, 'surf', 'rh.pial'),
        os.path.join(fastsurfer_subject_path, 'label', 'rh.aparc.DKTatlas.mapped.annot')
    )
except ValueError as e:
    print(f"Error loading FastSurfer right hemisphere: {e}")

# Debugging the mapping process
print(f'FreeSurfer LH Labels Shape: {fs_lh_labels.shape}')
print(f'FastSurfer LH Labels Shape: {fast_lh_labels.shape}')

# Compute Dice scores with mapping
try:
    lh_dice_scores = calculate_dice_score_with_mapping(fs_lh_labels, fast_lh_labels, fs_lh_pial.vertices, fast_lh_pial.vertices)
except ValueError as e:
    print(f"Error calculating left hemisphere Dice scores: {e}")

try:
    rh_dice_scores = calculate_dice_score_with_mapping(fs_rh_labels, fast_rh_labels, fs_rh_pial.vertices, fast_rh_pial.vertices)
except ValueError as e:
    print(f"Error calculating right hemisphere Dice scores: {e}")

# KD-Tree Self-Comparison test
try:
    lh_self_dice_scores_kdtree = calculate_dice_score_with_mapping(fs_lh_labels, fs_lh_labels, fs_lh_pial.vertices, fs_lh_pial.vertices)
except ValueError as e:
    print(f"Error calculating left hemisphere KD-Tree self-comparison Dice scores: {e}")

try:
    rh_self_dice_scores_kdtree = calculate_dice_score_with_mapping(fs_rh_labels, fs_rh_labels, fs_rh_pial.vertices, fs_rh_pial.vertices)
except ValueError as e:
    print(f"Error calculating right hemisphere KD-Tree self-comparison Dice scores: {e}")

# Print Dice scores in a tabular format
print("Left Hemisphere Dice Scores:")
print("{:<10} {:<10}".format('Label', 'Dice Score'))
for label, score in lh_dice_scores.items():
    print("{:<10} {:<10.4f}".format(label, score))

print("\nRight Hemisphere Dice Scores:")
print("{:<10} {:<10}".format('Label', 'Dice Score'))
for label, score in rh_dice_scores.items():
    print("{:<10} {:<10.4f}".format(label, score))

# Print self-comparison Dice scores in a tabular format using KD-Tree
print("\nLeft Hemisphere KD-Tree Self-Comparison Dice Scores:")
print("{:<10} {:<10}".format('Label', 'Dice Score'))
for label, score in lh_self_dice_scores_kdtree.items():
    print("{:<10} {:<10.4f}".format(label, score))

print("\nRight Hemisphere KD-Tree Self-Comparison Dice Scores:")
print("{:<10} {:<10}".format('Label', 'Dice Score'))
for label, score in rh_self_dice_scores_kdtree.items():
    print("{:<10} {:<10.4f}".format(label, score))

# Compute Chamfer and Hausdorff distances for left hemisphere pial surfaces
lh_chamfer_dist = compute_chamfer_distance(fs_lh_pial, fast_lh_pial)
lh_hausdorff_dist = compute_hausdorff_distance(fs_lh_pial, fast_lh_pial)
print(f'\nLeft Hemisphere Chamfer Distance: {lh_chamfer_dist:.4f}')
print(f'Left Hemisphere Hausdorff Distance: {lh_hausdorff_dist:.4f}')

# Compute Chamfer and Hausdorff distances for right hemisphere pial surfaces
rh_chamfer_dist = compute_chamfer_distance(fs_rh_pial, fast_rh_pial)
rh_hausdorff_dist = compute_hausdorff_distance(fs_rh_pial, fast_rh_pial)
print(f'Right Hemisphere Chamfer Distance: {rh_chamfer_dist:.4f}')
print(f'Right Hemisphere Hausdorff Distance: {rh_hausdorff_dist:.4f}')

# Compute self-intersections for left hemisphere pial surface
lh_self_collision_count = count_self_collisions(fast_lh_pial, k=30)
print(f'\nLeft Hemisphere Self-Intersections (SIF): {lh_self_collision_count}')

# Compute self-intersections for right hemisphere pial surface
rh_self_collision_count = count_self_collisions(fast_rh_pial, k=30)
print(f'Right Hemisphere Self-Intersections (SIF): {rh_self_collision_count}')
