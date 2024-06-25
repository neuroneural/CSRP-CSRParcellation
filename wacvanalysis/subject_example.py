import os
import nibabel as nib
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

# Function to load FreeSurfer surface files
def load_freesurfer_surface(filepath):
    coords, faces = nib.freesurfer.read_geometry(filepath)
    return trimesh.Trimesh(vertices=coords, faces=faces)

# Paths to the required data
freesurfer_subject_path = '/data/users2/washbee/csrf-v2cc-data-conference/test/201818'
fastsurfer_subject_path = '/data/users2/washbee/fastsurfer-output/test/201818/201818'

# Load FreeSurfer pial surfaces
fs_lh_pial_path = os.path.join(freesurfer_subject_path, 'surf', 'lh.pial')
fs_rh_pial_path = os.path.join(freesurfer_subject_path, 'surf', 'rh.pial')

# Load FastSurfer pial surfaces
fast_lh_pial_path = os.path.join(fastsurfer_subject_path, 'surf', 'lh.pial')
fast_rh_pial_path = os.path.join(fastsurfer_subject_path, 'surf', 'rh.pial')

# Load the meshes using nibabel
fs_lh_pial = load_freesurfer_surface(fs_lh_pial_path)
fs_rh_pial = load_freesurfer_surface(fs_rh_pial_path)
fast_lh_pial = load_freesurfer_surface(fast_lh_pial_path)
fast_rh_pial = load_freesurfer_surface(fast_rh_pial_path)

# Compute Chamfer and Hausdorff distances for left hemisphere pial surfaces
lh_chamfer_dist = compute_chamfer_distance(fs_lh_pial, fast_lh_pial)
lh_hausdorff_dist = compute_hausdorff_distance(fs_lh_pial, fast_lh_pial)

# Compute Chamfer and Hausdorff distances for right hemisphere pial surfaces
rh_chamfer_dist = compute_chamfer_distance(fs_rh_pial, fast_rh_pial)
rh_hausdorff_dist = compute_hausdorff_distance(fs_rh_pial, fast_rh_pial)

# Compute self-intersections for left hemisphere pial surface
lh_self_collision_count = count_self_collisions(fast_lh_pial, k=5)

# Compute self-intersections for right hemisphere pial surface
rh_self_collision_count = count_self_collisions(fast_rh_pial, k=5)

# Load annotation files
fs_lh_annot_path = os.path.join(freesurfer_subject_path, 'label', 'lh.aparc.DKTatlas40.annot')
fs_rh_annot_path = os.path.join(freesurfer_subject_path, 'label', 'rh.aparc.DKTatlas40.annot')
fast_lh_annot_path = os.path.join(fastsurfer_subject_path, 'label', 'lh.aparc.DKTatlas.mapped.annot')
fast_rh_annot_path = os.path.join(fastsurfer_subject_path, 'label', 'rh.aparc.DKTatlas.mapped.annot')

# Load the annotations
fs_lh_labels = nib.freesurfer.read_annot(fs_lh_annot_path)[0]
fs_rh_labels = nib.freesurfer.read_annot(fs_rh_annot_path)[0]
fast_lh_labels = nib.freesurfer.read_annot(fast_lh_annot_path)[0]
fast_rh_labels = nib.freesurfer.read_annot(fast_rh_annot_path)[0]

# Compute Dice scores
lh_dice_scores = calculate_dice_score(fs_lh_labels, fast_lh_labels)
rh_dice_scores = calculate_dice_score(fs_rh_labels, fast_rh_labels)

# Print results
print(f'Left Hemisphere Chamfer Distance: {lh_chamfer_dist}')
print(f'Left Hemisphere Hausdorff Distance: {lh_hausdorff_dist}')
print(f'Right Hemisphere Chamfer Distance: {rh_chamfer_dist}')
print(f'Right Hemisphere Hausdorff Distance: {rh_hausdorff_dist}')
print(f'Left Hemisphere Self-Intersections (SIF): {lh_self_collision_count}')
print(f'Right Hemisphere Self-Intersections (SIF): {rh_self_collision_count}')
print(f'Left Hemisphere Dice Scores: {lh_dice_scores}')
print(f'Right Hemisphere Dice Scores: {rh_dice_scores}')
