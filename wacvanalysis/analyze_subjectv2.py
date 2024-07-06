import os
import csv
import nibabel as nib
import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial import cKDTree
import vtk
from multiprocessing import Lock
import sys 
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
def load_freesurfer_surface_and_labels(surface_path, label_path, output_dir, hemi, surf_type,framework,subj_id):
    coords, faces = nib.freesurfer.read_geometry(surface_path)
    labels, ctab, names = nib.freesurfer.read_annot(label_path)
    
    assert len(labels) == coords.shape[0], f"Label and vertex count mismatch: labels {len(labels)}, vertices {coords.shape[0]}"
    
    # Save the surface and annotations to the output directory
    surf_output_path = os.path.join(output_dir, f'{framework}.{subj_id}.{hemi}.{surf_type}')
    annot_output_path = os.path.join(output_dir, f'{framework}.{subj_id}.{hemi}.{surf_type}.annot')
    
    nib.freesurfer.write_geometry(surf_output_path, coords, faces)
    nib.freesurfer.write_annot(annot_output_path, labels, ctab, names)
    
    mesh = trimesh.Trimesh(vertices=coords, faces=faces)
    assert mesh.vertices.shape == coords.shape
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
    
    mapped_labels2 = labels1[indices]

    return calculate_dice_score(labels2, mapped_labels2)

# Function to write results to CSV
def write_to_csv(file_path, lock, data):
    with lock:
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Framework', 'Subject ID', 'Hemisphere', 'Metric', 'Label', 'Score', 'Total Triangles'])
            writer.writerow(data)

# Process a subject and compute metrics
def process_subject(subj_id, csv_file_path, lock, framework_name, gt_subject_base_path, fastsurfer_subject_base_path, output_dir):
    # Paths to the required data
    freesurfer_subject_path = os.path.join(gt_subject_base_path, subj_id)
    fastsurfer_subject_path = os.path.join(fastsurfer_subject_base_path, subj_id, subj_id)

    hemispheres = ['lh', 'rh']
    
    for hemi in hemispheres:
        try:
            fs_pial, fs_labels = load_freesurfer_surface_and_labels(
                os.path.join(freesurfer_subject_path, 'surf', f'{hemi}.pial'),
                os.path.join(freesurfer_subject_path, 'label', f'{hemi}.aparc.DKTatlas40.annot'),
                output_dir, hemi, 'pial','freesurfer',subj_id
            )
        except ValueError as e:
            print(f"Error loading FreeSurfer {hemi} hemisphere: {e}")
            continue

        try:
            fast_pial, fast_labels = load_freesurfer_surface_and_labels(
                os.path.join(fastsurfer_subject_path, 'surf', f'{hemi}.pial'),
                os.path.join(fastsurfer_subject_path, 'label', f'{hemi}.aparc.DKTatlas.mapped.annot'),
                output_dir, hemi, 'pial','fastsurfer',subj_id
            )
        except ValueError as e:
            print(f"Error loading FastSurfer {hemi} hemisphere: {e}")
            continue

        # Compute Chamfer and Hausdorff distances for pial surfaces
        chamfer_dist = compute_chamfer_distance(fs_pial, fast_pial)
        hausdorff_dist = compute_hausdorff_distance(fs_pial, fast_pial)
        
        
        
        # Compute Dice scores with mapping
        try:
            dice_scores = calculate_dice_score_with_mapping(fs_labels, fast_labels, fs_pial.vertices, fast_pial.vertices)
        except ValueError as e:
            print(f"Error calculating left hemisphere Dice scores: {e}")
            return
        
        # Calculate macro Dice score excluding labels -1 and 4
        filtered_scores = {label: score for label, score in dice_scores.items() if label not in [-1, 4]}
        if filtered_scores:
            macro_dice_score = np.mean(list(filtered_scores.values()))
        else:
            macro_dice_score = 0.0

        # Compute self-intersections 
        total_triangles = len(fast_pial.faces)
        assert total_triangles > 10000, f"{total_triangles}"
        
        self_collision_count = count_self_collisions(fast_pial, k=30)
        # Write results to CSV
        for label, score in dice_scores.items():
            write_to_csv(csv_file_path, lock, [framework_name, subj_id, hemi, 'Dice', label, score, ''])

        write_to_csv(csv_file_path, lock, [framework_name, subj_id, hemi, 'Macro Dice', '', macro_dice_score, ''])
        write_to_csv(csv_file_path, lock, [framework_name, subj_id, hemi, 'Chamfer Distance', '', chamfer_dist, ''])
        write_to_csv(csv_file_path, lock, [framework_name, subj_id, hemi, 'Hausdorff Distance', '', hausdorff_dist, ''])
        write_to_csv(csv_file_path, lock, [framework_name, subj_id, hemi, 'Self-Intersections (SIF)', '', self_collision_count, total_triangles])

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python script.py <subj_id> <csv_file_path> <framework_name> <gt_subject_base_path> <fs_subject_base_path>")
        sys.exit(1)
    
    subj_id = sys.argv[1]
    csv_file_path = sys.argv[2]
    framework_name = sys.argv[3]
    gt_subject_base_path = sys.argv[4]
    fs_subject_base_path = sys.argv[5]
    output_dir = 'result/'

    print('subj_id','csv_file_path','framework_name','gt_subject_base_path','fs_subject_base_path',"output_dir")
    print(subj_id,csv_file_path,framework_name,gt_subject_base_path,fs_subject_base_path,output_dir)
    # Initialize a lock for parallel-safe writing
    lock = Lock()
    
    process_subject(subj_id, csv_file_path, lock, framework_name, gt_subject_base_path, fs_subject_base_path,output_dir)


# if __name__ == "__main__":
#     framework_name = 'fastsurfer'
#     subj_id = "201818"
#     csv_file_path = 'testing.csv'
#     gt_subject_base_path = '/data/users2/washbee/speedrun/cortexode-data-rp/test/'
#     fast_subject_base_path = '/data/users2/washbee/fastsurfer-output/test/'
#     output_dir = './archive/'

#     # Initialize a lock for parallel-safe writing
#     lock = Lock()
#     process_subject(subj_id, csv_file_path, lock, framework_name, gt_subject_base_path, fast_subject_base_path, output_dir)
