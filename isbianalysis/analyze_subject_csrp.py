import os
import csv
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
import argparse
import re
import pyvista as pv
from scipy.spatial import cKDTree
import sys
from multiprocessing import Lock

# Function to compute Chamfer Distance using KD-Tree
def compute_chamfer_distance(mesh1, mesh2):
    try:
        points1 = mesh1['vertices']
        points2 = mesh2['vertices']
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        dist1, _ = tree1.query(points2, k=1)
        dist2, _ = tree2.query(points1, k=1)
        chamfer_dist = np.mean(dist1) + np.mean(dist2)
        return chamfer_dist
    except Exception as e:
        print(f"Error computing Chamfer Distance: {e}")
        return np.nan

# Function to compute Hausdorff Distance using KD-Tree
def compute_hausdorff_distance(mesh1, mesh2):
    try:
        points1 = mesh1['vertices']
        points2 = mesh2['vertices']
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        dist1, _ = tree1.query(points2, k=1)
        dist2, _ = tree2.query(points1, k=1)
        hausdorff_dist = max(np.max(dist1), np.max(dist2))
        return hausdorff_dist
    except Exception as e:
        print(f"Error computing Hausdorff Distance: {e}")
        return np.nan

# Function to load Freesurfer surface files
def load_freesurfer_surface(surface_path):
    try:
        coords, faces = nib.freesurfer.read_geometry(surface_path)
        faces = faces.astype(np.int64)  # Ensure faces are of type int64
        mesh = {'vertices': coords, 'faces': faces}
        return mesh
    except Exception as e:
        print(f"Error loading surface file {surface_path}: {e}")
        raise

# Function to detect self-intersections using PyVista
def triangles_intersect(triangle1, vertices, faces):
    face = np.array([3, 0, 1, 2])
    faces_pv = np.hstack((np.full((faces.shape[0], 1), 3), faces)).astype(np.int64)
    surface1 = pv.PolyData(triangle1, face)
    surface2 = pv.PolyData(vertices, faces_pv.flatten())
    _, n_contacts = surface1.collision(surface2)
    return n_contacts > 0

def mesh2triangles(mesh):
    return mesh['vertices'][mesh['faces']]

def mesh2tricenters(mesh, triangles=None):
    if triangles is None:
        triangles = mesh2triangles(mesh)
    centers = np.mean(triangles, axis=1)
    return centers

def detachedtriangles(mesh, triangle_id, other_ids):
    # Exclude triangles that share vertices with the current triangle
    mask = np.any(np.isin(mesh['faces'][other_ids], mesh['faces'][triangle_id]), axis=1)
    faces = mesh['faces'][other_ids][~mask]
    return faces

def count_self_collisions(mesh, k=5):
    faces = mesh['faces']
    triangles = mesh2triangles(mesh)
    centers = mesh2tricenters(mesh, triangles=triangles)
    tree = cKDTree(centers)

    collision_count = 0
    for idx, triangle_center in enumerate(centers):
        dists, indices = tree.query(triangle_center.reshape(1, -1), k=k)
        other_indices = indices[0][1:]  # Exclude the triangle itself
        faces_to_check = detachedtriangles(mesh, idx, other_indices)
        if faces_to_check.size == 0:
            continue
        collision = triangles_intersect(triangles[idx], mesh['vertices'], faces_to_check)
        if collision:
            collision_count += 1

    return collision_count

# Function to write results to CSV
def write_to_csv(file_path, lock, data):
    # Since we're not using a lock, we can simplify this function
    file_exists = os.path.isfile(file_path)
    assert lock is not None, "error with lock"
    try:
        with lock:
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Framework', 'Subject ID', 'surf_type', 'Hemisphere', 'GNNLayers', 'Epoch', 'Metric', 'Label', 'Score', 'Total Triangles'])
                writer.writerow(data)
    except Exception as e:
        print(f"Error writing to CSV {file_path}: {e}")

# Function to extract subject ID from filename
def extract_subject_id(filename):
    match = re.search(r'hcp_[lr]h_(\d{6})_', filename)
    if match:
        return match.group(1)
    else:
        return None

# Process a subject and compute metrics
def process_subject(subj_id, csv_file_path, lock, framework_name, subject_base_path, gt_base_path, output_dir):
    print(f'Processing Subject: {subj_id}')
    hemispheres = ['lh', 'rh']
    surf_types = ['gm', 'wm']  # 'gm' for Grey Matter, 'wm' for White Matter

    for hemi in hemispheres:
        print(f'Hemisphere: {hemi}')
        for surf_type in surf_types:
            print(f'Surface Type: {surf_type}')
        
            # Define paths for predictions and ground truths
            pred_surf_dir = os.path.join(subject_base_path, surf_type, hemi)
            gt_surf_dir = os.path.join(gt_base_path, f"{surf_type}_gt", hemi)

            # Ensure prediction directory exists
            if not os.path.isdir(pred_surf_dir):
                print(f"Prediction directory does not exist: {pred_surf_dir}. Skipping.")
                continue

            # Ensure ground truth directory exists
            if not os.path.isdir(gt_surf_dir):
                print(f"Ground truth directory does not exist: {gt_surf_dir}. Skipping.")
                continue

            # List all prediction .surf files
            pred_files = [f for f in os.listdir(pred_surf_dir) if f.endswith('.surf') and (subj_id in f)]
            print(f'Prediction Files: {pred_files}')
            for pred_file in pred_files:
                # Extract GNN layers and epoch from the prediction filename
                print(f'Processing File: {pred_file}')
                if surf_type == 'gm':
                    pattern = r'hcp_[lr]h_(\d{6})_gnnlayers(\d+)_gm_pred_epoch(\d+)\.surf'
                else:
                    pattern = r'hcp_[lr]h_(\d{6})_gnnlayers(\d+)_wm_pred_epoch(\d+)\.surf'
                match = re.search(pattern, pred_file)
                if not match:
                    print(f"Filename pattern does not match expected format: {pred_file}. Skipping.")
                    continue
                gnn_layers = match.group(2)
                epoch = match.group(3)

                # Construct full paths for predicted and ground truth surfaces
                pred_surf_path = os.path.join(pred_surf_dir, pred_file)
                print(f'Prediction Surface Path: {pred_surf_path}')
                gt_file = f"hcp_{hemi}_{subj_id}_{surf_type}_gt.surf"
                gt_surf_path = os.path.join(gt_surf_dir, gt_file)
                print(f'Ground Truth Surface Path: {gt_surf_path}')
                # Check if ground truth file exists
                if not os.path.exists(gt_surf_path):
                    print(f"Ground truth file does not exist: {gt_surf_path}. Skipping.")
                    continue

                # Load meshes
                try:
                    pred_mesh = load_freesurfer_surface(pred_surf_path)
                    gt_mesh = load_freesurfer_surface(gt_surf_path)
                except Exception as e:
                    print(f"Error loading meshes for {subj_id}, {hemi}, {surf_type}: {e}")
                    continue

                # Compute Chamfer and Hausdorff distances
                chamfer_dist = compute_chamfer_distance(pred_mesh, gt_mesh)
                hausdorff_dist = compute_hausdorff_distance(pred_mesh, gt_mesh)

                # Compute self-intersections for the predicted mesh
                self_collision_count = count_self_collisions(pred_mesh, k=30)
                total_triangles = len(pred_mesh['faces'])

                # Prepare data rows
                data_chamfer = [framework_name, subj_id, surf_type, hemi, gnn_layers, epoch, 'Chamfer Distance', '', chamfer_dist, '']
                data_hausdorff = [framework_name, subj_id, surf_type, hemi, gnn_layers, epoch, 'Hausdorff Distance', '', hausdorff_dist, '']
                data_self_intersect = [framework_name, subj_id, surf_type, hemi, gnn_layers, epoch, 'Self-Intersections (SIF)', '', self_collision_count, total_triangles]

                # Write results to CSV
                write_to_csv(csv_file_path, lock, data_chamfer)
                write_to_csv(csv_file_path, lock, data_hausdorff)
                write_to_csv(csv_file_path, lock, data_self_intersect)

if __name__ == "__main__":
    # Use argparse for command-line argument parsing
    import argparse

    parser = argparse.ArgumentParser(description="Process subject arguments")

    # Define the command-line arguments
    parser.add_argument('--subj_id', type=str, required=True, help='Subject identifier (e.g., 201818)')
    parser.add_argument('--csv_file_path', type=str, required=True, help='Path to the CSV file where results will be appended')
    parser.add_argument('--framework_name', type=str, required=True, help='Name of the framework/model (e.g., csrp)')
    parser.add_argument('--subject_base_path', type=str, required=True, help='Base path to the subject\'s prediction directories')
    parser.add_argument('--gt_base_path', type=str, required=True, help='Base path to the subject\'s ground truth directories')

    args = parser.parse_args()

    subj_id = args.subj_id.strip()
    csv_file_path = args.csv_file_path.strip()
    framework_name = args.framework_name.strip()
    subject_base_path = args.subject_base_path.strip()
    gt_base_path = args.gt_base_path.strip()
    output_dir = '../result/'  # Retained for consistency

    print(f'Subject ID: {subj_id}')
    print(f'CSV File Path: {csv_file_path}')
    print(f'Framework Name: {framework_name}')
    print(f'Subject Base Path: {subject_base_path}')
    print(f'Ground Truth Base Path: {gt_base_path}')
    print(f'Output Directory: {output_dir}')

    # Initialize a lock for parallel-safe writing
    lock = Lock()
    process_subject(subj_id, csv_file_path, lock, framework_name, subject_base_path, gt_base_path, output_dir)
