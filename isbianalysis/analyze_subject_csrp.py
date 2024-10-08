import os
import csv
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
import vtk
from multiprocessing import Lock
import sys
import logging
import re

# Turn off VTK logging verbosity
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
import pyvista as pv

# Configure logging
logging.basicConfig(
    filename='compute_measures.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Function to compute Chamfer Distance using KD-Tree
def compute_chamfer_distance(mesh1, mesh2):
    """
    Compute the Chamfer Distance between two meshes.

    Parameters:
    - mesh1 (dict): Dictionary with 'vertices' and 'faces' for the first mesh.
    - mesh2 (dict): Dictionary with 'vertices' and 'faces' for the second mesh.

    Returns:
    - float: Chamfer Distance.
    """
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
        logging.error(f"Error computing Chamfer Distance: {e}")
        return np.nan

# Function to compute Hausdorff Distance using KD-Tree
def compute_hausdorff_distance(mesh1, mesh2):
    """
    Compute the Hausdorff Distance between two meshes.

    Parameters:
    - mesh1 (dict): Dictionary with 'vertices' and 'faces' for the first mesh.
    - mesh2 (dict): Dictionary with 'vertices' and 'faces' for the second mesh.

    Returns:
    - float: Hausdorff Distance.
    """
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
        logging.error(f"Error computing Hausdorff Distance: {e}")
        return np.nan

# Functions for self-intersection detection
def triangles_intersect(triangle1, vertices, faces):
    """
    Check if a triangle intersects with any other triangles in the mesh.

    Parameters:
    - triangle1 (np.ndarray): Coordinates of the triangle vertices.
    - vertices (np.ndarray): All vertices in the mesh.
    - faces (np.ndarray): Indices of vertices forming each face.

    Returns:
    - bool: True if intersection is found, False otherwise.
    """
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
    """
    Convert mesh faces to triangle coordinates.

    Parameters:
    - mesh (dict): Dictionary with 'vertices' and 'faces'.

    Returns:
    - np.ndarray: Array of triangle vertex coordinates.
    """
    return mesh['vertices'][mesh['faces']]

def mesh2tricenters(mesh, triangles=None):
    """
    Compute the centers of all triangles in the mesh.

    Parameters:
    - mesh (dict): Dictionary with 'vertices' and 'faces'.
    - triangles (np.ndarray, optional): Precomputed triangles.

    Returns:
    - np.ndarray: Array of triangle centers.
    """
    if triangles is None:
        triangles = mesh2triangles(mesh)
    centers = np.mean(triangles, axis=1)
    return centers

def count_self_collisions(mesh, k=5):
    """
    Count the number of self-intersections in a mesh.

    Parameters:
    - mesh (dict): Dictionary with 'vertices' and 'faces'.
    - k (int): Number of nearest neighbors to consider.

    Returns:
    - int: Number of self-intersections detected.
    """
    faces = mesh['faces']
    triangles = mesh2triangles(mesh)
    centers = mesh2tricenters(mesh, triangles=triangles)
    tree = cKDTree(centers)

    collision_count = 0
    for idx, triangle in enumerate(centers):
        dists, indices = tree.query(triangle.reshape(1, -1), k=k)
        faces_subset = detachedtriangles(mesh, idx, indices[0][1:])
        if faces_subset.size == 0:
            logging.warning('k is too small; no faces to compare for this triangle.')
            continue
        collision = triangles_intersect(triangles[idx, :, :],
                                        mesh['vertices'][np.sort(np.unique(faces_subset.flatten()))],
                                        faces_subset)
        if collision:
            collision_count += 1

    return collision_count

def detachedtriangles(mesh, triangle_id, other_ids):
    """
    Detach triangles that share vertices with the given triangle.

    Parameters:
    - mesh (dict): Dictionary with 'vertices' and 'faces'.
    - triangle_id (int): Index of the triangle to exclude.
    - other_ids (list or np.ndarray): Indices of other triangles.

    Returns:
    - np.ndarray: Faces of detached triangles.
    """
    mask = np.any(np.isin(mesh['faces'][other_ids], mesh['faces'][triangle_id]), axis=1)
    faces = mesh['faces'][other_ids][~mask]
    return faces

# Function to load Freesurfer surface files
def load_freesurfer_surface(surface_path):
    """
    Load a Freesurfer surface file.

    Parameters:
    - surface_path (str): Path to the .surf file.

    Returns:
    - dict: Dictionary containing 'vertices' and 'faces'.
    """
    try:
        coords, faces = nib.freesurfer.read_geometry(surface_path)
        mesh = {'vertices': coords, 'faces': faces}
        return mesh
    except Exception as e:
        logging.error(f"Error loading surface file {surface_path}: {e}")
        raise

# Function to write results to CSV
def write_to_csv(file_path, lock, data):
    """
    Write a row of data to the CSV file in a thread-safe manner.

    Parameters:
    - file_path (str): Path to the CSV file.
    - lock (multiprocessing.Lock): Lock object for thread-safe writing.
    - data (list): List of data elements to write as a row.
    """
    with lock:
        file_exists = os.path.isfile(file_path)
        try:
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Framework', 'Subject ID','surf_type','Hemisphere', 'Metric', 'Label', 'Score', 'Total Triangles'])
                writer.writerow(data)
        except Exception as e:
            logging.error(f"Error writing to CSV {file_path}: {e}")

# Function to extract subject ID from filename
def extract_subject_id(filename):
    """
    Extract the subject ID from the filename.

    Expected filename pattern: hcp_lh_201818_gnnlayers4_gm_pred_epoch100.surf

    Parameters:
    - filename (str): Filename to extract subject ID from.

    Returns:
    - str or None: Extracted subject ID or None if not found.
    """
    match = re.search(r'hcp_[lr]h_(\d{6})_', filename)
    if match:
        return match.group(1)
    else:
        return None

# Process a subject and compute metrics
def process_subject(subj_id, csv_file_path, lock, framework_name, subject_base_path, gt_base_path, output_dir):
    """
    Process a single subject by computing geometric measures for each surface type and hemisphere.

    Parameters:
    - subj_id (str): Subject identifier.
    - csv_file_path (str): Path to the CSV file for logging results.
    - lock (multiprocessing.Lock): Lock object for thread-safe writing.
    - framework_name (str): Name of the framework/model.
    - subject_base_path (str): Base path to the subject's prediction directories.
    - gt_base_path (str): Base path to the subject's ground truth directories.
    - output_dir (str): Directory where results are stored.
    """
    print('Processing Subject:', subj_id)
    hemispheres = ['lh', 'rh']
    surf_types = ['gm', 'wm']  # 'gm' for Grey Matter, 'wm' for White Matter

    for hemi in hemispheres:
        for surf_type in surf_types:
            # Define paths for predictions and ground truths
            pred_surf_dir = os.path.join(subject_base_path, surf_type, hemi)
            gt_surf_dir = os.path.join(gt_base_path, f"{surf_type}_gt", hemi)

            # Ensure prediction directory exists
            if not os.path.isdir(pred_surf_dir):
                logging.warning(f"Prediction directory does not exist: {pred_surf_dir}. Skipping.")
                continue

            # Ensure ground truth directory exists
            if not os.path.isdir(gt_surf_dir):
                logging.warning(f"Ground truth directory does not exist: {gt_surf_dir}. Skipping.")
                continue

            # List all prediction .surf files
            pred_files = [f for f in os.listdir(pred_surf_dir) if f.endswith('.surf')]

            for pred_file in pred_files:
                # Extract GNN layers and epoch from the prediction filename
                if surf_type == 'gm':
                    pattern = r'gnnlayers(\d+)_gm_pred_epoch(\d+)\.surf'
                else:
                    pattern = r'gnnlayers(\d+)_wm_pred_epoch(\d+)\.surf'
                match = re.search(pattern, pred_file)
                if not match:
                    logging.warning(f"Filename pattern does not match expected format: {pred_file}. Skipping.")
                    continue
                gnn_layers = match.group(1)
                epoch = match.group(2)

                # Construct full paths for predicted and ground truth surfaces
                pred_surf_path = os.path.join(pred_surf_dir, pred_file)

                # Determine corresponding ground truth file
                gt_file = f"hcp_{hemi}_{subj_id}_{surf_type}_gt.surf"
                gt_surf_path = os.path.join(gt_surf_dir, gt_file)

                # Check if ground truth file exists
                if not os.path.exists(gt_surf_path):
                    logging.warning(f"Ground truth file does not exist: {gt_surf_path}. Skipping.")
                    continue

                # Load meshes
                try:
                    pred_mesh = load_freesurfer_surface(pred_surf_path)
                    gt_mesh = load_freesurfer_surface(gt_surf_path)
                except Exception as e:
                    logging.error(f"Error loading meshes for {subj_id}, {hemi}, {surf_type}: {e}")
                    continue

                # Compute Chamfer and Hausdorff distances
                chamfer_dist = compute_chamfer_distance(pred_mesh, gt_mesh)
                hausdorff_dist = compute_hausdorff_distance(pred_mesh, gt_mesh)

                # Compute self-intersections for the predicted mesh
                self_collision_count = count_self_collisions(pred_mesh, k=30)
                total_triangles = len(pred_mesh['faces'])

                # Prepare data rows
                data_chamfer = [framework_name, subj_id, surf_type, hemi, 'Chamfer Distance', '', chamfer_dist, '']
                data_hausdorff = [framework_name, subj_id, surf_type, hemi, 'Hausdorff Distance', '', hausdorff_dist, '']
                data_self_intersect = [framework_name, subj_id, surf_type, hemi, 'Self-Intersections (SIF)', '', self_collision_count, total_triangles]

                # Write results to CSV
                write_to_csv(csv_file_path, lock, data_chamfer)
                write_to_csv(csv_file_path, lock, data_hausdorff)
                write_to_csv(csv_file_path, lock, data_self_intersect)

    print(f"Completed processing for subject {subj_id}")

if __name__ == "__main__":
    """
    Command-line Arguments:
    1. subj_id: Subject identifier (e.g., 201818)
    2. csv_file_path: Path to the CSV file where results will be appended
    3. framework_name: Name of the framework/model (e.g., csrp)
    4. subject_base_path: Base path to the subject's prediction directories
    5. gt_base_path: Base path to the subject's ground truth directories
    """

    if len(sys.argv) < 6:
        print("Usage: python analyze_subject_csrp.py <subj_id> <csv_file_path> <framework_name> <subject_base_path> <gt_base_path>")
        sys.exit(1)
    
    subj_id = sys.argv[1]
    csv_file_path = sys.argv[2]
    framework_name = sys.argv[3]
    subject_base_path = sys.argv[4]
    gt_base_path = sys.argv[5]
    output_dir = '../result/'

    print('Subject ID:', subj_id)
    print('CSV File Path:', csv_file_path)
    print('Framework Name:', framework_name)
    print('Subject Base Path:', subject_base_path)
    print('Ground Truth Base Path:', gt_base_path)
    print("Output Directory:", output_dir)
    
    # Initialize a lock for parallel-safe writing
    lock = Lock()
    
    process_subject(subj_id, csv_file_path, lock, framework_name, subject_base_path, gt_base_path, output_dir)
