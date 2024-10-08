import pyvista as pv
import os
import sys
import numpy as np
import argparse
import pickle
from scipy.spatial import cKDTree

# Function to apply affine transformation
def apply_affine(vertices, affine):
    """Apply affine transformation to vertices."""
    return np.dot(vertices, affine[:3, :3].T) + affine[:3, 3]

# Function to save PyVista mesh
def save_mesh(mesh, file_path, file_format):
    """Save PyVista mesh to file."""
    if file_format == 'stl':
        mesh.save(file_path, binary=True)
    elif file_format == 'ply':
        mesh.save(file_path, binary=True)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

# Minuspatch function
def minuspatch_optimized(meshA, patch, K=60):
    """
    Remove patches from meshA based on proximity to points in patch.
    """
    centroids = meshA.triangles_center
    kd_tree = cKDTree(centroids)
    face_indices = set()
    for point in patch:
        _, indices = kd_tree.query(point, k=K)
        face_indices.update(indices)
    mask = np.ones(len(meshA.faces), dtype=bool)
    mask[list(face_indices)] = False
    clean_mesh = meshA.submesh([mask], append=True)
    return clean_mesh

def process_mesh(result_dir, condition, hemi, surfType, output_folder, project):
    """Process a single mesh for a given condition and hemisphere."""
    # Define input paths
    pred_surf_path = os.path.join(result_dir, condition, 'wm', hemi, f'hcp_{hemi}_sub-01_gnnlayers4_wm_pred_epoch50.surf')
    gt_surf_path = os.path.join(result_dir, condition, 'wm_gt', hemi, f'hcp_{hemi}_sub-01_wm_gt.surf')
    medial_wall_pkl = os.path.join(output_folder, f"{project}_{hemi}_{surfType}_medial_wall.pkl")  # Adjust as per new naming

    # Load meshes
    pred_vertices, pred_faces = load_freesurfer_mesh(pred_surf_path)
    gt_vertices, gt_faces = load_freesurfer_mesh(gt_surf_path)

    # Load medial wall vertices
    with open(medial_wall_pkl, 'rb') as f:
        medial_wall_vertices = pickle.load(f)

    # Create PyVista meshes
    pred_mesh = pv.PolyData(pred_vertices, pred_faces)
    gt_mesh = pv.PolyData(gt_vertices, gt_faces)

    # Remove medial wall from predicted mesh
    cleaned_pred_mesh = minuspatch_optimized(pred_mesh, medial_wall_vertices, K=60)

    # Save cleaned mesh
    save_mesh(cleaned_pred_mesh, os.path.join(output_folder, f"{project}_{hemi}_cleaned_pred_wm.stl"), 'stl')

    # Similarly, process ground truth mesh if needed
    # ...

def load_freesurfer_mesh(fs_path):
    """Load FreeSurfer geometry."""
    import nibabel as nib
    vertices, faces = nib.freesurfer.io.read_geometry(fs_path)
    return vertices, faces

def main():
    parser = argparse.ArgumentParser(description='Remove medial walls from meshes.')
    parser.add_argument('--result_dir', required=True, help='Path to result directory with conditions.')
    parser.add_argument('--conditions', nargs='+', required=True, help='List of conditions to process (e.g., a b cortexode).')
    parser.add_argument('--hemis', nargs='+', required=True, help='List of hemispheres to process (e.g., lh rh).')
    parser.add_argument('--surfType', required=True, help='Surface type (pial/white).')
    parser.add_argument('--project', required=True, help='Project name.')
    parser.add_argument('--output_folder', required=True, help='Folder to save cleaned meshes.')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    for condition in args.conditions:
        for hemi in args.hemis:
            process_mesh(args.result_dir, condition, hemi, args.surfType, args.output_folder, args.project)

if __name__ == "__main__":
    main()
