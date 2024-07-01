import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
import nibabel as nib
import re
import os
import csv
import sys

# Add the base directory (two levels up) to the system path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from data.preprocess import process_surface_inverse
from data.dataloader import BrainDataset
from util.mesh import compute_dice
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from config import load_config

def apply_affine(vertices, affine):
    """Apply affine transformation to vertices."""
    return np.dot(vertices, affine[:3, :3].T) + affine[:3, 3]

def compute_chamfer_distance(pc1, pc2):
    """Compute the Chamfer distance between two point clouds using cKDTree."""
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    distances1, _ = tree1.query(pc2, k=1)
    distances2, _ = tree2.query(pc1, k=1)
    
    chamfer_dist = np.mean(distances1) + np.mean(distances2)
    return chamfer_dist

def compute_hausdorff_distance(pc1, pc2):
    """Compute the Hausdorff distance between two point clouds using cKDTree."""
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    distances1, _ = tree1.query(pc2, k=1)
    distances2, _ = tree2.query(pc1, k=1)
    
    hausdorff_dist = max(np.max(distances1), np.max(distances2))
    return hausdorff_dist

def translate_to_origin(vertices):
    """Translate vertices to the origin based on their centroid."""
    centroid = np.mean(vertices, axis=0)
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -centroid
    translated_vertices = apply_affine(vertices, translation_matrix)
    return translated_vertices, translation_matrix

def align_and_compare_surfaces(vertices, faces, target_vertices, target_faces, result_dir, subid, config):
    # Define the 90-degree rotation matrix around the X-axis
    rotation_matrix = np.array([
        [1,  0,  0,  0],
        [0,  0, -1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  1]
    ])
    
    # Apply the rotation
    rotated_vertices = apply_affine(vertices, rotation_matrix)
    
    # Translate meshes to origin
    translated_rotated_vertices, rotation_translation_matrix = translate_to_origin(rotated_vertices)
    save_mesh(translated_rotated_vertices, faces, os.path.join(result_dir, f'{subid}_{config.surf_type}_{config.surf_hemi}_translated_rotated'))

    translated_target_vertices, target_translation_matrix = translate_to_origin(target_vertices)
    save_mesh(translated_target_vertices, target_faces, os.path.join(result_dir, f'{subid}_{config.surf_type}_{config.surf_hemi}_translated_target'))

    # Convert to numpy arrays
    rotated_points = translated_rotated_vertices.astype(np.float64)
    target_points = translated_target_vertices.astype(np.float64)
    
    # Apply ICP (rigid body transformation)
    icp_result = trimesh.registration.icp(rotated_points, target_points, max_iterations=50)
    icp_matrix = icp_result[0]
    
    # Combine all transformations
    combined_matrix = np.eye(4)
    combined_matrix = np.dot(combined_matrix, rotation_matrix)
    combined_matrix = np.dot(combined_matrix, rotation_translation_matrix)
    combined_matrix = np.dot(combined_matrix, icp_matrix)
    combined_matrix = np.dot(combined_matrix, np.linalg.inv(target_translation_matrix))

    # Apply the combined transformation to get the final aligned vertices
    final_aligned_vertices = apply_affine(vertices, combined_matrix)
    
    return final_aligned_vertices, faces, combined_matrix

def save_mesh(vertices, faces, file_path, affine=None):
    """Save the mesh to a file in FreeSurfer format."""
    if affine is not None:
        vertices = apply_affine(vertices, affine)
    nib.freesurfer.io.write_geometry(file_path, vertices, faces)

def main(config):
    dataset = BrainDataset(config, 'test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    result_dir = config.result_dir
    csv_file_path = os.path.join(result_dir, 'cortexoderesults.csv')
    framework_name = config.model_type
    
    # Open CSV file for writing results
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Framework", "Subject ID", "Surf_type", "Hemisphere", "Metric Name", "Metric Value"])
        
        for brain_arr, v_in, v_gt, f_in, f_gt, subid, aff in dataloader:
            # Ensure subid is a number
            subid = re.sub(r'\D', '', str(subid))
            if subid != '201818':
                continue            
            
            # Apply process_surface_inverse to v_gt and f_gt
            v_gt, f_gt = process_surface_inverse(v_gt.squeeze().numpy(), f_gt.squeeze().numpy(), config.data_name)
            
            # Convert surf_type to FreeSurfer format
            surf_type_fs = 'white' if config.surf_type == 'wm' else 'pial'
            
            # Construct target surface file path
            target_surface_path = os.path.join(config.data_dir, 'test', f'{subid}/surf/{config.surf_hemi}.{surf_type_fs}')
            target_surface = nib.freesurfer.io.read_geometry(target_surface_path)
            target_vertices, target_faces, target_affine = target_surface
            
            # Save the target mesh for debugging
            save_mesh(target_vertices, target_faces, os.path.join(result_dir, f'{subid}_{config.surf_type}_{config.surf_hemi}_target'))
            
            # Align and compare surfaces
            final_aligned_vertices, final_aligned_faces, combined_matrix = align_and_compare_surfaces(v_gt, f_gt, target_vertices, target_faces, result_dir, subid, config)
            
            # Save the meshes in FreeSurfer format
            aligned_mesh_path = os.path.join(result_dir, f'{subid}_{config.surf_type}_{config.surf_hemi}_alignedgt')
            save_mesh(final_aligned_vertices, final_aligned_faces, aligned_mesh_path, target_affine)
            
            # Print results
            print(f'Subject ID: {subid}')
            
            # Load the generated surface file
            generated_surface_path = os.path.join(result_dir, f'{config.data_name}_{config.surf_hemi}_{subid}.{surf_type_fs}')
            generated_vertices, generated_faces = nib.freesurfer.io.read_geometry(generated_surface_path)
            
            # Apply the combined transformation to the generated surface
            transformed_generated_vertices = apply_affine(generated_vertices, combined_matrix)
            
            # Save the transformed generated surface
            transformed_generated_path = os.path.join(result_dir, f'{subid}_{config.surf_type}_{config.surf_hemi}_generated_transformed')
            save_mesh(transformed_generated_vertices, generated_faces, transformed_generated_path, target_affine)
            
            # Compute distances between final aligned and final input surfaces
            chamfer_dist = compute_chamfer_distance(final_aligned_vertices, transformed_generated_vertices)
            hausdorff_dist = compute_hausdorff_distance(final_aligned_vertices, transformed_generated_vertices)
            
            # Print distance results
            print(f'Chamfer distance: {chamfer_dist}')
            print(f'Hausdorff distance: {hausdorff_dist}')
            
            # Write results to CSV
            writer.writerow([framework_name, subid, config.surf_type, config.surf_hemi, "Chamfer Distance", chamfer_dist])
            writer.writerow([framework_name, subid, config.surf_type, config.surf_hemi, "Hausdorff Distance", hausdorff_dist])

if __name__ == "__main__":
    config = load_config()
    main(config)
