import pyvista as pv
import os
import subprocess
import numpy as np
import argparse
import pickle
import sys
import nibabel as nib
# Ensure current_dir is defined appropriately
current_dir = os.getcwd()

# Assuming `remove_medial_wall.py` is located in '../../scripts/'
rmw_path = os.path.abspath(os.path.join(current_dir, '../../scripts/'))
sys.path.append(rmw_path)
from remove_medial_wall import save_mesh, scaleToMatchBoundingBox, alignCentersAndGetMatrix

# Import the private method using getattr
import remove_medial_wall
_alignMeshesAndGetMatrix = getattr(remove_medial_wall, '_alignMeshesAndGetMatrix')

def apply_affine(vertices, affine):
    """Apply affine transformation to vertices."""
    return np.dot(vertices, affine[:3, :3].T) + affine[:3, 3]

def save_mesh(mesh, file_path, file_format):
    """Save PyVista mesh to file."""
    if file_format == 'stl':
        mesh.save(file_path, binary=True)
    elif file_format == 'ply':
        mesh.save(file_path, binary=True)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

# Setting up argparse to handle command line arguments
parser = argparse.ArgumentParser(description="Mesh processing script")
parser.add_argument("--subjects_dir", required=True, help="Directory containing subject folders")
parser.add_argument("--subject_id", required=True, help="Subject ID")
parser.add_argument("--hemi", required=True, help="Hemisphere (lh/rh)")
parser.add_argument("--surfType", required=True, help="Surface type (pial/white)")
parser.add_argument("--project", required=True, help="Project name")
parser.add_argument("--project_gt_base_path", required=True, help="Base path for project's ground truth")
parser.add_argument("--project_pred_base_path", required=True, help="Base path for project's predictions")
parser.add_argument("--output_folder", required=True, help="Folder to save output meshes")

# Parse arguments
args = parser.parse_args()

# Assigning values from command line arguments
subjects_dir = args.subjects_dir
subject_id = args.subject_id
hemi = args.hemi
surfType = args.surfType
project = args.project
project_pred_base_path = args.project_pred_base_path
output_folder = args.output_folder

# Ensure output_folder exists
os.makedirs(output_folder, exist_ok=True)

proj_gt_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.{surfType}.deformed')
fs_gt_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.{surfType}')
print('a')

source_mesh = nib.freesurfer.io.read_geometry(proj_gt_path)  # Project's transformed ground truth
target_mesh = nib.freesurfer.io.read_geometry(fs_gt_path)
target_vertices, target_faces = target_mesh
target_mesh = pv.PolyData(target_vertices, target_faces)
print('b')

rotation_matrix = np.array([
        [1,  0,  0,  0],
        [0,  0, -1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  1]
    ])
print('c')
# Apply the rotation
source_vertices, source_faces = source_mesh
rotated_source_vertices = apply_affine(source_vertices, rotation_matrix)
rotated_source_mesh = pv.PolyData(rotated_source_vertices, source_faces)

print('d')
centered_source, centering_matrix = alignCentersAndGetMatrix(target_mesh, rotated_source_mesh)
print('d2')
scaled_source, scaling_matrix = scaleToMatchBoundingBox(centered_source, target_mesh)
print('d3')
aligned_source, icp_matrix = _alignMeshesAndGetMatrix(target_mesh, scaled_source, rigid=True)#if i add this line in get the segmentation fault.

print('e')
exit()#for debugging quickly. 
# Combine all transformations into one matrix
combined_transformation_matrix = icp_matrix @ scaling_matrix @ centering_matrix @ rotation_matrix

# Save the combined transformation matrix as a pickle file
matrix_filename = os.path.join(output_folder, f"{project}_{subject_id}_{hemi}_{surfType}_transformation_matrix.pkl")
with open(matrix_filename, 'wb') as matrix_file:
    pickle.dump(combined_transformation_matrix, matrix_file)

# Load predicted mesh from project's predictionshcp_lh_298455.white.stl
pred_path = os.path.join(project_pred_base_path, f"hcp_{hemi}_{subject_id}.{surfType}")
third_mesh_vertices, third_mesh_faces = nib.freesurfer.io.read_geometry(pred_path)

# Create a PyVista mesh from the loaded vertices and faces
third_mesh = pv.PolyData(third_mesh_vertices, third_mesh_faces)

# Save third_mesh as STL format
save_mesh(third_mesh, os.path.join(output_folder, f"{project}_{subject_id}_C_{hemi}_{surfType}.stl"), 'stl')

# Transform third_mesh using the combined transformation matrix
transformed_third_mesh = third_mesh.copy().transform(combined_transformation_matrix)

# Save transformed_third_mesh as STL format
save_mesh(transformed_third_mesh, os.path.join(output_folder, f"{project}_{subject_id}_CA_{hemi}_{surfType}.stl"), 'stl')

# Load the medial wall ply file
mw_file_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.{surfType}.medial_wall.ply')
# if not os.path.exists(mw_file_path):
#     createMedialWallPly(mw_file_path)

medial_wall = pv.read(mw_file_path)

# Save medial_wall as PLY format
save_mesh(medial_wall, os.path.join(output_folder, f"{project}_{subject_id}_mw_{hemi}_{surfType}.ply"), 'ply')

# Transform medial_wall using the inverse of combined_transformation_matrix
transformed_medial_wall = medial_wall.copy().transform(np.linalg.inv(combined_transformation_matrix))

# Save transformed_medial_wall as PLY format
save_mesh(transformed_medial_wall, os.path.join(output_folder, f"{project}_{subject_id}_invmw_{hemi}_{surfType}.ply"), 'ply')

# Perform minuspatch operation on third_mesh with transformed_medial_wall
modified_mesh = remove_medial_wall.minuspatch_optimized(third_mesh, transformed_medial_wall.points, K=60)
if isinstance(modified_mesh, pv.UnstructuredGrid):
    modified_mesh = modified_mesh.extract_surface()

modified_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)

# Save modified_mesh as STL format
save_mesh(modified_mesh, os.path.join(output_folder, f"{project}_{subject_id}_C_mwrm_{hemi}_{surfType}.stl"), 'stl')
