import os
import numpy as np
import argparse
import pickle
import sys
import nibabel as nib
import trimesh
from scipy.spatial import cKDTree

def apply_affine(vertices, affine):
    """Apply affine transformation to vertices."""
    return np.dot(vertices, affine[:3, :3].T) + affine[:3, 3]

def save_freesurfer_mesh(vertices, faces, file_path):
    """Save vertices and faces as a FreeSurfer geometry file."""
    nib.freesurfer.io.write_geometry(file_path, vertices, faces)

def load_freesurfer_mesh(fs_path):
    """Load FreeSurfer geometry."""
    vertices, faces = nib.freesurfer.io.read_geometry(fs_path)
    return vertices, faces

def icp_alignment(target_vertices, source_vertices, max_iterations=1000):
    """Perform ICP alignment using trimesh."""
    matrix, aligned, cost = trimesh.registration.icp(source_vertices, target_vertices, max_iterations=max_iterations)
    aligned_mesh = trimesh.Trimesh(vertices=aligned, faces=source_faces)
    return matrix, aligned_mesh

def compute_initial_transformations(target_mesh, source_mesh):
    """Compute initial transformations: centering and scaling."""
    target_center = target_mesh.centroid
    source_center = source_mesh.centroid
    translation = target_center - source_center
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation

    scale_factors = (target_mesh.bounds[1] - target_mesh.bounds[0]) / (source_mesh.bounds[1] - source_mesh.bounds[0])
    scaling_matrix = np.diag(np.append(scale_factors, 1))

    return translation_matrix, scaling_matrix

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

# Paths to FreeSurfer geometry files
proj_gt_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.{surfType}.deformed')
fs_gt_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.{surfType}')

# Load FreeSurfer meshes
source_vertices, source_faces = load_freesurfer_mesh(proj_gt_path)
target_vertices, target_faces = load_freesurfer_mesh(fs_gt_path)

# Create trimesh objects
source_mesh = trimesh.Trimesh(vertices=source_vertices, faces=source_faces)
target_mesh = trimesh.Trimesh(vertices=target_vertices, faces=target_faces)

# Save the loaded meshes
save_freesurfer_mesh(source_vertices, source_faces, os.path.join(output_folder, f"{project}_{subject_id}_B_{hemi}_{surfType}"))
save_freesurfer_mesh(target_vertices, target_faces, os.path.join(output_folder, f"{project}_{subject_id}_A_{hemi}_{surfType}"))

# Apply initial rotation to source mesh
rotation_matrix = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

rotated_source_vertices = apply_affine(source_mesh.vertices, rotation_matrix)
rotated_source_mesh = trimesh.Trimesh(vertices=rotated_source_vertices, faces=source_mesh.faces)

# Compute initial transformations: centering and scaling
centering_matrix, scaling_matrix = compute_initial_transformations(target_mesh, rotated_source_mesh)

# Apply initial transformations
centered_source_vertices = apply_affine(rotated_source_vertices, centering_matrix)
centered_source_mesh = trimesh.Trimesh(vertices=centered_source_vertices, faces=rotated_source_mesh.faces)

scaled_source_vertices = apply_affine(centered_source_vertices, scaling_matrix)
scaled_source_mesh = trimesh.Trimesh(vertices=scaled_source_vertices, faces=centered_source_mesh.faces)

# Perform ICP alignment
icp_matrix, aligned_source_mesh = icp_alignment(target_mesh.vertices, scaled_source_mesh.vertices)

# Combine all transformations into one matrix
combined_transformation_matrix = icp_matrix @ scaling_matrix @ centering_matrix @ rotation_matrix

# Save the aligned mesh
save_freesurfer_mesh(aligned_source_mesh.vertices, aligned_source_mesh.faces, os.path.join(output_folder, f"{project}_{subject_id}_BA_{hemi}_{surfType}"))

# Save the combined transformation matrix
matrix_filename = os.path.join(output_folder, f"{project}_{subject_id}_{hemi}_{surfType}_transformation_matrix.pkl")
with open(matrix_filename, 'wb') as matrix_file:
    pickle.dump(combined_transformation_matrix, matrix_file)

# Load predicted mesh from project's predictions
pred_path = os.path.join(project_pred_base_path, f"hcp_{hemi}_{subject_id}.pred.{surfType}")
pred_vertices, pred_faces = load_freesurfer_mesh(pred_path)
pred_mesh = trimesh.Trimesh(vertices=pred_vertices, faces=pred_faces)

# Save predicted mesh
save_freesurfer_mesh(pred_vertices, pred_faces, os.path.join(output_folder, f"{project}_{subject_id}_C_{hemi}_{surfType}"))

# Transform predicted mesh using the combined transformation matrix
transformed_pred_vertices = apply_affine(pred_mesh.vertices, combined_transformation_matrix)
transformed_pred_mesh = trimesh.Trimesh(vertices=transformed_pred_vertices, faces=pred_mesh.faces)

# Save transformed predicted mesh
save_freesurfer_mesh(transformed_pred_mesh.vertices, transformed_pred_mesh.faces, os.path.join(output_folder, f"{project}_{subject_id}_CA_{hemi}_{surfType}"))

# Load the medial wall ply file
mw_file_path = os.path.join('/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/freesurfer/mwremoved',
                            f'{subject_id}.{hemi}.{surfType}.medial_wall.ply')

medial_wall = trimesh.load(mw_file_path)

# Check if medial_wall is a PointCloud
if isinstance(medial_wall, trimesh.points.PointCloud):
    medial_wall = trimesh.Trimesh(vertices=medial_wall.vertices, faces=[[0, 0, 0]])

# Transform medial wall using the inverse of combined transformation_matrix
inverse_transformation_matrix = np.linalg.inv(combined_transformation_matrix)
transformed_mw_vertices = apply_affine(medial_wall.vertices, inverse_transformation_matrix)
transformed_medial_wall = trimesh.Trimesh(vertices=transformed_mw_vertices, faces=medial_wall.faces)

# Save transformed medial wall mesh
save_freesurfer_mesh(transformed_medial_wall.vertices, transformed_medial_wall.faces, os.path.join(output_folder, f"{project}_{subject_id}_invmw_{hemi}_{surfType}"))

# Perform minuspatch operation (adjust the function as needed for trimesh)
def minuspatch_optimized(meshA, patch, K=1):
    # Calculate centroids of faces in the mesh
    faces = meshA.faces
    centroids = meshA.triangles_center

    # Build KDTree using centroids
    kd_tree = cKDTree(centroids)

    # Find nearest neighbor faces for each point in the patch
    face_indices = set()
    for point in patch:
        _, indices = kd_tree.query(point, k=K)
        face_indices.update(indices)

    # Remove the faces from the mesh
    mask = np.ones(len(meshA.faces), dtype=bool)
    mask[list(face_indices)] = False
    clean_mesh = meshA.submesh([mask], append=True)

    return clean_mesh

modified_mesh = minuspatch_optimized(pred_mesh, transformed_medial_wall.vertices, K=60)

# Save modified mesh
save_freesurfer_mesh(modified_mesh.vertices, modified_mesh.faces, os.path.join(output_folder, f"{project}_{subject_id}_C_mwrm_{hemi}_{surfType}"))
