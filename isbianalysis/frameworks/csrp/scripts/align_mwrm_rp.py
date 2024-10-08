import os
import numpy as np
import argparse
import pickle
import trimesh
from scipy.spatial import cKDTree
import nibabel as nib

def apply_affine(vertices, affine):
    """Apply affine transformation to vertices."""
    return np.dot(vertices, affine[:3, :3].T) + affine[:3, 3]

def save_mesh_as_stl(vertices, faces, file_path):
    """Save vertices and faces as an STL file."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(file_path)

def load_freesurfer_mesh(fs_path):
    """Load FreeSurfer geometry."""
    vertices, faces = nib.freesurfer.io.read_geometry(fs_path)
    return vertices, faces

def icp_alignment(target_vertices, source_vertices, max_iterations=1000):
    """Perform ICP alignment using trimesh."""
    # Using trimesh's built-in ICP might require adaptation
    # Here, we perform a simple ICP alignment
    import open3d as o3d

    source = o3d.geometry.TriangleMesh()
    source.vertices = o3d.utility.Vector3dVector(source_vertices)
    source.triangles = o3d.utility.Vector3iVector(source_faces)
    source.compute_vertex_normals()

    target = o3d.geometry.TriangleMesh()
    target.vertices = o3d.utility.Vector3dVector(target_vertices)
    target.triangles = o3d.utility.Vector3iVector(target_faces)
    target.compute_vertex_normals()

    threshold = 1.0  # Distance threshold
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    
    aligned_source = source.transform(reg_p2p.transformation)
    return reg_p2p.transformation, np.asarray(aligned_source.vertices)

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

def process_condition(result_dir, condition, hemi, surfType, project, output_folder):
    """Process a single condition and hemisphere."""
    # Define paths
    predicted_wm_surf = os.path.join(result_dir, condition, 'wm', hemi, f'hcp_{hemi}_sub-01_gnnlayers4_wm_pred_epoch50.surf')  # Example pattern
    predicted_gm_surf = os.path.join(result_dir, condition, 'gm', hemi, f'hcp_{hemi}_sub-01_gnnlayers4_gm_pred_epoch25.surf')
    ground_truth_wm_surf = os.path.join(result_dir, condition, 'wm_gt', hemi, f'hcp_{hemi}_sub-01_wm_gt.surf')
    ground_truth_gm_surf = os.path.join(result_dir, condition, 'gm_gt', hemi, f'hcp_{hemi}_sub-01_gm_gt.surf')

    # Load meshes
    pred_wm_vertices, pred_wm_faces = load_freesurfer_mesh(predicted_wm_surf)
    pred_gm_vertices, pred_gm_faces = load_freesurfer_mesh(predicted_gm_surf)
    gt_wm_vertices, gt_wm_faces = load_freesurfer_mesh(ground_truth_wm_surf)
    gt_gm_vertices, gt_gm_faces = load_freesurfer_mesh(ground_truth_gm_surf)

    # Create trimesh objects
    pred_wm_mesh = trimesh.Trimesh(vertices=pred_wm_vertices, faces=pred_wm_faces)
    gt_wm_mesh = trimesh.Trimesh(vertices=gt_wm_vertices, faces=gt_wm_faces)

    # Save initial meshes
    save_mesh_as_stl(pred_wm_vertices, pred_wm_faces, os.path.join(output_folder, f"{project}_{hemi}_pred_wm.stl"))
    save_mesh_as_stl(gt_wm_vertices, gt_wm_faces, os.path.join(output_folder, f"{project}_{hemi}_gt_wm.stl"))

    # Apply initial rotation
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    rotated_pred_vertices = apply_affine(pred_wm_vertices, rotation_matrix)
    rotated_pred_mesh = trimesh.Trimesh(vertices=rotated_pred_vertices, faces=pred_wm_faces)

    # Compute initial transformations
    centering_matrix, scaling_matrix = compute_initial_transformations(gt_wm_mesh, rotated_pred_mesh)

    # Apply transformations
    centered_pred_vertices = apply_affine(rotated_pred_vertices, centering_matrix)
    centered_pred_mesh = trimesh.Trimesh(vertices=centered_pred_vertices, faces=rotated_pred_mesh.faces)

    scaled_pred_vertices = apply_affine(centered_pred_vertices, scaling_matrix)
    scaled_pred_mesh = trimesh.Trimesh(vertices=scaled_pred_vertices, faces=centered_pred_mesh.faces)

    # Perform ICP alignment
    icp_matrix, aligned_pred_vertices = icp_alignment(gt_wm_vertices, scaled_pred_vertices)

    # Combine all transformations
    combined_transformation_matrix = icp_matrix @ scaling_matrix @ centering_matrix @ rotation_matrix

    # Save aligned mesh
    save_mesh_as_stl(aligned_pred_vertices, pred_wm_faces, os.path.join(output_folder, f"{project}_{hemi}_aligned_pred_wm.stl"))

    # Save transformation matrix
    matrix_filename = os.path.join(output_folder, f"{project}_{hemi}_transformation_matrix.pkl")
    with open(matrix_filename, 'wb') as f:
        pickle.dump(combined_transformation_matrix, f)

    # Further processing like medial wall removal can be added here

def main():
    parser = argparse.ArgumentParser(description='Align cortical surface meshes.')
    parser.add_argument('--result_dir', required=True, help='Path to result directory with conditions.')
    parser.add_argument('--conditions', nargs='+', required=True, help='List of conditions to process (e.g., a b cortexode).')
    parser.add_argument('--hemis', nargs='+', required=True, help='List of hemispheres to process (e.g., lh rh).')
    parser.add_argument('--surfType', required=True, help='Surface type (pial/white).')
    parser.add_argument('--project', required=True, help='Project name.')
    parser.add_argument('--output_folder', required=True, help='Folder to save output meshes.')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    for condition in args.conditions:
        for hemi in args.hemis:
            process_condition(args.result_dir, condition, hemi, args.surfType, args.project, args.output_folder)

if __name__ == "__main__":
    main()
