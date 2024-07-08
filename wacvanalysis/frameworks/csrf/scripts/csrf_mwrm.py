#/data/users2/washbee/speedrun/CortexODE_fork/ckpts/rp2-hcp
#hcp_lh_201818.pial
#hcp_lh_201818.white

import numpy as np
from stl import mesh
from scipy.spatial import cKDTree
import os
import nibabel as nib

def load_freesurfer_points(fs_path):
    surf = nib.freesurfer.read_geometry(fs_path)
    points = surf[0]  # vertices
    return points

def nearest_neighbor_faces(stl_mesh, points, num_faces_per_point):
    face_centroids = np.mean(np.array([stl_mesh.v0, stl_mesh.v1, stl_mesh.v2]), axis=0)
    kd_tree = cKDTree(face_centroids)
    
    face_indices = set()
    for point in points:
        _, indices = kd_tree.query(point, k=num_faces_per_point)
        face_indices.update(indices)

    # Filter out invalid indices
    valid_indices = set(index for index in face_indices if index != np.inf and index < len(stl_mesh.data))
    return valid_indices

def remove_faces(stl_mesh, face_indices_to_remove):
    return mesh.Mesh(np.delete(stl_mesh.data, list(face_indices_to_remove), axis=0))

# Paths
subjects_dir = "/data/users2/washbee/speedrun/cortexode-data-rp/"
subject = '201818'
fs_path = os.path.join(subjects_dir,subject,'surf','lh.pial.deformed')  # FreeSurfer surface file path
pred_path = os.path.join('/data/users2/washbee/speedrun/CortexODE_fork/ckpts/rp2-hcp/', 'wm_hcp_lh_201818.stl')
output_path = os.path.join('/data/users2/washbee/speedrun/CortexODE_fork/ckpts/rp2-hcp/wm_hcp_lh_201818_modified.stl')

# Load medial wall points from FreeSurfer surface file
medial_wall_points = load_freesurfer_points(fs_path)

# Load STL mesh
stl_mesh = mesh.Mesh.from_file(pred_path)

# Number of nearest neighbor faces per point
num_faces_per_point = 30  # Default value, adjust as needed

# Find nearest neighbor faces in the STL mesh for medial wall points
face_indices_to_remove = nearest_neighbor_faces(stl_mesh, medial_wall_points, num_faces_per_point)

# Remove the identified faces from the STL mesh
modified_stl_mesh = remove_faces(stl_mesh, face_indices_to_remove)

# Save the modified STL mesh
modified_stl_mesh.save(output_path)
print(f"Modified STL mesh saved to {output_path}")
