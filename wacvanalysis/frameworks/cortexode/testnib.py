import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

def compute_chamfer_distance(pc1, pc2):
    """Compute the Chamfer distance between two point clouds using cKDTree."""
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    distances1, _ = tree1.query(pc2, k=1)
    distances2, _ = tree2.query(pc1, k=1)
    
    chamfer_dist = np.mean(distances1) + np.mean(distances2)
    return chamfer_dist

# Paths to the original and test files
original_surface_path = '/data/users2/washbee/speedrun/cortexode-data-rp/test/201818/surf/lh.pial.deformed'
test_surface_path = '/data/users2/washbee/speedrun/cortexode-data-rp/test/201818/surf/lh.pial.deformed.test'

# Read the original surface file
original_vertices, original_faces = nib.freesurfer.io.read_geometry(original_surface_path)

# Write the surface to a new file
nib.freesurfer.io.write_geometry(test_surface_path, original_vertices, original_faces)

# Read the newly written surface file
test_vertices, test_faces = nib.freesurfer.io.read_geometry(test_surface_path)

# Compute the Chamfer distance
chamfer_dist = compute_chamfer_distance(original_vertices, test_vertices)
print(f'Chamfer distance between original and written vertices: {chamfer_dist}')
