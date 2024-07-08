import os
import csv
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
import vtk
from multiprocessing import Lock
import sys
# Turn off VTK logging verbosity
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
import pyvista as pv
# Function to compute Chamfer Distance using KD-Tree
def compute_chamfer_distance(mesh1, mesh2):
    points1 = mesh1['vertices']
    points2 = mesh2['vertices']
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    dist1, _ = tree1.query(points2, k=1)
    dist2, _ = tree2.query(points1, k=1)
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    return chamfer_dist

# Function to compute Hausdorff Distance using KD-Tree
def compute_hausdorff_distance(mesh1, mesh2):
    points1 = mesh1['vertices']
    points2 = mesh2['vertices']
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
    return mesh['vertices'][mesh['faces']]

def mesh2tricenters(mesh, triangles=None):
    if triangles is None:
        triangles = mesh2triangles(mesh)
    centers = np.mean(triangles, axis=1)
    return centers

def count_self_collisions(mesh, k=5):
    faces = mesh['faces']
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
                                        mesh['vertices'][np.sort(np.unique(faces.flatten()))],
                                        faces)
        if collision:
            collision_count += 1

    return collision_count

def detachedtriangles(mesh, triangle_id, other_ids):
    mask = np.any(np.isin(mesh['faces'][other_ids], mesh['faces'][triangle_id]), axis=1)
    faces = mesh['faces'][other_ids][~mask]
    return faces

# Adjusted function to ensure labels match vertex counts
def load_freesurfer_surface(surface_path):
    coords, faces = nib.freesurfer.read_geometry(surface_path)
    mesh = {'vertices': coords, 'faces': faces}
    return mesh

# Function to write results to CSV
def write_to_csv(file_path, lock, data):
    with lock:
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Framework', 'Subject ID','surf_type','Hemisphere', 'Metric', 'Label', 'Score', 'Total Triangles'])
            writer.writerow(data)

# Process a subject and compute metrics
def process_subject(subj_id, csv_file_path, lock, framework_name, subject_base_path, gt_base_path, output_dir):
    # Paths to the required data
    print('subject_base_path',subject_base_path)
    print('subj_id',subj_id)
    hemispheres = ['lh', 'rh']
    for surf_type in ['pial', 'white']:
        for hemi in hemispheres:
            # Paths for BA and CA surfaces
            try:
                ba_surf_path = os.path.join(subject_base_path, f'csrf_{subj_id}_BA_{hemi}_{surf_type}')
                print('ba_surf_path',ba_surf_path)
                ca_surf_path = os.path.join(subject_base_path, f'csrf_{subj_id}_CA_{hemi}_{surf_type}')
                ca_mwrm_surf_path = os.path.join(subject_base_path, f'csrf_{subj_id}_CA_mwrm_{hemi}_{surf_type}')
                
                ba_surf = load_freesurfer_surface(ba_surf_path)
                ca_surf = load_freesurfer_surface(ca_surf_path)
                ca_mwrm_surf = load_freesurfer_surface(ca_mwrm_surf_path)
            except ValueError as e:
                print(f"Error loading surfaces for {hemi} hemisphere: {e}")
                continue

            # Compute Chamfer and Hausdorff distances for BA and CA surfaces
            chamfer_dist = compute_chamfer_distance(ba_surf, ca_surf)
            hausdorff_dist = compute_hausdorff_distance(ba_surf, ca_surf)
            
            # Compute self-intersections for CA_mwrm surfaces
            #self_collision_count = count_self_collisions(ca_mwrm_surf, k=30)
            self_collision_count = -1
            total_triangles = len(ca_mwrm_surf['faces'])

            # Write results to CSV
            write_to_csv(csv_file_path, lock, [framework_name, subj_id, surf_type, hemi, 'Chamfer Distance', '', chamfer_dist, ''])
            write_to_csv(csv_file_path, lock, [framework_name, subj_id, surf_type, hemi, 'Hausdorff Distance', '', hausdorff_dist, ''])
            write_to_csv(csv_file_path, lock, [framework_name, subj_id, surf_type, hemi, 'Self-Intersections (SIF)', '', self_collision_count, total_triangles])

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python script.py <subj_id> <csv_file_path> <framework_name> <subject_base_path>")
        sys.exit(1)
    
    subj_id = sys.argv[1]
    csv_file_path = sys.argv[2]
    framework_name = sys.argv[3]
    gt_base_path = sys.argv[4]
    subject_base_path = sys.argv[5]
    output_dir = 'result/'

    print('subj_id', 'csv_file_path', 'framework_name', 'subject_base_path', "output_dir")
    print(subj_id, csv_file_path, framework_name, subject_base_path, output_dir)
    # Initialize a lock for parallel-safe writing
    lock = Lock()
    
    process_subject(subj_id, csv_file_path, lock, framework_name, subject_base_path, gt_base_path, output_dir)
