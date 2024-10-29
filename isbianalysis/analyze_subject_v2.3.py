import os
import csv
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
import argparse
import re
import pyvista as pv
import sys
from multiprocessing import Lock

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

def load_freesurfer_surface_and_labels(surface_path,annot_path = None):
    assert surface_path is not None, "surface_path is None"
    try:
        if annot_path is None:
            
            pred_surf_dir = os.path.dirname(surface_path)
            
            # Load the annotation file with the same basename but .annot extension
            # annot_path = os.path.splitext(surface_path)[0] + '.annot'
            pred_files = [f for f in os.listdir(pred_surf_dir) if f.endswith('.annot') and (subj_id in f)]
            print(f'Prediction Files: {pred_files}')
            for pred_file in pred_files:
                print(f'Processing File: {pred_file}')
                if 'gm' in surface_path:
                    pattern = r'hcp_[lr]h_(\d{6})_gnnlayers(\d+)_gm_pred(?:_epochdef(\d+))?(?:_epochcls(\d+))?\.annot'
                else:
                    pattern = r'hcp_[lr]h_(\d{6})_gnnlayers(\d+)_wm_pred(?:_epochdef(\d+))?(?:_epochcls(\d+))?\.annot'
                match = re.search(pattern, pred_file)
                if not match:
                    print(f"Filename pattern does not match expected format: {pred_file}. Skipping.")
                    continue
                
                gnn_layers = match.group(2)
                epoch_def = match.group(3) if match.group(3) is not None else None
                epoch_cls = match.group(4) if match.group(4) is not None else None
                annot_path = os.path.join(pred_surf_dir,pred_file)
                break
            assert annot_path is not None, 'update annot_path logic'
            
        print('reading surface_path',surface_path)
        coords, faces = nib.freesurfer.read_geometry(surface_path)
        faces = faces.astype(np.int64)  # Ensure faces are of type int64
        mesh = {'vertices': coords, 'faces': faces}

        print('reading annot_path',annot_path)
        
        if annot_path != 'cortexode':
            labels, ctab, names = nib.freesurfer.read_annot(annot_path)
        else:
            labels = None
        return mesh, labels
        
    except Exception as e:
        print(f"Error loading surface or annotation file {surface_path}: {e}")
        raise

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

def write_to_csv(file_path, lock, data):
    file_exists = os.path.isfile(file_path)
    print('writetocsv',file_path)
    assert lock is not None, "Error: lock is None"
    try:
        with lock:
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Framework','Condition', 'Subject ID', 'surf_type', 'Hemisphere', 'GNNLayers', 'Epoch', 'Metric', 'Label', 'Score', 'Total Triangles'])
                writer.writerow(data)
    except Exception as e:
        print(f"Error writing to CSV {file_path}: {e}")

def extract_subject_id(filename):
    match = re.search(r'hcp_[lr]h_(\d{6})_', filename)
    if match:
        return match.group(1)
    else:
        return None

def compute_dice(pred, target, num_classes=36, exclude_classes=[-1,4]):
    dice_scores = []
    for i in range(num_classes):
        if i in exclude_classes:
            continue
        
        pred_i = (pred == i)
        target_i = (target == i)
        
        intersection = np.sum(pred_i & target_i)
        union = np.sum(pred_i) + np.sum(target_i)
        
        if union == 0:
            dice_score = 1.0
        else:
            dice_score = 2. * intersection / union
        dice_scores.append(dice_score)
        
    return np.mean(dice_scores)


def map_labels(labels_source, vertices_source, vertices_target):
    tree = cKDTree(vertices_source)
    distances, indices = tree.query(vertices_target, k=1)
    mapped_labels = labels_source[indices]
    return mapped_labels

def process_subject(subj_id, csv_file_path, lock, framework_name, pred_base_path, condition, output_dir,gt_subject_base_path):
    print(f'Processing Subject: {subj_id}')
    hemispheres = ['lh', 'rh']
    surf_types = ['gm', 'wm']  # 'gm' for Grey Matter, 'wm' for White Matter

    freesurfer_subject_path = os.path.join(gt_subject_base_path, subj_id)
    for hemi in hemispheres:
        print(f'Hemisphere: {hemi}')
        for surf_type in surf_types:
            print(f'Surface Type: {surf_type}')
            if framework_name.lower() in ['csrp','cortexode']:
                if surf_type == 'gm':
                    gt_surf_path = os.path.join(freesurfer_subject_path, 'surf', f'{hemi}.pial.deformed')
                    gt_annot_path = os.path.join(freesurfer_subject_path, 'label', f'{hemi}.aparc.DKTatlas40.annot')
                elif surf_type == 'wm':
                    gt_surf_path = os.path.join(freesurfer_subject_path, 'surf', f'{hemi}.white.deformed')
                    gt_annot_path = os.path.join(freesurfer_subject_path, 'label', f'{hemi}.aparc.DKTatlas40.annot')
                else: 
                    assert False, 'bad surf_type'
                
                pred_surf_dir = os.path.join(pred_base_path,condition,'test', surf_type, hemi)
                
                if not os.path.isdir(pred_surf_dir):
                    print(f"Prediction directory does not exist: {pred_surf_dir}. Skipping.")
                    continue
                
                pred_files = [f for f in os.listdir(pred_surf_dir) if f.endswith('.surf') and (subj_id in f)]
                print(f'Prediction Files: {pred_files}')
                for pred_file in pred_files:
                    print(f'Processing File: {pred_file}')
                    if surf_type == 'gm':
                        pattern = r'hcp_[lr]h_(\d{6})_gnnlayers(\d+)_gm_pred(?:_epochdef(\d+))?\.surf'
                    else:
                        pattern = r'hcp_[lr]h_(\d{6})_gnnlayers(\d+)_wm_pred(?:_epochdef(\d+))?\.surf'
                    match = re.search(pattern, pred_file)
                    if not match:
                        print(f"Filename pattern does not match expected format: {pred_file}. Skipping.")
                        continue
                    
                    gnn_layers = match.group(2)
                    epoch = match.group(3) if match.group(3) is not None else None

                    if epoch is None:#TODO: this could be more general. 
                        if framework_name == 'cortexode':
                            epoch = '90'
                        else:
                            epoch = 'unknown'

                    pred_surf_path = os.path.join(pred_surf_dir, pred_file)
                    print(f'Prediction Surface Path: {pred_surf_path}')
                    annot_path = None
            elif framework_name.lower() in ['fastsurfer']:
                if surf_type == 'gm':
                    gt_surf_path = os.path.join(freesurfer_subject_path, 'surf', f'{hemi}.pial')
                    gt_annot_path = os.path.join(freesurfer_subject_path, 'label', f'{hemi}.aparc.DKTatlas40.annot')
                elif surf_type == 'wm':
                    gt_surf_path = os.path.join(freesurfer_subject_path, 'surf', f'{hemi}.white')
                    gt_annot_path = os.path.join(freesurfer_subject_path, 'label', f'{hemi}.aparc.DKTatlas40.annot')
                else: 
                    assert False, 'bad surf_type'
                
                fastsurfer_subject_path = os.path.join(base_path, subj_id, subj_id)
                if surf_type == 'gm':
                    pred_surf_path = os.path.join(fastsurfer_subject_path, 'surf', f'{hemi}.pial')
                elif surf_type == 'wm':
                    pred_surf_path = os.path.join(fastsurfer_subject_path, 'surf', f'{hemi}.white')
                else:
                    assert False, "wrong surf_type"
                annot_path = os.path.join(fastsurfer_subject_path, 'label', f'{hemi}.aparc.DKTatlas.mapped.annot')
                epoch = None
                gnn_layers = 0
            try:
                if framework_name.lower() == 'cortexode':
                    annot_path = 'cortexode' #TODO: UPDATE FOR DICE FROM CORTEX ODE 
                pred_mesh, pred_labels = load_freesurfer_surface_and_labels(pred_surf_path, annot_path)
                gt_mesh, gt_labels = load_freesurfer_surface_and_labels(gt_surf_path,gt_annot_path)
            except Exception as e:
                print(f"Error loading meshes or labels for {subj_id}, {hemi}, {surf_type}: {e}")
                continue
            
            print('max pred',np.max(pred_mesh['vertices']))
            print('min pred',np.min(pred_mesh['vertices']))
            print('max gt',np.max(gt_mesh['vertices']))
            print('min gt',np.min(gt_mesh['vertices']))
            
            chamfer_dist = compute_chamfer_distance(pred_mesh, gt_mesh)
            hausdorff_dist = compute_hausdorff_distance(pred_mesh, gt_mesh)

            # Map labels if necessary
            if pred_mesh['vertices'].shape[0] != gt_mesh['vertices'].shape[0] and pred_labels is not None:
                pred_labels = map_labels(pred_labels, pred_mesh['vertices'], gt_mesh['vertices'])


            if pred_labels is not None:
                dice_scores = compute_dice(pred_labels, gt_labels)

            self_collision_count = count_self_collisions(pred_mesh, k=30)
            total_triangles = len(pred_mesh['faces'])

            #TODO: UPDATE EPOCHS FOR MORE HYPERPARAMETER VARIATIONS
            data_chamfer = [framework_name, condition, subj_id, surf_type, hemi, gnn_layers, epoch, 'Chamfer Distance', '', chamfer_dist, '']
            data_hausdorff = [framework_name, condition, subj_id, surf_type, hemi, gnn_layers, epoch, 'Hausdorff Distance', '', hausdorff_dist, '']
            if pred_labels is not None:
                data_dice = [framework_name, condition, subj_id, surf_type, hemi, gnn_layers, epoch, 'Macro Dice', '', dice_scores, '']
            data_self_intersect = [framework_name, condition, subj_id, surf_type, hemi, gnn_layers, epoch, 'Self-Intersections (SIF)', '', self_collision_count, total_triangles]

            write_to_csv(csv_file_path, lock, data_chamfer)
            write_to_csv(csv_file_path, lock, data_hausdorff)
            write_to_csv(csv_file_path, lock, data_self_intersect)
            if pred_labels is not None:
                write_to_csv(csv_file_path,lock, data_dice)
            
            

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process subject arguments")

    parser.add_argument('--subj_id', type=str, required=True, help='Subject identifier (e.g., 201818)')
    parser.add_argument('--csv_file_path', type=str, required=True, help='Path to the CSV file where results will be appended')
    parser.add_argument('--framework_name', type=str, required=True, help='Name of the framework/model (e.g., csrp)')
    parser.add_argument('--base_path', type=str, required=True, help='Base path to the test directory generated by generateISBITestSurfaces.py')
    parser.add_argument('--gt_subject_base_path', type=str, required=True, help='Base path to the test directory generated by generateISBITestSurfaces.py')
    parser.add_argument('--condition', type=str, default='NA', help='[a,b] represents the training methodology')
    
    args = parser.parse_args()

    subj_id = args.subj_id.strip()
    csv_file_path = args.csv_file_path.strip()
    framework_name = args.framework_name.strip()
    base_path = args.base_path.strip()
    condition = args.condition.strip()
    output_dir = 'result/'

    print(f'Subject ID: {subj_id}')
    print(f'CSV File Path: {csv_file_path}')
    print(f'Framework Name: {framework_name}')
    print(f'Base Path: {base_path}')
    print(f'Output Directory: {output_dir}')

    lock = Lock()
    process_subject(subj_id, csv_file_path, lock, framework_name, base_path, condition, output_dir,args.gt_subject_base_path.strip())
