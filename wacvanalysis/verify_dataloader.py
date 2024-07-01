import torch
from torch.utils.data import DataLoader
import trimesh
import nibabel as nib
import os
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataloader import BrainDataset
from data.preprocess import process_surface, process_surface_inverse
from config import load_config
import torch.multiprocessing as mp
import numpy as np

def invert_affine(affine):
    # Compute the inverse of the affine matrix
    inverse_affine = np.linalg.inv(affine)
    print(f"Inverse affine matrix:\n{inverse_affine}")
    return inverse_affine

def apply_affine(affine, coordinates):
    # Apply affine transformation to the coordinates
    homogenous_coordinates = np.hstack([coordinates, np.ones((coordinates.shape[0], 1))])
    transformed_coordinates = homogenous_coordinates.dot(affine.T)[:, :3]
    return transformed_coordinates

def save_mesh_as_stl(verts, faces, filename):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)

def save_mesh_as_freesurfer(verts, faces, filename):
    nib.freesurfer.io.write_geometry(filename, verts, faces)

def save_test_meshes(config):
    # Load configuration
    result_dir = config.result_dir#need to set
    device = config.device
    data_name = config.data_name#need to set
    
    # Load test dataset
    logging.info("Load dataset ...")
    testset = BrainDataset(config, 'test')
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    # Iterate over the test set
    logging.info("Iterating over the test set ...")
    import re
    for idx, data in enumerate(testloader):
        volume_in, v_in, v_gt, f_in, f_gt,aff, subj_id = data
        subj_id = re.sub(r'\D', '', str(subj_id))

        print('subj_id',subj_id)
        
        v_in = v_in.to(device).cpu().numpy()
        f_in = f_in.to(device).cpu().numpy()
        v_gt = v_gt.to(device).cpu().numpy()
        f_gt = f_gt.to(device).cpu().numpy()
        
        
        # Process surfaces
        v_in, f_in = process_surface_inverse(v_in[0], f_in[0], data_name)
        v_gt, f_gt = process_surface_inverse(v_gt[0], f_gt[0], data_name)

        # Compute the inverse of the affine matrix
        inverse_affine = invert_affine(aff)
        
        # Apply the inverse affine transformation to the vertices
        v_gt_transformed = apply_affine(aff, v_gt)
        
        #investigate whether the inverse is actually an inverse. 
        gt_fs_filename2 = os.path.join(result_dir, f'v2before{config.surf_hemi}.{config.surf_type}.gt_mesh_{subj_id}.surf')
        save_mesh_as_freesurfer(v_gt, f_gt, gt_fs_filename2)
        
        v_gt2, f_gt2 = process_surface(v_gt, f_gt, data_name)
        v_gt2, f_gt2 = process_surface_inverse(v_gt2, f_gt2, data_name)
        gt_fs_filename2 = os.path.join(result_dir, f'v2after{config.surf_hemi}.{config.surf_type}.gt_mesh_{subj_id}.surf')
        save_mesh_as_freesurfer(v_gt2, f_gt, gt_fs_filename2)
        
        
        
        # Save meshes as STL files
        input_stl_filename = os.path.join(result_dir, f'{config.surf_hemi}.{config.surf_type}.input_mesh_{subj_id}.stl')
        gt_stl_filename = os.path.join(result_dir, f'{config.surf_hemi}.{config.surf_type}.gt_mesh_{subj_id}.stl')
        
        save_mesh_as_stl(v_in, f_in, input_stl_filename)
        save_mesh_as_stl(v_gt, f_gt, gt_stl_filename)
        
        logging.info(f"Saved input mesh to {input_stl_filename}")
        logging.info(f"Saved ground truth mesh to {gt_stl_filename}")

        # Save meshes as FreeSurfer surfaces
        input_fs_filename = os.path.join(result_dir, f'{config.surf_hemi}.{config.surf_type}.input_mesh_{subj_id}.surf')
        gt_fs_filename = os.path.join(result_dir, f'{config.surf_hemi}.{config.surf_type}.gt_mesh_{subj_id}.surf')
        gt_fs_filename3 = os.path.join(result_dir, f'v3{config.surf_hemi}.{config.surf_type}.gt_mesh_{subj_id}.surf')
        
        save_mesh_as_freesurfer(v_in, f_in, input_fs_filename)
        save_mesh_as_freesurfer(v_gt, f_gt, gt_fs_filename)
        save_mesh_as_freesurfer(v_gt_transformed, f_gt, gt_fs_filename3)
        
        logging.info(f"Saved input mesh to {input_fs_filename}")
        logging.info(f"Saved ground truth mesh to {gt_fs_filename}")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    config = load_config()
    save_test_meshes(config)
