import os
import re
import nibabel as nib
import trimesh
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_cdt as cdt
from skimage.measure import marching_cubes
from skimage.measure import label as compute_cc
from skimage.filters import gaussian


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import argparse
from data.preprocess import process_volume, process_surface, process_surface_inverse
from data.datautil import decode_names
from util.mesh import laplacian_smooth, compute_normal, compute_mesh_distance, check_self_intersect
from util.tca import topology
from model.net import Unet
from model.csrvcv3 import CSRVCV3
from config import load_config
from data.csrandvcdataloader import SegDataset, BrainDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import copy

# Initialize topology correction
topo_correct = topology()

def seg2surf(seg, data_name='hcp', sigma=0.5, alpha=16, level=0.8, n_smooth=2, device='cpu'):
    """
    Extract the surface based on the segmentation.
    """
    # ------ Connected Components Checking ------
    cc, nc = compute_cc(seg, connectivity=2, return_num=True)
    if nc == 0:
        raise ValueError("No connected components found in segmentation.")
    cc_id = 1 + np.argmax(np.array([np.count_nonzero(cc == i) for i in range(1, nc + 1)]))
    seg = (cc == cc_id).astype(np.float64)

    # ------ Generate Signed Distance Function ------
    sdf = -cdt(seg) + cdt(1 - seg)
    sdf = sdf.astype(float)
    sdf = gaussian(sdf, sigma=sigma)

    # ------ Topology Correction ------
    sdf_topo = topo_correct.apply(sdf, threshold=alpha)

    # ------ Marching Cubes ------
    v_mc, f_mc, _, _ = marching_cubes(-sdf_topo, level=-level, method='lorensen')
    v_mc = v_mc[:, [2, 1, 0]].copy()  # Reorder axes if necessary
    f_mc = f_mc.copy()
    D1, D2, D3 = sdf_topo.shape
    D = max(D1, D2, D3)
    v_mc = (2 * v_mc - [D3, D2, D1]) / D   # Rescale to [-1,1]

    # ------ Bias Correction ------
    if data_name == 'hcp':
        v_mc = v_mc + [0.0090, 0.0058, 0.0088]
    elif data_name == 'adni':
        v_mc = v_mc + [0.0090, 0.0000, 0.0095]

    # ------ Mesh Smoothing ------
    v_mc = torch.Tensor(v_mc).unsqueeze(0).to(device)
    f_mc = torch.LongTensor(f_mc).unsqueeze(0).to(device)
    for j in range(n_smooth):    # Smooth and inflate the mesh
        v_mc = laplacian_smooth(v_mc, f_mc, 'uniform', lambd=1)
    v_mc = v_mc[0].cpu().numpy()
    f_mc = f_mc[0].cpu().numpy()

    return v_mc, f_mc

def save_mesh_with_annotations(verts, faces, labels, ctab, save_path_fs, data_name='hcp'):
    """
    Save the mesh with annotations using nibabel.
    """
    verts = verts.squeeze()
    faces = faces.squeeze().astype(np.int32)
    verts, faces = process_surface_inverse(verts, faces, data_name)
    labels = labels.squeeze().astype(np.int32)
    if isinstance(ctab, torch.Tensor):
        ctab = ctab.numpy()
    ctab = ctab.astype(np.int32)
    assert ctab.shape[1] == 5, "ctab should have 5 columns for RGBA and region labels."
    nib.freesurfer.write_geometry(save_path_fs + '.surf', verts, faces)
    nib.freesurfer.write_annot(save_path_fs + '.annot', labels, ctab, decode_names(), fill_ctab=False)

def load_models_and_weights(device, config_wm,config_gm):
    """
    Load the CSRVCV3 model and its weights.
    """
    C = config_wm.dim_h
    K = config_wm.kernel_size
    Q = config_wm.n_scale
    use_gcn = config_wm.gnn == 'gcn'
    if config_wm.model_file_wm is not None:
        model_file_path_wm = os.path.join(config_wm.model_dir,config.model_file_wm) # Full path to the model .pt file
    model_file_path_gm = os.path.join(config_gm.model_dir,config.model_file_gm) # Full path to the model .pt file
    
    if config_wm.model_file_wm is not None:
        model_wm = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config_wm.sf, gnn_layers=config_wm.gnn_layers,
                        use_gcn=use_gcn, gat_heads=config_wm.gat_heads, num_classes=config_wm.num_classes).to(device)
        model_wm.load_state_dict(torch.load(model_file_path_wm, map_location=torch.device(device)))
        model_wm.eval()
    
    model_gm = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config_gm.sf, gnn_layers=config_gm.gnn_layers,
                    use_gcn=use_gcn, gat_heads=config_gm.gat_heads, num_classes=config_gm.num_classes).to(device)
    model_gm.load_state_dict(torch.load(model_file_path_gm, map_location=torch.device(device)))
    model_gm.eval()
    if config_wm.model_file_wm is not None:
        return model_wm,model_gm
    else:
        return None,model_gm

if __name__ == '__main__':

    # ------ Load configuration ------
    config = load_config()
    test_type = config.test_type           # e.g., 'init', 'pred', 'eval'
    data_dir = config.data_dir             # Directory of datasets
    model_dir = config.model_dir           # Directory of pretrained models
    init_dir = config.init_dir             # Directory for saving initial surfaces
    result_dir = config.result_dir         # Directory for saving predicted surfaces
    data_name = config.data_name             # e.g., 'hcp', 'adni', 'dhcp'
    surf_hemi = config.surf_hemi            # 'lh', 'rh'
    device = config.device                 # e.g., 'cuda' or 'cpu'
    tag = config.tag                       # Identity of the experiment

    C = config.dim_h                       # Hidden dimension of features
    K = config.kernel_size                 # Kernel / cube size
    Q = config.n_scale                     # Multi-scale input

    step_size = config.step_size           # Step size of integration
    solver = config.solver                   # ODE solver (e.g., 'euler', 'rk4')
    n_inflate = config.n_inflate           # Inflation iterations
    rho = config.rho                       # Inflation scale

    # ------ Load model-specific parameters from arguments ------
    if config.model_file_wm is not None:
        model_file_path_wm = os.path.join(config.model_dir,config.model_file_wm) # Full path to the model .pt file
    model_file_path_gm = os.path.join(config.model_dir,config.model_file_gm) # Full path to the model .pt file
    result_dir = config.result_dir      # Base name for saving outputs
    
    # ------ Load the segmentation network ------
    if config.model_file_wm is not None:
        segnet = Unet(c_in=1, c_out=3).to(device)
        segnet.load_state_dict(torch.load(os.path.join(config.model_dir, config.seg_model_file), map_location=torch.device(device)))
        segnet.eval()

        # ------ Load the CSRVCV3 model ------
        if not os.path.exists(model_file_path_wm):
            print(f"Model file {model_file_path_wm} not found. Exiting.")
            exit(1)
    
        print(f"Processing model: {model_file_path_wm}")
    
    if not os.path.exists(model_file_path_gm):
        print(f"Model file {model_file_path_gm} not found. Exiting.")
        exit(1)

    # Create a subdirectory for saving results based on result_dir
    #TODO: create more subfolders based on model_file without the pt, need to makedir, then 
    model_subdir = result_dir#os.path.join(result_dir, model_file without the pt)
    os.makedirs(model_subdir, exist_ok=True)

    print(f"Processing model: {model_file_path_gm}")
    print(f"Saving results to: {model_subdir}",'update for multi model')

    config_wm = copy.deepcopy(config)
    config_wm.surf_type = 'wm'
    config_gm = copy.deepcopy(config)
    config_gm.surf_type = 'gm'

    # Load the CSRVCV3 model
    csrvcv3_wm,csrvcv3_gm = load_models_and_weights(device, config_wm,config_gm)

    # ------ Prepare test data ------
    testset = SegDataset(config=config, data_usage='test')#need to update for gm/wm later
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    brain_dataset_wm = BrainDataset(config_wm, data_usage='test', affCtab=True)
    brain_dataset_gm = BrainDataset(config_gm, data_usage='test', affCtab=True)
    
    T = torch.Tensor([0,1]).to(device)
    for batch_idx, data in enumerate(testloader):
        volume_in, seg_gt, subid, _aff = data
        
        subid = str(subid[0])
        volume_in = volume_in.to(device)

        # Ensure the index corresponds to the current batch
        try:
            brain_arr_wm, v_in_wm, v_gt_wm, f_in_wm, f_gt_wm, labels_wm, aff_wm, ctab_wm, sub_id_wm = brain_dataset_wm[batch_idx]
            brain_arr_gm, v_in_gm, v_gt_gm, f_in_gm, f_gt_gm, labels_gm, aff_gm, ctab_gm, sub_id_gm = brain_dataset_gm[batch_idx]
            assert subid == sub_id_wm
            assert subid == sub_id_gm
            
        except IndexError:
            print(f"BrainDataset index {batch_idx} out of range. Skipping subject {subid}.")
            continue

        # ------ Predict segmentation -------
        if config.model_file_wm is not None:
            with torch.no_grad():
                seg_out = segnet(volume_in)
                seg_pred = torch.argmax(seg_out, dim=1)[0]
                if surf_hemi == 'lh':
                    seg = (seg_pred == 1).cpu().numpy()  # lh
                elif surf_hemi == 'rh':
                    seg = (seg_pred == 2).cpu().numpy()  # rh
                else:
                    print(f"Unknown hemisphere {surf_hemi}. Skipping subject {subid}.")
                    continue

            # ------ Extract initial surface -------
            try:
                v_in, f_in = seg2surf(seg, data_name, sigma=0.5, alpha=16, level=0.8, n_smooth=2, device=device)
            except ValueError as e:
                print(f"Error in seg2surf for subject {subid}: {e}. Skipping.")
                continue

        # ------ Save initial surface if required -------
        # if test_type == 'init':
        #     init_surface_path = os.path.join(init_dir, f'init_{data_name}_{surf_hemi}_{subid}.obj')
        #     mesh_init = trimesh.Trimesh(v_in, f_in)
        #     mesh_init.export(init_surface_path)
        #     print(f"Saved initial surface for {subid} in {init_surface_path}")

        

        #we need to inflate and smooth for grey matter!
        # ------ Predict the surface using the model -------
        if test_type == 'pred' or test_type == 'eval':
            with torch.no_grad():
                if config.model_file_wm is not None:
                    v_in = torch.Tensor(v_in).unsqueeze(0).to(device)
                    f_in = torch.LongTensor(f_in).unsqueeze(0).to(device)
                    
                    # wm surface
                    csrvcv3_wm.set_data(v_in, volume_in,f_in)
                    v_wm_pred = odeint(csrvcv3_wm, v_in, t=T, method=solver,
                                    options=dict(step_size=step_size))[-1]
                    
                    csrvcv3_wm.set_data(v_wm_pred, volume_in, f=f_in)
                    _dx = csrvcv3_wm(T, v_wm_pred)  # Forward pass to get classification logits
                    class_logits_wm = csrvcv3_wm.get_class_logits()
                    # Add LogSoftmax
                    class_logits_wm = F.log_softmax(class_logits_wm, dim=1)
                    class_logits_wm = class_logits_wm.unsqueeze(0)
                    class_probs_wm = class_logits_wm.permute(0, 2, 1)
                    class_wm_pred = torch.argmax(class_probs_wm, dim=1).cpu().numpy()
                    v_gm_in = v_wm_pred.clone()
                else:
                    v_gm_in = v_gt_wm.unsqueeze(0).to(device)
                    f_in = f_in_wm.unsqueeze(0).to(device)
                    
                # inflate and smooth
                for i in range(2):
                    v_gm_in = laplacian_smooth(v_gm_in, f_in, lambd=1.0)
                    n_in = compute_normal(v_gm_in, f_in)
                    v_gm_in += 0.002 * n_in

                # pial surface
                csrvcv3_gm.set_data(v_gm_in, volume_in,f_in)
                v_gm_pred = odeint(csrvcv3_gm, v_gm_in, t=T, method=solver,
                                   options=dict(step_size=step_size/2))[-1]  # divided by 2 to reduce SIFs

                csrvcv3_gm.set_data(v_gm_pred, volume_in, f=f_in)
                _dx = csrvcv3_gm(T, v_gm_pred)  # Forward pass to get classification logits

                class_logits_gm = csrvcv3_gm.get_class_logits()
                # Add LogSoftmax
                class_logits_gm = F.log_softmax(class_logits_gm, dim=1)
                class_logits_gm = class_logits_gm.unsqueeze(0)
                class_probs_gm = class_logits_gm.permute(0, 2, 1)
                class_gm_pred = torch.argmax(class_probs_gm, dim=1).cpu().numpy()
                
                if config.model_file_wm is not None:
                    v_wm_pred = v_wm_pred[0].cpu().numpy()
                    f_wm_pred = f_in[0].cpu().numpy()
                v_gm_pred = v_gm_pred[0].cpu().numpy()
                f_gm_pred = f_in[0].cpu().numpy()
                # map the surface coordinate from [-1,1] to its original space
                if config.model_file_wm is not None:
                    v_wm_pred, f_wm_pred = process_surface_inverse(v_wm_pred, f_wm_pred, data_name)
                v_gm_pred, f_gm_pred = process_surface_inverse(v_gm_pred, f_gm_pred, data_name)

                # Re-set data with the latest prediction for classification
                # Get classification logits if applicable
                # ------ Save predicted surfaces and annotations -------
                # Determine surface type for naming and ctab
                if config.model_file_wm is not None:
                    pred_surface_basename_wm = f'{data_name}_{surf_hemi}_{subid}_wm_pred'
                pred_surface_basename_gm = f'{data_name}_{surf_hemi}_{subid}_gm_pred'

                # Define the save path
                if config.model_file_wm is not None:
                    pred_surface_path_wm = os.path.join(model_subdir, pred_surface_basename_wm)
                pred_surface_path_gm = os.path.join(model_subdir, pred_surface_basename_gm)

                # Save the predicted surface with annotations
                try:
                    #TODO: add ground truth saves
                    if config.model_file_wm is not None:
                        save_mesh_with_annotations(v_wm_pred, f_wm_pred, class_wm_pred, ctab_wm, pred_surface_path_wm, data_name)
                        print(f"Saved predicted surface for {subid} in {pred_surface_path_wm}")
                    save_mesh_with_annotations(v_gm_pred, f_gm_pred, class_gm_pred, ctab_gm, pred_surface_path_gm, data_name)
                    print(f"Saved predicted surface for {subid} in {pred_surface_path_gm}")
                    
                except Exception as e:
                    print(f"Error saving mesh for subject {subid}: {e}. Skipping.")
                    continue

    print("Processing completed.")
