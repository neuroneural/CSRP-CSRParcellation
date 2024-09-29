import os
import re
import nibabel as nib
import trimesh
import numpy as np
import pandas as pd
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
from model.csrvcv2 import CSRVCV2
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

def load_model_and_weights(model_file, device, config):
    """
    Load the CSRVCV2 model and its weights.
    """
    C = config.dim_h
    K = config.kernel_size
    Q = config.n_scale
    use_gcn = config.gnn == 'gcn'
    model = CSRVCV2(dim_h=C, kernel_size=K, n_scale=Q, sf=config.sf, gnn_layers=config.gnn_layers,
                    use_gcn=use_gcn, gat_heads=config.gat_heads, num_classes=config.num_classes).to(device)
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
    model.eval()
    return model

if __name__ == '__main__':

    # ------ Load configuration ------
    config = load_config()
    test_type = config.test_type           # e.g., 'init', 'pred', 'eval'
    data_dir = config.data_dir             # Directory of datasets
    model_dir = config.model_dir           # Directory of pretrained models
    init_dir = config.init_dir             # Directory for saving initial surfaces
    result_dir = config.result_dir         # Directory for saving predicted surfaces
    data_name = args.data_name             # e.g., 'hcp', 'adni', 'dhcp'
    surf_hemi = args.hemisphere            # 'lh', 'rh'
    device = config.device                 # e.g., 'cuda' or 'cpu'
    tag = config.tag                       # Identity of the experiment

    C = config.dim_h                       # Hidden dimension of features
    K = config.kernel_size                 # Kernel / cube size
    Q = config.n_scale                     # Multi-scale input

    step_size = config.step_size           # Step size of integration
    solver = args.solver                   # ODE solver (e.g., 'euler', 'rk4')
    n_inflate = config.n_inflate           # Inflation iterations
    rho = config.rho                       # Inflation scale

    # ------ Load model-specific parameters from arguments ------
    model_file_path = args.model_file_path  # Full path to the model .pt file
    result_dir = args.result_dir      # Base name for saving outputs

    # ------ Load the segmentation network ------
    segnet = Unet(c_in=1, c_out=3).to(device)
    segnet.load_state_dict(torch.load(os.path.join(model_dir, config.seg_model_file), map_location=torch.device(device)))
    segnet.eval()

    # ------ Load the CSRVCV2 model ------
    if not os.path.exists(model_file_path):
        print(f"Model file {model_file_path} not found. Exiting.")
        exit(1)

    # Create a subdirectory for saving results based on result_dir
    model_subdir = os.path.join(result_dir, result_dir)
    os.makedirs(model_subdir, exist_ok=True)

    print(f"Processing model: {model_file_path}")
    print(f"Saving results to: {model_subdir}")

    # Load the CSRVCV2 model
    csrvcv2_model = load_model_and_weights(model_file_path, device, config)

    # ------ Prepare test data ------
    testset = SegDataset(config=config, data_usage='test')
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    for batch_idx, data in enumerate(testloader):
        volume_in, seg_gt, subid, _aff = data
        subid = str(subid[0])
        volume_in = volume_in.to(device)

        # Initialize BrainDatasets for WM and GM to obtain ctab
        config_wm = copy.deepcopy(config)
        config_wm.surf_type = 'wm'
        config_gm = copy.deepcopy(config)
        config_gm.surf_type = 'gm'
        brain_dataset_wm = BrainDataset(config_wm, data_usage='test', affCtab=True)
        brain_dataset_gm = BrainDataset(config_gm, data_usage='test', affCtab=True)

        # Ensure the index corresponds to the current batch
        try:
            brain_arr_wm, v_in_wm, v_gt_wm, f_in_wm, f_gt_wm, labels_wm, aff_wm, ctab_wm = brain_dataset_wm[batch_idx]
            brain_arr_gm, v_in_gm, v_gt_gm, f_in_gm, f_gt_gm, labels_gm, aff_gm, ctab_gm = brain_dataset_gm[batch_idx]
        except IndexError:
            print(f"BrainDataset index {batch_idx} out of range. Skipping subject {subid}.")
            continue

        # ------ Predict segmentation -------
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
        if test_type == 'init':
            init_surface_path = os.path.join(init_dir, f'init_{data_name}_{surf_hemi}_{subid}.obj')
            mesh_init = trimesh.Trimesh(v_in, f_in)
            mesh_init.export(init_surface_path)
            print(f"Saved initial surface for {subid} in {init_surface_path}")

        # ------ Predict the surface using the model -------
        with torch.no_grad():
            v_in_tensor = torch.Tensor(v_in).unsqueeze(0).to(device)
            f_in_tensor = torch.LongTensor(f_in).unsqueeze(0).to(device)
            csrvcv2_model.set_data(v_in_tensor, volume_in, f=f_in_tensor)
            T = torch.Tensor([0, 1]).to(device)
            v_pred = odeint(csrvcv2_model, v_in_tensor, T, method=solver, options=dict(step_size=step_size))[-1]

            # Re-set data with the latest prediction for classification
            csrvcv2_model.set_data(v_pred, volume_in, f=f_in_tensor)
            _ = csrvcv2_model(T, v_pred)  # Forward pass to get classification logits

            # Get classification logits if applicable
            if config.num_classes > 1:
                class_logits = csrvcv2_model.get_class_logits()
                # Add LogSoftmax
                log_softmax = nn.LogSoftmax(dim=1)
                class_probs = log_softmax(class_logits)
                class_pred = torch.argmax(class_probs, dim=1).cpu().numpy()
            else:
                class_pred = np.zeros(v_pred.shape[1], dtype=np.int32)

            # Convert to numpy arrays
            v_pred_np = v_pred[0].cpu().numpy()
            f_pred_np = f_in_tensor[0].cpu().numpy()

        # ------ Save predicted surfaces and annotations -------
        # Determine surface type for naming and ctab
        surf_type = surf_hemi  # 'wm' or 'gm'
        if surf_type == 'wm':
            pred_surface_basename = f'{data_name}_{surf_hemi}_{subid}_wm_pred'
            ctab = ctab_wm
        elif surf_type == 'gm':
            pred_surface_basename = f'{data_name}_{surf_hemi}_{subid}_gm_pred'
            ctab = ctab_gm
        else:
            print(f"Unknown surface type {surf_type} for subject {subid}. Skipping.")
            continue

        # Define the save path
        pred_surface_path = os.path.join(model_subdir, pred_surface_basename)

        # Save the predicted surface with annotations
        try:
            save_mesh_with_annotations(v_pred_np, f_pred_np, class_pred, ctab, pred_surface_path, data_name)
            print(f"Saved predicted surface for {subid} in {pred_surface_path}")
        except Exception as e:
            print(f"Error saving mesh for subject {subid}: {e}. Skipping.")
            continue

    print("Processing completed.")
