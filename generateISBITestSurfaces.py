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
    
    assert not np.isnan(verts.max()), "The value is NaN"
    assert not np.isnan(verts.min()), "The value is NaN"
    
    # verts, faces = process_surface_inverse(verts, faces, data_name) # comment out because done elsewhere
    labels = labels.squeeze().astype(np.int32)
    if isinstance(ctab, torch.Tensor):
        ctab = ctab.numpy()
    ctab = ctab.astype(np.int32)
    assert ctab.shape[1] == 5, "ctab should have 5 columns for RGBA and region labels."
    nib.freesurfer.write_geometry(save_path_fs + '.surf', verts, faces)
    nib.freesurfer.write_annot(save_path_fs + '.annot', labels, ctab, decode_names(), fill_ctab=False)

def extract_rand_num_and_epoch_from_filename(filename):
    """
    Extract the random number and epoch from the model filename.
    """
    # Extract random number and epoch
    # Assuming the filename format includes '_{epoch}epochs_..._{rand_num}.pt'
    match = re.search(r'_(\d+)epochs_.*?_(\d+)(?:\.pt|_final\.pt)$', filename)
    if match:
        epoch = int(match.group(1)) + 1  # Add 1 to get start epoch (epoch + 1)
        rand_num = int(match.group(2))
        return rand_num, epoch
    else:
        # Handle case where filename ends with '_final.pt' without epoch number
        match = re.search(r'_(\d+)(?:\.pt|_final\.pt)$', filename)
        if match:
            rand_num = int(match.group(1))
            epoch = None  # Epoch not found
            return rand_num, epoch
        else:
            return None, None

def load_models_and_weights(device, config_wm, config_gm):
    """
    Load the CSRVCV3 models and their weights, extract rand_num and start_epoch.
    """
    print('device', device)
    C = config_wm.dim_h
    K = config_wm.kernel_size
    Q = config_wm.n_scale
    use_gcn = config_wm.gnn == 'gcn'

    models = {}
    # Infer condition based on the number of models passed
    # Check if separate deformation and classification models are specified
    wm_def_specified = hasattr(config_wm, 'model_file_wm_deformation') and config_wm.model_file_wm_deformation is not None
    wm_cls_specified = hasattr(config_wm, 'model_file_wm_classification') and config_wm.model_file_wm_classification is not None
    gm_def_specified = hasattr(config_gm, 'model_file_gm_deformation') and config_gm.model_file_gm_deformation is not None
    gm_cls_specified = hasattr(config_gm, 'model_file_gm_classification') and config_gm.model_file_gm_classification is not None

    if wm_def_specified and wm_cls_specified and gm_def_specified and gm_cls_specified:
        # Condition 'a'
        condition = 'a'
    elif hasattr(config_wm, 'model_file_wm') and config_wm.model_file_wm is not None and \
         hasattr(config_gm, 'model_file_gm') and config_gm.model_file_gm is not None:
        # Condition 'b'
        condition = 'b'
    else:
        print("Error: Cannot infer condition from the provided model files.")
        exit(1)

    print(f"Inferred condition: {condition}")

    if condition == 'a':
        # Load WM deformation model
        if wm_def_specified:
            model_file_wm_def = os.path.join(config_wm.model_dir, config_wm.model_file_wm_deformation)
            if not os.path.exists(model_file_wm_def):
                print(f"WM Deformation Model file {model_file_wm_def} not found. Exiting.")
                exit(1)
            rand_num_wm_def, start_epoch_wm_def = extract_rand_num_and_epoch_from_filename(config_wm.model_file_wm_deformation)
            print(f"WM Deformation Model Random Number: {rand_num_wm_def}")
            model_wm_def = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config_wm.sf, gnn_layers=config_wm.gnn_layers,
                                   use_gcn=use_gcn, gat_heads=config_wm.gat_heads, num_classes=config_wm.num_classes).to(device)
            checkpoint_wm_def = torch.load(model_file_wm_def, map_location=device)
            if 'model_state_dict' in checkpoint_wm_def:
                model_wm_def.load_state_dict(checkpoint_wm_def['model_state_dict'])
            else:
                model_wm_def.load_state_dict(checkpoint_wm_def)
            model_wm_def.eval()
            models['model_wm_def'] = model_wm_def
        else:
            models['model_wm_def'] = None

        # Load WM classification model
        if wm_cls_specified:
            model_file_wm_cls = os.path.join(config_wm.model_dir, config_wm.model_file_wm_classification)
            if not os.path.exists(model_file_wm_cls):
                print(f"WM Classification Model file {model_file_wm_cls} not found. Exiting.")
                exit(1)
            rand_num_wm_cls, start_epoch_wm_cls = extract_rand_num_and_epoch_from_filename(config_wm.model_file_wm_classification)
            print(f"WM Classification Model Random Number: {rand_num_wm_cls}")
            model_wm_cls = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config_wm.sf, gnn_layers=config_wm.gnn_layers,
                                   use_gcn=use_gcn, gat_heads=config_wm.gat_heads, num_classes=config_wm.num_classes).to(device)
            checkpoint_wm_cls = torch.load(model_file_wm_cls, map_location=device)
            if 'model_state_dict' in checkpoint_wm_cls:
                model_wm_cls.load_state_dict(checkpoint_wm_cls['model_state_dict'])
            else:
                model_wm_cls.load_state_dict(checkpoint_wm_cls)
            model_wm_cls.eval()
            models['model_wm_cls'] = model_wm_cls
        else:
            models['model_wm_cls'] = None

        # Load GM deformation model
        if gm_def_specified:
            model_file_gm_def = os.path.join(config_gm.model_dir, config_gm.model_file_gm_deformation)
            if not os.path.exists(model_file_gm_def):
                print(f"GM Deformation Model file {model_file_gm_def} not found. Exiting.")
                exit(1)
            rand_num_gm_def, start_epoch_gm_def = extract_rand_num_and_epoch_from_filename(config_gm.model_file_gm_deformation)
            print(f"GM Deformation Model Random Number: {rand_num_gm_def}")
            model_gm_def = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config_gm.sf, gnn_layers=config_gm.gnn_layers,
                                   use_gcn=use_gcn, gat_heads=config_gm.gat_heads, num_classes=config_gm.num_classes).to(device)
            checkpoint_gm_def = torch.load(model_file_gm_def, map_location=device)
            if 'model_state_dict' in checkpoint_gm_def:
                model_gm_def.load_state_dict(checkpoint_gm_def['model_state_dict'])
            else:
                model_gm_def.load_state_dict(checkpoint_gm_def)
            model_gm_def.eval()
            models['model_gm_def'] = model_gm_def
        else:
            models['model_gm_def'] = None

        # Load GM classification model
        if gm_cls_specified:
            model_file_gm_cls = os.path.join(config_gm.model_dir, config_gm.model_file_gm_classification)
            if not os.path.exists(model_file_gm_cls):
                print(f"GM Classification Model file {model_file_gm_cls} not found. Exiting.")
                exit(1)
            rand_num_gm_cls, start_epoch_gm_cls = extract_rand_num_and_epoch_from_filename(config_gm.model_file_gm_classification)
            print(f"GM Classification Model Random Number: {rand_num_gm_cls}")
            model_gm_cls = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config_gm.sf, gnn_layers=config_gm.gnn_layers,
                                   use_gcn=use_gcn, gat_heads=config_gm.gat_heads, num_classes=config_gm.num_classes).to(device)
            checkpoint_gm_cls = torch.load(model_file_gm_cls, map_location=device)
            if 'model_state_dict' in checkpoint_gm_cls:
                model_gm_cls.load_state_dict(checkpoint_gm_cls['model_state_dict'])
            else:
                model_gm_cls.load_state_dict(checkpoint_gm_cls)
            model_gm_cls.eval()
            models['model_gm_cls'] = model_gm_cls
        else:
            models['model_gm_cls'] = None

    elif condition == 'b':
        # Load combined WM model
        if hasattr(config_wm, 'model_file_wm') and config_wm.model_file_wm is not None:
            model_file_path_wm = os.path.join(config_wm.model_dir, config_wm.model_file_wm)
            if not os.path.exists(model_file_path_wm):
                print(f"WM Model file {model_file_path_wm} not found. Exiting.")
                exit(1)
            rand_num_wm, start_epoch_wm = extract_rand_num_and_epoch_from_filename(config_wm.model_file_wm)
            print(f"WM Model Random Number: {rand_num_wm}")
            model_wm = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config_wm.sf, gnn_layers=config_wm.gnn_layers,
                               use_gcn=use_gcn, gat_heads=config_wm.gat_heads, num_classes=config_wm.num_classes).to(device)
            checkpoint_wm = torch.load(model_file_path_wm, map_location=device)
            if 'model_state_dict' in checkpoint_wm:
                model_wm.load_state_dict(checkpoint_wm['model_state_dict'])
            else:
                model_wm.load_state_dict(checkpoint_wm)
            model_wm.eval()
            models['model_wm'] = model_wm
        else:
            models['model_wm'] = None

        # Load combined GM model
        if hasattr(config_gm, 'model_file_gm') and config_gm.model_file_gm is not None:
            model_file_path_gm = os.path.join(config_gm.model_dir, config_gm.model_file_gm)
            if not os.path.exists(model_file_path_gm):
                print(f"GM Model file {model_file_path_gm} not found. Exiting.")
                exit(1)
            rand_num_gm, start_epoch_gm = extract_rand_num_and_epoch_from_filename(config_gm.model_file_gm)
            print(f"GM Model Random Number: {rand_num_gm}")
            model_gm = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config_gm.sf, gnn_layers=config_gm.gnn_layers,
                               use_gcn=use_gcn, gat_heads=config_gm.gat_heads, num_classes=config_gm.num_classes).to(device)
            checkpoint_gm = torch.load(model_file_path_gm, map_location=device)
            if 'model_state_dict' in checkpoint_gm:
                model_gm.load_state_dict(checkpoint_gm['model_state_dict'])
            else:
                model_gm.load_state_dict(checkpoint_gm)
            model_gm.eval()
            models['model_gm'] = model_gm
        else:
            models['model_gm'] = None

    else:
        print(f"Unknown condition inferred. Exiting.")
        exit(1)

    return models, condition

if __name__ == '__main__':

    # ------ Load configuration ------
    config = load_config()
    test_type = config.test_type           # e.g., 'init', 'pred', 'eval'
    data_dir = config.data_dir             # Directory of datasets
    model_dir = config.model_dir           # Directory of pretrained models
    init_dir = config.init_dir             # Directory for saving initial surfaces
    result_dir = config.result_dir         # Directory for saving predicted surfaces
    data_name = config.data_name           # e.g., 'hcp', 'adni', 'dhcp'
    surf_hemi = config.surf_hemi           # 'lh', 'rh'
    device = config.device                 # e.g., 'cuda' or 'cpu'
    tag = config.tag                       # Identity of the experiment

    C = config.dim_h                       # Hidden dimension of features
    K = config.kernel_size                 # Kernel / cube size
    Q = config.n_scale                     # Multi-scale input

    step_size = config.step_size           # Step size of integration
    solver = config.solver                 # ODE solver (e.g., 'euler', 'rk4')
    n_inflate = config.n_inflate           # Inflation iterations
    rho = config.rho                       # Inflation scale

    # ------ Load model-specific parameters from arguments ------
    config_wm = copy.deepcopy(config)
    config_wm.surf_type = 'wm'
    config_gm = copy.deepcopy(config)
    config_gm.surf_type = 'gm'

    # ------ Load the segmentation network ------
    if config.seg_model_file is not None:
        segnet = Unet(c_in=1, c_out=3).to(device)
        segnet.load_state_dict(torch.load(os.path.join(config.model_dir, config.seg_model_file), map_location=device))
        segnet.eval()

    # Load the models and infer the condition
    models, condition = load_models_and_weights(device, config_wm, config_gm)

    # Get GAT layers for folder naming
    gat_layers = config.gnn_layers

    # Create result subdirectories based on condition and GAT layers
    folder_name = f"{condition}_{gat_layers}"
    result_subdir = os.path.join(result_dir, folder_name)
    os.makedirs(result_subdir, exist_ok=True)

    wm_hemi_dir = os.path.join(result_subdir, 'wm', surf_hemi)
    gm_hemi_dir = os.path.join(result_subdir, 'gm', surf_hemi)

    os.makedirs(wm_hemi_dir, exist_ok=True)
    os.makedirs(gm_hemi_dir, exist_ok=True)

    print(f"Saving WM results to: {wm_hemi_dir}")
    print(f"Saving GM results to: {gm_hemi_dir}")

    # ------ Prepare test data ------
    testset = SegDataset(config=config, data_usage='test')
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
        if config.seg_model_file is not None:
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

        # ------ Predict the surface using the model -------
        if test_type == 'pred' or test_type == 'eval':
            with torch.no_grad():
                # For white matter
                if config.seg_model_file is not None:
                    v_in = torch.Tensor(v_in).float().unsqueeze(0).to(device)
                    f_in = torch.LongTensor(f_in).unsqueeze(0).to(device)
                else:
                    v_in = v_in_wm.unsqueeze(0).to(device)
                    f_in = f_in_wm.unsqueeze(0).to(device)

                if condition == 'a':
                    # Deformation model
                    model_wm_def = models['model_wm_def']
                    model_wm_def.set_data(v_in, volume_in, f_in)
                    v_wm_pred = odeint(model_wm_def, v_in, t=T, method=solver,
                                       options=dict(step_size=step_size))[-1]

                    # Classification model
                    model_wm_cls = models['model_wm_cls']
                    model_wm_cls.set_data(v_wm_pred, volume_in, f=f_in)
                    _dx = model_wm_cls(T, v_wm_pred)
                    class_logits_wm = model_wm_cls.get_class_logits()
                    # Add LogSoftmax
                    class_logits_wm = F.log_softmax(class_logits_wm, dim=1)
                    class_logits_wm = class_logits_wm.unsqueeze(0)
                    class_probs_wm = class_logits_wm.permute(0, 2, 1)
                    class_wm_pred = torch.argmax(class_probs_wm, dim=1).cpu().numpy()

                elif condition == 'b':
                    # Combined model
                    model_wm = models['model_wm']
                    model_wm.set_data(v_in, volume_in, f_in)
                    v_wm_pred = odeint(model_wm, v_in, t=T, method=solver,
                                       options=dict(step_size=step_size))[-1]

                    model_wm.set_data(v_wm_pred, volume_in, f=f_in)
                    _dx = model_wm(T, v_wm_pred)
                    class_logits_wm = model_wm.get_class_logits()
                    # Add LogSoftmax
                    class_logits_wm = F.log_softmax(class_logits_wm, dim=1)
                    class_logits_wm = class_logits_wm.unsqueeze(0)
                    class_probs_wm = class_logits_wm.permute(0, 2, 1)
                    class_wm_pred = torch.argmax(class_probs_wm, dim=1).cpu().numpy()

                # Inflate and smooth for grey matter
                v_gm_in = v_wm_pred.clone()
                for i in range(2):
                    v_gm_in = laplacian_smooth(v_gm_in, f_in, lambd=1.0)
                    n_in = compute_normal(v_gm_in, f_in)
                    v_gm_in += 0.002 * n_in

                if condition == 'a':
                    # Deformation model for grey matter
                    model_gm_def = models['model_gm_def']
                    model_gm_def.set_data(v_gm_in, volume_in, f_in)
                    v_gm_pred = odeint(model_gm_def, v_gm_in, t=T, method=solver,
                                       options=dict(step_size=step_size/2))[-1]

                    # Classification model
                    model_gm_cls = models['model_gm_cls']
                    model_gm_cls.set_data(v_gm_pred, volume_in, f=f_in)
                    _dx = model_gm_cls(T, v_gm_pred)
                    class_logits_gm = model_gm_cls.get_class_logits()
                    # Add LogSoftmax
                    class_logits_gm = F.log_softmax(class_logits_gm, dim=1)
                    class_logits_gm = class_logits_gm.unsqueeze(0)
                    class_probs_gm = class_logits_gm.permute(0, 2, 1)
                    class_gm_pred = torch.argmax(class_probs_gm, dim=1).cpu().numpy()

                elif condition == 'b':
                    # Combined model
                    model_gm = models['model_gm']
                    model_gm.set_data(v_gm_in, volume_in, f_in)
                    v_gm_pred = odeint(model_gm, v_gm_in, t=T, method=solver,
                                       options=dict(step_size=step_size/2))[-1]

                    model_gm.set_data(v_gm_pred, volume_in, f=f_in)
                    _dx = model_gm(T, v_gm_pred)
                    class_logits_gm = model_gm.get_class_logits()
                    # Add LogSoftmax
                    class_logits_gm = F.log_softmax(class_logits_gm, dim=1)
                    class_logits_gm = class_logits_gm.unsqueeze(0)
                    class_probs_gm = class_logits_gm.permute(0, 2, 1)
                    class_gm_pred = torch.argmax(class_probs_gm, dim=1).cpu().numpy()

                v_wm_pred = v_wm_pred[0].cpu().numpy()
                f_wm_pred = f_in[0].cpu().numpy()
                v_gm_pred = v_gm_pred[0].cpu().numpy()
                f_gm_pred = f_in[0].cpu().numpy()
                # Map the surface coordinate from [-1,1] to its original space
                v_wm_pred, f_wm_pred = process_surface_inverse(v_wm_pred, f_wm_pred, data_name)
                v_gm_pred, f_gm_pred = process_surface_inverse(v_gm_pred, f_gm_pred, data_name)

                # ------ Save predicted surfaces and annotations -------
                # Define the save path
                pred_surface_basename_wm = f'{data_name}_{surf_hemi}_{subid}_wm_pred'
                pred_surface_basename_gm = f'{data_name}_{surf_hemi}_{subid}_gm_pred'

                pred_surface_path_wm = os.path.join(wm_hemi_dir, pred_surface_basename_wm)
                pred_surface_path_gm = os.path.join(gm_hemi_dir, pred_surface_basename_gm)

                # Save the predicted surface with annotations
                try:
                    save_mesh_with_annotations(v_wm_pred, f_wm_pred, class_wm_pred, ctab_wm, pred_surface_path_wm, data_name)
                    print(f"Saved predicted white matter surface for {subid} in {pred_surface_path_wm}")
                    
                    save_mesh_with_annotations(v_gm_pred, f_gm_pred, class_gm_pred, ctab_gm, pred_surface_path_gm, data_name)
                    print(f"Saved predicted grey matter surface for {subid} in {pred_surface_path_gm}")
                    
                except Exception as e:
                    print(f"Error saving mesh for subject {subid}: {e}. Skipping.")
                    continue

    print("Processing completed.")
