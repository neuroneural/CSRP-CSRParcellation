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
from model.net import Unet, CortexODE
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

def save_mesh_with_annotations(verts, faces, labels=None, ctab=None, save_path_fs=None, data_name='hcp', epoch_info=None):
    """
    Save the mesh with annotations using nibabel.

    Parameters:
    - verts: NumPy array of vertices.
    - faces: NumPy array of faces.
    - labels: (Optional) NumPy array of labels.
    - ctab: (Optional) Color table for annotations.
    - save_path_fs: Path to save the FreeSurfer files.
    - data_name: Name of the dataset (e.g., 'hcp', 'adni').
    - epoch_info: (Optional) String containing epoch information to include in file names.
    """
    if save_path_fs is None:
        raise ValueError("save_path_fs must be provided to save the mesh.")

    verts = verts.squeeze()
    faces = faces.squeeze().astype(np.int32)

    assert not np.isnan(verts.max()), "The value is NaN in vertices."
    assert not np.isnan(verts.min()), "The value is NaN in vertices."

    if labels is not None:
        labels = labels.squeeze().astype(np.int32)
        if isinstance(ctab, torch.Tensor):
            ctab = ctab.numpy()
        ctab = ctab.astype(np.int32)
        assert ctab.shape[1] == 5, "ctab should have 5 columns for RGBA and region labels."

    # Modify save paths to include epoch information
    if epoch_info is not None:
        surf_path = f"{save_path_fs}_epoch{epoch_info}.surf"
        annot_path = f"{save_path_fs}_epoch{epoch_info}.annot"
    else:
        surf_path = f"{save_path_fs}.surf"
        annot_path = f"{save_path_fs}.annot"

    # Save geometry
    nib.freesurfer.io.write_geometry(surf_path, verts, faces)

    # Save annotations if labels are provided
    if labels is not None and ctab is not None:
        nib.freesurfer.io.write_annot(annot_path, labels, ctab, decode_names(), fill_ctab=False)

def extract_rand_num_and_epoch_from_filename(filename):
    """
    Extract the random number and epoch from the model filename.
    """
    # Extract random number and epoch
    # Update the regex pattern to match your filename formats
    match = re.search(r'_([\d]+)epochs.*_(\d+)\.pt$', filename)
    if match:
        epoch = int(match.group(1))
        rand_num = int(match.group(2))
        return rand_num, epoch
    else:
        # Handle cases without epoch in the filename
        match = re.search(r'_(\d+)\.pt$', filename)
        if match:
            rand_num = int(match.group(1))
            epoch = None  # Epoch not found
            return rand_num, epoch
        else:
            return None, None

def load_models_and_weights(device, config):
    """
    Load the models and their weights based on the configuration.

    Parameters:
    - device: Torch device ('cuda' or 'cpu').
    - config: Configuration object.

    Returns:
    - models: Dictionary containing loaded models.
    - condition: String indicating the loading condition.
    - epoch_info: Dictionary containing epoch information for models.
    """
    print('Loading models on device:', device)
    C = config.dim_h
    K = config.kernel_size
    Q = config.n_scale
    use_gcn = config.gnn == 'gcn'

    models = {}
    epoch_info = {}

    # Determine model architecture
    model_type = config.model_type.lower()
    if model_type not in ['csrvcv3', 'cortexode']:
        print(f"Unsupported model_type '{config.model_type}'. Supported types: 'csrvcv3', 'cortexode'. Exiting.")
        exit(1)

    if model_type == 'csrvcv3':
        # Existing CSRVCV3 loading logic
        wm_def_specified = hasattr(config, 'model_file_wm_deformation') and config.model_file_wm_deformation is not None
        wm_cls_specified = hasattr(config, 'model_file_wm_classification') and config.model_file_wm_classification is not None
        gm_def_specified = hasattr(config, 'model_file_gm_deformation') and config.model_file_gm_deformation is not None
        gm_cls_specified = hasattr(config, 'model_file_gm_classification') and config.model_file_gm_classification is not None

        if wm_def_specified and wm_cls_specified and gm_def_specified and gm_cls_specified:
            # Condition 'a'
            condition = 'a'
            print(f"Model condition: {condition}")

            # Load WM Deformation Model
            model_file_wm_def = os.path.join(config.model_dir.strip(), config.model_file_wm_deformation.strip())
            if not os.path.exists(model_file_wm_def):
                print(f"WM Deformation Model file '{model_file_wm_def}' not found. Exiting.")
                exit(1)
            rand_num_wm_def, epoch_wm_def = extract_rand_num_and_epoch_from_filename(config.model_file_wm_deformation)
            print(f"WM Deformation Model Random Number: {rand_num_wm_def}, Epochs: {epoch_wm_def}")
            epoch_info['wm_def_epoch'] = epoch_wm_def
            model_wm_def = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config.sf, gnn_layers=config.gnn_layers,
                                   use_gcn=use_gcn, gat_heads=config.gat_heads, num_classes=config.num_classes).to(device)
            checkpoint_wm_def = torch.load(model_file_wm_def, map_location=device)
            if 'model_state_dict' in checkpoint_wm_def:
                model_wm_def.load_state_dict(checkpoint_wm_def['model_state_dict'])
            else:
                model_wm_def.load_state_dict(checkpoint_wm_def)
            model_wm_def.eval()
            models['model_wm_def'] = model_wm_def

            # Load WM Classification Model
            model_file_wm_cls = os.path.join(config.model_dir.strip(), config.model_file_wm_classification.strip())
            if not os.path.exists(model_file_wm_cls):
                print(f"WM Classification Model file '{model_file_wm_cls}' not found. Exiting.")
                exit(1)
            rand_num_wm_cls, epoch_wm_cls = extract_rand_num_and_epoch_from_filename(config.model_file_wm_classification)
            print(f"WM Classification Model Random Number: {rand_num_wm_cls}, Epochs: {epoch_wm_cls}")
            epoch_info['wm_cls_epoch'] = epoch_wm_cls
            model_wm_cls = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config.sf, gnn_layers=config.gnn_layers,
                                   use_gcn=use_gcn, gat_heads=config.gat_heads, num_classes=config.num_classes).to(device)
            checkpoint_wm_cls = torch.load(model_file_wm_cls, map_location=device)
            if 'model_state_dict' in checkpoint_wm_cls:
                model_wm_cls.load_state_dict(checkpoint_wm_cls['model_state_dict'])
            else:
                model_wm_cls.load_state_dict(checkpoint_wm_cls)
            model_wm_cls.eval()
            models['model_wm_cls'] = model_wm_cls

            # Load GM Deformation Model
            model_file_gm_def = os.path.join(config.model_dir.strip(), config.model_file_gm_deformation.strip())
            if not os.path.exists(model_file_gm_def):
                print(f"GM Deformation Model file '{model_file_gm_def}' not found. Exiting.")
                exit(1)
            rand_num_gm_def, epoch_gm_def = extract_rand_num_and_epoch_from_filename(config.model_file_gm_deformation)
            print(f"GM Deformation Model Random Number: {rand_num_gm_def}, Epochs: {epoch_gm_def}")
            epoch_info['gm_def_epoch'] = epoch_gm_def
            model_gm_def = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config.sf, gnn_layers=config.gnn_layers,
                                   use_gcn=use_gcn, gat_heads=config.gat_heads, num_classes=config.num_classes).to(device)
            checkpoint_gm_def = torch.load(model_file_gm_def, map_location=device)
            if 'model_state_dict' in checkpoint_gm_def:
                model_gm_def.load_state_dict(checkpoint_gm_def['model_state_dict'])
            else:
                model_gm_def.load_state_dict(checkpoint_gm_def)
            model_gm_def.eval()
            models['model_gm_def'] = model_gm_def

            # Load GM Classification Model
            model_file_gm_cls = os.path.join(config.model_dir.strip(), config.model_file_gm_classification.strip())
            if not os.path.exists(model_file_gm_cls):
                print(f"GM Classification Model file '{model_file_gm_cls}' not found. Exiting.")
                exit(1)
            rand_num_gm_cls, epoch_gm_cls = extract_rand_num_and_epoch_from_filename(config.model_file_gm_classification)
            print(f"GM Classification Model Random Number: {rand_num_gm_cls}, Epochs: {epoch_gm_cls}")
            epoch_info['gm_cls_epoch'] = epoch_gm_cls
            model_gm_cls = CSRVCV3(dim_h=C, kernel_size=K, n_scale=Q, sf=config.sf, gnn_layers=config.gnn_layers,
                                   use_gcn=use_gcn, gat_heads=config.gat_heads, num_classes=config.num_classes).to(device)
            checkpoint_gm_cls = torch.load(model_file_gm_cls, map_location=device)
            if 'model_state_dict' in checkpoint_gm_cls:
                model_gm_cls.load_state_dict(checkpoint_gm_cls['model_state_dict'])
            else:
                model_gm_cls.load_state_dict(checkpoint_gm_cls)
            model_gm_cls.eval()
            models['model_gm_cls'] = model_gm_cls

        else:
            # Other conditions can be handled similarly if needed
            print("Unsupported condition for CSRVCV3 models. Exiting.")
            exit(1)

    elif model_type == 'cortexode':
        # Loading CortexODE model
        condition = 'cortexode'
        print(f"Model condition: {condition}")

        # Load CortexODE for WM
        if hasattr(config, 'model_file_wm') and config.model_file_wm is not None:
            model_file_wm = os.path.join(config.model_dir.strip(), config.model_file_wm.strip())
            if not os.path.exists(model_file_wm):
                print(f"CortexODE WM Model file '{model_file_wm}' not found. Exiting.")
                exit(1)
            rand_num_wm, epoch_wm = extract_rand_num_and_epoch_from_filename(config.model_file_wm)
            print(f"CortexODE WM Model Random Number: {rand_num_wm}, Epochs: {epoch_wm}")
            epoch_info['wm_epoch'] = epoch_wm
            model_wm = CortexODE(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)
            checkpoint_wm = torch.load(model_file_wm, map_location=device)
            if 'model_state_dict' in checkpoint_wm:
                model_wm.load_state_dict(checkpoint_wm['model_state_dict'])
            else:
                model_wm.load_state_dict(checkpoint_wm)
            model_wm.eval()
            models['model_wm'] = model_wm
        else:
            models['model_wm'] = None

        # Load CortexODE for GM
        if hasattr(config, 'model_file_gm') and config.model_file_gm is not None:
            model_file_gm = os.path.join(config.model_dir.strip(), config.model_file_gm.strip())
            if not os.path.exists(model_file_gm):
                print(f"CortexODE GM Model file '{model_file_gm}' not found. Exiting.")
                exit(1)
            rand_num_gm, epoch_gm = extract_rand_num_and_epoch_from_filename(config.model_file_gm)
            print(f"CortexODE GM Model Random Number: {rand_num_gm}, Epochs: {epoch_gm}")
            epoch_info['gm_epoch'] = epoch_gm
            model_gm = CortexODE(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)
            checkpoint_gm = torch.load(model_file_gm, map_location=device)
            if 'model_state_dict' in checkpoint_gm:
                model_gm.load_state_dict(checkpoint_gm['model_state_dict'])
            else:
                model_gm.load_state_dict(checkpoint_gm)
            model_gm.eval()
            models['model_gm'] = model_gm
        else:
            models['model_gm'] = None
    else:
        print(f"Unknown model_type '{model_type}'. Exiting.")
        exit(1)

    return models, condition, epoch_info

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
    model_type = config.model_type # 'csrvcv3' or 'cortexode'

    C = config.dim_h                       # Hidden dimension of features
    K = config.kernel_size                 # Kernel / cube size
    Q = config.n_scale                     # Multi-scale input

    step_size = config.step_size           # Step size of integration
    solver = config.solver                 # ODE solver (e.g., 'euler', 'rk4')
    n_inflate = config.n_inflate           # Inflation iterations
    rho = config.rho                       # Inflation scale

    # ------ Load the segmentation network ------
    if config.seg_model_file is not None:
        segnet = Unet(c_in=1, c_out=3).to(device)
        segnet.load_state_dict(torch.load(os.path.join(config.model_dir.strip(), config.seg_model_file.strip()), map_location=device))
        segnet.eval()
        print(f"Loaded segmentation model from '{config.seg_model_file}'")
    else:
        print("No segmentation model file provided. Exiting.")
        exit(1)

    # ------ Load the models and infer the condition ------
    models, condition, epoch_info = load_models_and_weights(device, config)

    # ------ Create result subdirectories based on condition ------
    folder_name = f"{condition}"
    result_subdir = os.path.join(result_dir.strip(), folder_name.strip())
    os.makedirs(result_subdir, exist_ok=True)

    wm_hemi_dir = os.path.join(result_subdir.strip(), 'wm', surf_hemi.strip())
    gm_hemi_dir = os.path.join(result_subdir.strip(), 'gm', surf_hemi.strip())
    wm_gt_dir = os.path.join(result_subdir.strip(), 'wm_gt', surf_hemi.strip())
    gm_gt_dir = os.path.join(result_subdir.strip(), 'gm_gt', surf_hemi.strip())

    os.makedirs(wm_hemi_dir, exist_ok=True)
    os.makedirs(gm_hemi_dir, exist_ok=True)
    os.makedirs(wm_gt_dir, exist_ok=True)
    os.makedirs(gm_gt_dir, exist_ok=True)

    print(f"Saving WM results to: {wm_hemi_dir}")
    print(f"Saving GM results to: {gm_hemi_dir}")
    print(f"Saving WM ground truth to: {wm_gt_dir}")
    print(f"Saving GM ground truth to: {gm_gt_dir}")

    # ------ Prepare test data ------
    testset = SegDataset(config=config, data_usage='test')
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    brain_dataset_wm = BrainDataset(config, data_usage='test', affCtab=True)
    brain_dataset_gm = BrainDataset(config, data_usage='test', affCtab=True)

    T = torch.Tensor([0,1]).to(device)

    for batch_idx, data in enumerate(testloader):
        volume_in, seg_gt, subid, _aff = data

        subid = str(subid[0])
        volume_in = volume_in.to(device)

        # Ensure the index corresponds to the current batch
        try:
            brain_arr_wm, v_in_wm, v_gt_wm, f_in_wm, f_gt_wm, labels_wm, aff_wm, ctab_wm, sub_id_wm = brain_dataset_wm[batch_idx]
            brain_arr_gm, v_in_gm, v_gt_gm, f_in_gm, f_gt_gm, labels_gm, aff_gm, ctab_gm, sub_id_gm = brain_dataset_gm[batch_idx]
            assert subid == sub_id_wm, f"Mismatch in WM subject IDs: {subid} vs {sub_id_wm}"
            assert subid == sub_id_gm, f"Mismatch in GM subject IDs: {subid} vs {sub_id_gm}"
        except IndexError:
            print(f"BrainDataset index {batch_idx} out of range. Skipping subject {subid}.")
            continue
        except AssertionError as ae:
            print(f"Assertion error: {ae}. Skipping subject {subid}.")
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
                    print(f"Unknown hemisphere '{surf_hemi}'. Skipping subject {subid}.")
                    continue

            # ------ Extract initial surface -------
            try:
                v_in, f_in = seg2surf(seg, data_name, sigma=0.5, alpha=16, level=0.8, n_smooth=2, device=device)
            except ValueError as e:
                print(f"Error in seg2surf for subject {subid}: {e}. Skipping.")
                continue

        # ------ Predict the surface using the model -------
        if test_type in ['pred', 'eval']:
            with torch.no_grad():
                # Prepare input tensors
                if config.seg_model_file is not None:
                    v_in_tensor = torch.Tensor(v_in).float().unsqueeze(0).to(device)
                    f_in_tensor = torch.LongTensor(f_in).unsqueeze(0).to(device)
                else:
                    v_in_tensor = v_in_wm.unsqueeze(0).to(device)
                    f_in_tensor = f_in_wm.unsqueeze(0).to(device)

                if condition == 'a':
                    if model_type == 'csrvcv3':
                        # Deformation and Classification for WM
                        model_wm_def = models.get('model_wm_def', None)
                        model_wm_cls = models.get('model_wm_cls', None)
                        if model_wm_def is not None and model_wm_cls is not None:
                            model_wm_def.set_data(v_in_tensor, volume_in, f_in_tensor)
                            v_wm_pred = odeint(model_wm_def, v_in_tensor, t=T, method=solver,
                                               options=dict(step_size=step_size))[-1]

                            model_wm_cls.set_data(v_wm_pred, volume_in, f=f_in_tensor)
                            _dx = model_wm_cls(T, v_wm_pred)
                            class_logits_wm = model_wm_cls.get_class_logits()
                            # Add LogSoftmax
                            class_logits_wm = class_logits_wm.unsqueeze(0)
                            class_logits_wm = F.log_softmax(class_logits_wm, dim=1)
                            class_probs_wm = class_logits_wm.permute(0, 2, 1)
                            class_wm_pred = torch.argmax(class_probs_wm, dim=1).cpu().numpy()
                        else:
                            print(f"WM Deformation or Classification model not loaded for subject {subid}. Skipping.")
                            continue

                        # Inflate and smooth for grey matter
                        v_gm_in = v_wm_pred.clone()
                        for i in range(n_inflate):
                            v_gm_in = laplacian_smooth(v_gm_in, f_in_tensor, lambd=1.0)
                            n_in = compute_normal(v_gm_in, f_in_tensor)
                            v_gm_in += rho * n_in

                        model_gm_def = models.get('model_gm_def', None)
                        model_gm_cls = models.get('model_gm_cls', None)
                        if model_gm_def is not None and model_gm_cls is not None:
                            model_gm_def.set_data(v_gm_in, volume_in, f_in_tensor)
                            v_gm_pred = odeint(model_gm_def, v_gm_in, t=T, method=solver,
                                               options=dict(step_size=step_size/2))[-1]

                            model_gm_cls.set_data(v_gm_pred, volume_in, f=f_in_tensor)
                            _dx = model_gm_cls(T, v_gm_pred)
                            class_logits_gm = model_gm_cls.get_class_logits()
                            # Add LogSoftmax
                            class_logits_gm = class_logits_gm.unsqueeze(0)
                            class_logits_gm = F.log_softmax(class_logits_gm, dim=1)
                            class_probs_gm = class_logits_gm.permute(0, 2, 1)
                            class_gm_pred = torch.argmax(class_probs_gm, dim=1).cpu().numpy()
                        else:
                            print(f"GM Deformation or Classification model not loaded for subject {subid}. Skipping.")
                            continue

                    else:
                        print(f"Unsupported model architecture '{model_type}' for condition 'a'. Skipping.")
                        continue

                elif condition == 'cortexode':
                    if model_type == 'cortexode':
                        # Deformation using CortexODE for WM
                        model_wm = models.get('model_wm', None)
                        if model_wm is not None:
                            model_wm.set_data(v_in_tensor, volume_in)
                            v_wm_pred = odeint(model_wm, v_in_tensor, t=T, method=solver,
                                               options=dict(step_size=step_size))[-1]
                        else:
                            print(f"No CortexODE WM model loaded for subject {subid}. Skipping.")
                            continue

                        # Inflate and smooth for GM
                        v_gm_in = v_wm_pred.clone()
                        for i in range(n_inflate):
                            v_gm_in = laplacian_smooth(v_gm_in, f_in_tensor, lambd=1.0)
                            n_in = compute_normal(v_gm_in, f_in_tensor)
                            v_gm_in += rho * n_in

                        # Deformation using CortexODE for GM
                        model_gm = models.get('model_gm', None)
                        if model_gm is not None:
                            model_gm.set_data(v_gm_in, volume_in)
                            v_gm_pred = odeint(model_gm, v_gm_in, t=T, method=solver,
                                               options=dict(step_size=step_size/2))[-1]
                        else:
                            print(f"No CortexODE GM model loaded for subject {subid}. Skipping.")
                            continue

                        # No classification for CortexODE
                        class_wm_pred = None
                        class_gm_pred = None

                    else:
                        print(f"Unsupported model architecture '{model_type}' for condition '{condition}'. Skipping.")
                        continue

                else:
                    print(f"Unsupported condition '{condition}'. Skipping subject {subid}.")
                    continue

                # Convert predictions to NumPy
                v_wm_pred_np = v_wm_pred[0].cpu().numpy()
                f_wm_pred_np = f_in_tensor[0].cpu().numpy()
                v_gm_pred_np = v_gm_pred[0].cpu().numpy()
                f_gm_pred_np = f_in_tensor[0].cpu().numpy()

                # Map the surface coordinate from [-1,1] to its original space
                v_wm_pred_mapped, f_wm_pred_mapped = process_surface_inverse(v_wm_pred_np, f_wm_pred_np, data_name)
                v_gm_pred_mapped, f_gm_pred_mapped = process_surface_inverse(v_gm_pred_np, f_gm_pred_np, data_name)

                # Map ground truth surfaces to original space
                v_gt_wm_np = v_gt_wm.cpu().numpy()
                f_gt_wm_np = f_gt_wm.cpu().numpy()
                v_gt_gm_np = v_gt_gm.cpu().numpy()
                f_gt_gm_np = f_gt_gm.cpu().numpy()

                v_gt_wm_mapped, f_gt_wm_mapped = process_surface_inverse(v_gt_wm_np, f_gt_wm_np, data_name)
                v_gt_gm_mapped, f_gt_gm_mapped = process_surface_inverse(v_gt_gm_np, f_gt_gm_np, data_name)

                # ------ Save predicted surfaces and annotations -------
                # Define the save paths, including epoch information
                if condition == 'a':
                    wm_epoch = epoch_info.get('wm_def_epoch', None)
                    gm_epoch = epoch_info.get('gm_def_epoch', None)
                elif condition == 'cortexode':
                    wm_epoch = epoch_info.get('wm_epoch', None)
                    gm_epoch = epoch_info.get('gm_epoch', None)
                else:
                    wm_epoch = None
                    gm_epoch = None

                pred_surface_basename_wm = f'{data_name}_{surf_hemi}_{subid}_wm_pred'
                pred_surface_basename_gm = f'{data_name}_{surf_hemi}_{subid}_gm_pred'
                gt_surface_basename_wm = f'{data_name}_{surf_hemi}_{subid}_wm_gt'
                gt_surface_basename_gm = f'{data_name}_{surf_hemi}_{subid}_gm_gt'

                pred_surface_path_wm = os.path.join(wm_hemi_dir.strip(), pred_surface_basename_wm.strip())
                pred_surface_path_gm = os.path.join(gm_hemi_dir.strip(), pred_surface_basename_gm.strip())
                gt_surface_path_wm = os.path.join(wm_gt_dir.strip(), gt_surface_basename_wm.strip())
                gt_surface_path_gm = os.path.join(gm_gt_dir.strip(), gt_surface_basename_gm.strip())

                # Save the predicted surface with annotations if using CSRVCV3
                try:
                    if model_type == 'csrvcv3':
                        save_mesh_with_annotations(v_wm_pred_mapped, f_wm_pred_mapped, labels=class_wm_pred, ctab=ctab_wm, save_path_fs=pred_surface_path_wm, data_name=data_name, epoch_info=wm_epoch)
                        print(f"Saved predicted white matter surface with annotations for {subid} at '{pred_surface_path_wm}_epoch{wm_epoch}.surf' and '.annot'")

                        save_mesh_with_annotations(v_gm_pred_mapped, f_gm_pred_mapped, labels=class_gm_pred, ctab=ctab_gm, save_path_fs=pred_surface_path_gm, data_name=data_name, epoch_info=gm_epoch)
                        print(f"Saved predicted grey matter surface with annotations for {subid} at '{pred_surface_path_gm}_epoch{gm_epoch}.surf' and '.annot'")
                    elif model_type == 'cortexode':
                        # Save without annotations
                        save_mesh_with_annotations(v_wm_pred_mapped, f_wm_pred_mapped, labels=None, ctab=None, save_path_fs=pred_surface_path_wm, data_name=data_name, epoch_info=wm_epoch)
                        print(f"Saved predicted white matter surface for {subid} at '{pred_surface_path_wm}_epoch{wm_epoch}.surf'")

                        save_mesh_with_annotations(v_gm_pred_mapped, f_gm_pred_mapped, labels=None, ctab=None, save_path_fs=pred_surface_path_gm, data_name=data_name, epoch_info=gm_epoch)
                        print(f"Saved predicted grey matter surface for {subid} at '{pred_surface_path_gm}_epoch{gm_epoch}.surf'")
                    else:
                        print(f"Unsupported model architecture '{model_type}'. Skipping saving predicted surfaces.")
                except Exception as e:
                    print(f"Error saving predicted mesh for subject {subid}: {e}. Skipping.")
                    continue

                # ------ Save ground truth surfaces -------
                try:
                    # Save WM ground truth surface
                    #gt_surface_path_wm_full = f"{gt_surface_path_wm}.surf"
                    #nib.freesurfer.io.write_geometry(gt_surface_path_wm_full, v_gt_wm_mapped, f_gt_wm_mapped)
                    save_mesh_with_annotations(v_gt_wm_mapped, f_gt_wm_mapped, labels=labels_wm.cpu().numpy(), ctab=ctab_wm, save_path_fs=gt_surface_path_wm, data_name=data_name)
                        
                    print(f"Saved ground truth white matter surface for {subid} at '{gt_surface_path_wm}'")

                    # Save GM ground truth surface
                    # gt_surface_path_gm_full = f"{gt_surface_path_gm}.surf"
                    # nib.freesurfer.io.write_geometry(gt_surface_path_gm_full, v_gt_gm_mapped, f_gt_gm_mapped)
                    save_mesh_with_annotations(v_gt_gm_mapped, f_gt_gm_mapped, labels=labels_gm.cpu().numpy(), ctab=ctab_gm, save_path_fs=gt_surface_path_gm, data_name=data_name)
                    
                    print(f"Saved ground truth grey matter surface for {subid} at '{gt_surface_path_gm}'")
                except Exception as e:
                    print(f"Error saving ground truth mesh for subject {subid}: {e}. Skipping.")
                    continue

    print("Processing completed.")
