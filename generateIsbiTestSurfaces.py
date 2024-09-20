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

from data.preprocess import process_volume, process_surface, process_surface_inverse
from data.datautil import decode_names  # Import your tested decode_names function
from util.mesh import laplacian_smooth, compute_normal, compute_mesh_distance, check_self_intersect
from util.tca import topology
from model.net import Unet
from model.csrvcv2 import CSRVCV2  # Updated import for new model
from config import load_config
from data.csrandvcdataloader import SegDataset, BrainDataset
from torch.utils.data import DataLoader

# Initialize topology correction
topo_correct = topology()

import matplotlib.pyplot as plt

import copy


def seg2surf(seg, data_name='hcp', sigma=0.5, alpha=16, level=0.8, n_smooth=2):
    """
    Extract the surface based on the segmentation.
    """
    # ------ connected components checking ------ 
    cc, nc = compute_cc(seg, connectivity=2, return_num=True)
    cc_id = 1 + np.argmax(np.array([np.count_nonzero(cc == i) for i in range(1, nc + 1)]))
    seg = (cc == cc_id).astype(np.float64)

    # ------ generate signed distance function ------ 
    sdf = -cdt(seg) + cdt(1 - seg)
    sdf = sdf.astype(float)
    sdf = gaussian(sdf, sigma=sigma)

    # ------ topology correction ------
    sdf_topo = topo_correct.apply(sdf, threshold=alpha)

    # ------ marching cubes ------
    v_mc, f_mc, _, _ = marching_cubes(-sdf_topo, level=-level, method='lorensen')
    v_mc = v_mc[:, [2, 1, 0]].copy()
    f_mc = f_mc.copy()
    D1, D2, D3 = sdf_topo.shape
    D = max(D1, D2, D3)
    v_mc = (2 * v_mc - [D3, D2, D1]) / D   # rescale to [-1,1]
    
    # ------ bias correction ------
    if data_name == 'hcp':
        v_mc = v_mc + [0.0090, 0.0058, 0.0088]
    elif data_name == 'adni':
        v_mc = v_mc + [0.0090, 0.0000, 0.0095]
        
    # ------ mesh smoothing ------
    v_mc = torch.Tensor(v_mc).unsqueeze(0).to(device)
    f_mc = torch.LongTensor(f_mc).unsqueeze(0).to(device)
    for j in range(n_smooth):    # smooth and inflate the mesh
        v_mc = laplacian_smooth(v_mc, f_mc, 'uniform', lambd=1)
    v_mc = v_mc[0].cpu().numpy()
    f_mc = f_mc[0].cpu().numpy()
    
    return v_mc, f_mc

def extract_model_info(filename):
    """
    Extract GNN layers and epoch information from the model filename.
    """
    layers = re.search(r'layers(\d+)', filename).group(1)
    epochs = re.search(r'(\d+)epochs', filename).group(1)
    return layers, epochs

# --- Add the save_mesh_with_annotations function ---
def save_mesh_with_annotations(verts, faces, labels, ctab, save_path_fs, data_name='hcp'):
    """
    Save the mesh with annotations using nibabel.
    """
    # Ensure inputs are correctly processed
    verts = verts.squeeze()
    faces = faces.squeeze().astype(np.int32)
    verts, faces = process_surface_inverse(verts, faces, data_name)
    
    labels = labels.squeeze().astype(np.int32)
    
    # Ensure ctab is correctly sized and in numpy array format
    if isinstance(ctab, torch.Tensor):
        ctab = ctab.numpy()
    ctab = ctab.astype(np.int32)
    print(f"ctab size: {ctab.shape}")
    assert ctab.shape[1] == 5, "ctab should have 5 columns for RGBA and region labels."
    
    # Save the surface in FreeSurfer format
    nib.freesurfer.write_geometry(save_path_fs + '.surf', verts, faces)
    
    # Save the annotation
    nib.freesurfer.write_annot(save_path_fs + '.annot', 
                               labels,
                               ctab,
                               decode_names(),
                               fill_ctab=False)

if __name__ == '__main__':
    # ------ load configuration ------
    config = load_config()
    test_type = config.test_type  # initial surface / prediction / evaluation
    data_dir = config.data_dir  # directory of datasets
    model_dir = config.model_dir  # directory of pretrained models
    init_dir = config.init_dir  # directory for saving the initial surfaces
    result_dir = config.result_dir  # directory for saving the predicted surfaces
    data_name = config.data_name  # hcp, adni, dhcp
    surf_hemi = config.surf_hemi  # lh, rh
    device = config.device
    tag = config.tag  # identity of the experiment

    C = config.dim_h     # hidden dimension of features
    K = config.kernel_size    # kernel / cube size
    Q = config.n_scale    # multi-scale input
    
    step_size = config.step_size    # step size of integration
    solver = config.solver    # ODE solver
    n_inflate = config.n_inflate  # inflation iterations
    rho = config.rho  # inflation scale

    print('loading models...')
    # ------ load models ------
    segnet = Unet(c_in=1, c_out=3).to(device)
    segnet.load_state_dict(torch.load(os.path.join(model_dir, config.seg_model_file), map_location=torch.device(config.device)))

    T = torch.Tensor([0, 1]).to(device)

    print('C', "K", "Q", "num_classes")
    print(C, K, Q, config.num_classes)
    
    if config.gnn == 'gat':
        use_gcn = False
    elif config.gnn == 'gcn':
        use_gcn = True
    else:
        use_gcn = False  # default to False if not specified

    csrvcv2_wm = CSRVCV2(dim_h=C,
                            kernel_size=K,
                            n_scale=Q,
                            sf=config.sf,
                            gnn_layers=config.gnn_layers,
                            use_gcn=use_gcn,
                            gat_heads=config.gat_heads,
                            num_classes=config.num_classes).to(device)
    csrvcv2_gm = CSRVCV2(dim_h=C,
                            kernel_size=K,
                            n_scale=Q,
                            sf=config.sf,
                            gnn_layers=config.gnn_layers,
                            use_gcn=use_gcn,
                            gat_heads=config.gat_heads,
                            num_classes=config.num_classes).to(device)

    # Load models for white matter and pial surfaces
    csrvcv2_wm.load_state_dict(torch.load(os.path.join(model_dir, config.wm_model_file), map_location=torch.device(config.device)))
    csrvcv2_gm.load_state_dict(torch.load(os.path.join(model_dir, config.gm_model_file), map_location=torch.device(config.device)))
    
    csrvcv2_wm.eval()
    csrvcv2_gm.eval()

    # Extract GNN layers and epochs from the model filenames
    wm_layers, wm_epochs = extract_model_info(config.wm_model_file)
    gm_layers, gm_epochs = extract_model_info(config.gm_model_file)

    # ------ start testing ------
    if test_type in ['eval', 'pred']:
        testset = SegDataset(config=config, data_usage='test')
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    for idx, data in enumerate(testloader):
        volume_in, seg_gt, subid, _aff = data
        subid = str(subid[0])
        volume_in = volume_in.to(device)

        # Obtain the colormap (ctab) using BrainDataset for a single patient
        config_wm = copy.deepcopy(config)
        config_wm.surf_type='wm'
        config_gm = copy.deepcopy(config)
        config_gm.surf_type='gm'
        brain_dataset_wm = BrainDataset(config_wm, data_usage='test', affCtab=True)
        brain_dataset_gm = BrainDataset(config_gm, data_usage='test', affCtab=True)
        # _, _, _, _, _, _, _, ctab = brain_dataset_wm[idx]
        # Unpack all items from the dataset
        brain_arr, v_in, v_gt, f_in, f_gt, labels, aff, ctab = brain_dataset_wm[idx]


        # ------- predict segmentation -------
        with torch.no_grad():
            seg_out = segnet(volume_in)
            seg_pred = torch.argmax(seg_out, dim=1)[0]

            # Handle hemisphere segmentation
            if surf_hemi == 'lh':
                seg = (seg_pred == 1).cpu().numpy()  # lh
            elif surf_hemi == 'rh':
                seg = (seg_pred == 2).cpu().numpy()  # rh
            elif surf_hemi == 'none':
                print('Skipping surface creation, evaluating segmentation only')
                exit()

        # ------- extract initial surface ------- 
        v_in, f_in = seg2surf(seg, data_name, sigma=0.5, alpha=16, level=0.8, n_smooth=2)

        # ------- save initial surface ------- 
        if test_type == 'init':
            mesh_init = trimesh.Trimesh(v_in, f_in)
            mesh_init.export(init_dir + f'init_{data_name}_{surf_hemi}_{subid}.obj')

        # ------- predict white matter surface ------- 
        with torch.no_grad():
            v_in_tensor = torch.Tensor(v_in).unsqueeze(0).to(device)
            f_in_tensor = torch.LongTensor(f_in).unsqueeze(0).to(device)
            
            csrvcv2_wm.set_data(v_in_tensor, volume_in, f=f_in_tensor)
            v_wm_pred = odeint(csrvcv2_wm, v_in_tensor, t=T, method=solver, options=dict(step_size=step_size))[-1]
            class_logits_wm = csrvcv2_wm.get_class_logits()
            class_pred_wm = torch.argmax(class_logits_wm, dim=1).cpu().numpy()

            # Inflate and smooth to create the initial gray matter (GM) surface
            v_gm_in = v_wm_pred.clone()
            for i in range(2):
                v_gm_in = laplacian_smooth(v_gm_in, f_in_tensor, lambd=1.0)
                n_in = compute_normal(v_gm_in, f_in_tensor)
                v_gm_in += 0.002 * n_in

            # ------- predict gray matter surface ------- 
            csrvcv2_gm.set_data(v_gm_in, volume_in, f=f_in_tensor)
            v_gm_pred = odeint(csrvcv2_gm, v_gm_in, t=T, method=solver, options=dict(step_size=step_size / 2))[-1]
            class_logits_gm = csrvcv2_gm.get_class_logits()
            class_pred_gm = torch.argmax(class_logits_gm, dim=1).cpu().numpy()

            # Convert to numpy arrays for saving
            v_wm_pred = v_wm_pred[0].cpu().numpy()
            f_wm_pred = f_in_tensor[0].cpu().numpy()
            v_gm_pred = v_gm_pred[0].cpu().numpy()
            f_gm_pred = f_in_tensor[0].cpu().numpy()

        # ------- save predicted surfaces and annotations ------- 
        if test_type == 'pred':
            # Construct file naming convention using extracted model info
            wm_basename = f'{data_name}_{surf_hemi}_{subid}_wm_layers{wm_layers}_epochs{wm_epochs}'
            gm_basename = f'{data_name}_{surf_hemi}_{subid}_gm_layers{gm_layers}_epochs{gm_epochs}'
            wm_save_path = os.path.join(result_dir, wm_basename)
            gm_save_path = os.path.join(result_dir, gm_basename)

            # Save white matter surface and annotations
            save_mesh_with_annotations(v_wm_pred, f_wm_pred, class_pred_wm, ctab, wm_save_path, data_name)

            # Save gray matter surface and annotations
            save_mesh_with_annotations(v_gm_pred, f_gm_pred, class_pred_gm, ctab, gm_save_path, data_name)
            
            
            # Construct file naming convention for ground truth, now including surf_type
            gt_basename = f'{data_name}_{surf_hemi}_{subid}_wm_gt'#chatgpt, i may need to create two dataloaders of 
            gt_save_path = os.path.join(result_dir, gt_basename)

            # Save ground truth surface and annotations
            v_gt = v_gt.cpu().numpy()
            f_gt = f_gt.cpu().numpy()
            labels = labels.cpu().numpy()
            save_mesh_with_annotations(v_gt, f_gt, labels, ctab, gt_save_path, data_name)
            
            brain_arr, v_in, v_gt, f_in, f_gt, labels, aff, ctab = brain_dataset_gm[idx]

            # Construct file naming convention for ground truth, now including surf_type
            gt_basename = f'{data_name}_{surf_hemi}_{subid}_gm_gt'#chatgpt, i may need to create two dataloaders of 
            gt_save_path = os.path.join(result_dir, gt_basename)

            # Save ground truth surface and annotations
            v_gt = v_gt.cpu().numpy()
            f_gt = f_gt.cpu().numpy()
            labels = labels.cpu().numpy()
            save_mesh_with_annotations(v_gt, f_gt, labels, ctab, gt_save_path, data_name)
            