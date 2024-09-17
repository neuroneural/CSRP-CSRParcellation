import os
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
from util.mesh import laplacian_smooth, compute_normal
from util.tca import topology
from model.net import Unet
from model.csrvcv2 import CSRVCV2  # Replaces CortexODE
from config import load_config
from data.csrandvcdataloader import SegDataset, BrainDataset
from torch.utils.data import DataLoader

# Initialize topology correction
topo_correct = topology()

def seg2surf(seg, data_name='hcp', sigma=0.5, alpha=16, level=0.8, n_smooth=2):
    """
    Extract the surface based on the segmentation.
    """
    # Connected components checking
    cc, nc = compute_cc(seg, connectivity=2, return_num=True)
    cc_id = 1 + np.argmax(np.array([np.count_nonzero(cc == i) for i in range(1, nc+1)]))
    seg = (cc == cc_id).astype(np.float64)

    # Generate signed distance function
    sdf = -cdt(seg) + cdt(1 - seg)
    sdf = sdf.astype(float)
    sdf = gaussian(sdf, sigma=sigma)

    # Topology correction
    sdf_topo = topo_correct.apply(sdf, threshold=alpha)

    # Marching cubes
    v_mc, f_mc, _, _ = marching_cubes(-sdf_topo, level=-level, method='lorensen')
    v_mc = v_mc[:, [2, 1, 0]].copy()
    f_mc = f_mc.copy()
    D1, D2, D3 = sdf_topo.shape
    D = max(D1, D2, D3)
    v_mc = (2 * v_mc - [D3, D2, D1]) / D  # Rescale to [-1,1]

    # Bias correction
    if data_name == 'hcp':
        v_mc = v_mc + [0.0090, 0.0058, 0.0088]
    elif data_name == 'adni':
        v_mc = v_mc + [0.0090, 0.0000, 0.0095]

    # Mesh smoothing
    v_mc = torch.Tensor(v_mc).unsqueeze(0).to(device)
    f_mc = torch.LongTensor(f_mc).unsqueeze(0).to(device)
    for j in range(n_smooth):
        v_mc = laplacian_smooth(v_mc, f_mc, 'uniform', lambd=1)
    v_mc = v_mc[0].cpu().numpy()
    f_mc = f_mc[0].cpu().numpy()

    return v_mc, f_mc

def get_ctab_from_dataset(config, data_usage='test'):
    # Initialize the dataset to extract ctab
    dataset = BrainDataset(config=config, data_usage=data_usage, affCtab=True)
    brain_data = dataset[0]  # Load the first subject
    ctab = brain_data[-1]  # Extract ctab (assumed to be the last returned item)
    return ctab

if __name__ == '__main__':
    # Load configuration
    config = load_config()
    test_type = config.test_type  # initial surface / prediction / evaluation
    data_dir = config.data_dir
    model_dir = config.model_dir
    init_dir = config.init_dir
    result_dir = config.result_dir
    data_name = config.data_name
    surf_hemi = config.surf_hemi
    device = config.device
    tag = config.tag

    C = config.dim_h
    K = config.kernel_size
    Q = config.n_scale
    step_size = config.step_size
    solver = config.solver
    n_inflate = config.n_inflate
    rho = config.rho

    # Load segmentation model
    segnet = Unet(c_in=1, c_out=3).to(device)
    segnet.load_state_dict(torch.load(model_dir + 'model_seg_' + data_name + '_' + tag + '.pt'))

    # Load deformation model (CSRVCV2)
    if test_type in ['pred', 'eval']:
        if surf_hemi != 'none':
            T = torch.Tensor([0, 1]).to(device)
            csrvcv2 = CSRVCV2(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)
            csrvcv2.load_state_dict(torch.load(model_dir + 'model_csrvcv2_' + data_name + '_' + surf_hemi + '_' + tag + '.pt', map_location=device))
            csrvcv2.eval()

    # Obtain ctab for annotations
    ctab = get_ctab_from_dataset(config)

    # Start testing
    testset = SegDataset(config=config, data_usage='test')
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)

    for idx, data in enumerate(testloader):
        volume_in, seg_gt, subid = data
        subid = str(subid[0])
        volume_in = volume_in.to(device)

        # Predict segmentation
        with torch.no_grad():
            seg_out = segnet(volume_in)
            seg_pred = torch.argmax(seg_out, dim=1)[0]

            if surf_hemi == 'lh':
                seg = (seg_pred == 1).cpu().numpy()
            elif surf_hemi == 'rh':
                seg = (seg_pred == 2).cpu().numpy()
            elif surf_hemi == 'none':
                print('Skipping surface creation, evaluating segmentation only')
                exit()

        # Extract initial surface
        v_in, f_in = seg2surf(seg, data_name, sigma=0.5, alpha=16, level=0.8, n_smooth=2)

        # Save initial surface
        if test_type == 'init':
            mesh_init = trimesh.Trimesh(v_in, f_in)
            mesh_init.export(init_dir + 'init_' + data_name + '_' + surf_hemi + '_' + subid + '.obj')

        # Predict WM surface
        if test_type in ['pred', 'eval']:
            with torch.no_grad():
                v_in = torch.Tensor(v_in).unsqueeze(0).to(device)
                f_in = torch.LongTensor(f_in).unsqueeze(0).to(device)

                # WM surface prediction and class prediction using CSRVCV2
                csrvcv2.set_data(v_in, volume_in)
                v_wm_pred, class_pred = odeint(csrvcv2, v_in, t=T, method=solver, options=dict(step_size=step_size))[-1]

                # Inflate and smooth to create initial GM surface
                v_gm_in = v_wm_pred.clone()
                for i in range(2):
                    v_gm_in = laplacian_smooth(v_gm_in, f_in, lambd=1.0)
                    n_in = compute_normal(v_gm_in, f_in)
                    v_gm_in += 0.002 * n_in

                # GM surface prediction using CSRVCV2
                csrvcv2.set_data(v_gm_in, volume_in)
                v_gm_pred, _ = odeint(csrvcv2, v_gm_in, t=T, method=solver, options=dict(step_size=step_size / 2))[-1]

            # Convert tensors to numpy arrays
            v_wm_pred = v_wm_pred[0].cpu().numpy()
            f_wm_pred = f_in[0].cpu().numpy()
            v_gm_pred = v_gm_pred[0].cpu().numpy()
            f_gm_pred = f_in[0].cpu().numpy()
            class_pred = torch.argmax(class_pred, dim=1).cpu().numpy()

            # Map the surface coordinates from [-1, 1] to their original space
            v_wm_pred, f_wm_pred = process_surface_inverse(v_wm_pred, f_wm_pred, data_name)
            v_gm_pred, f_gm_pred = process_surface_inverse(v_gm_pred, f_gm_pred, data_name)

        # Save predicted surfaces in FreeSurfer format
        if test_type == 'pred':
            # Save WM surface
            nib.freesurfer.io.write_geometry(result_dir + data_name + '_' + surf_hemi + '_' + subid + '.white',
                                             v_wm_pred, f_wm_pred)
            # Save GM surface
            nib.freesurfer.io.write_geometry(result_dir + data_name + '_' + surf_hemi + '_' + subid + '.pial',
                                             v_gm_pred, f_gm_pred)

            # Save the annotations with class predictions
            nib.freesurfer.write_annot(result_dir + data_name + '_' + surf_hemi + '_' + subid + '.annot',
                                       class_pred, ctab.cpu().numpy(), None)
