import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys

from data.dataloader import SegDataset
from model.net import Unet
from model.csrfusionnet import CSRFnet
from model.csrfusionnetv2 import CSRFnetV2
from model.csrfusionnetv3 import CSRFnetV3
from model.csrfusionnetv4 import CSRFnetV4
from model.csrfusionnetv5 import CSRFnetV5
from model.net import CortexODE
from skimage import measure
import nibabel as nib
import logging
from torchdiffeq import odeint_adjoint as odeint
from config import load_config
from scipy.ndimage import distance_transform_cdt as cdt
from skimage.measure import marching_cubes
from skimage.measure import label as compute_cc
from skimage.filters import gaussian
from util.mesh import laplacian_smooth
from util.tca import topology
import trimesh

from data.preprocess import process_surface_inverse

def seg2surf(seg, device, data_name='hcp', sigma=0.5, alpha=16, level=0.8, n_smooth=2):
    """
    Extract the surface based on the segmentation.
    
    seg: input segmentation
    sigma: standard deviation of guassian blurring
    alpha: threshold for obtaining boundary of topology correction
    level: extracted surface level for Marching Cubes
    n_smooth: iteration of Laplacian smoothing
    """
    
    # ------ connected components checking ------ 
    cc, nc = compute_cc(seg, connectivity=2, return_num=True)
    cc_id = 1 + np.argmax(np.array([np.count_nonzero(cc == i) for i in range(1, nc+1)]))
    seg = (cc == cc_id).astype(np.float64)

    # ------ generate signed distance function ------ 
    sdf = -cdt(seg) + cdt(1 - seg)
    sdf = sdf.astype(float)
    sdf = gaussian(sdf, sigma=sigma)

    # ------ topology correction ------
    topo_correct = topology()
    sdf_topo = topo_correct.apply(sdf, threshold=alpha)

    # ------ marching cubes ------
    v_mc, f_mc, _, _ = measure.marching_cubes(-sdf_topo, level=-level, method='lorensen')
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

def save_mesh(verts, faces, path, data_name):
    verts, faces = process_surface_inverse(verts, faces, data_name)
    assert verts.shape[1] == 3, f'verts {verts.shape}'
    assert faces.shape[1] == 3, f'faces {faces.shape}'
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(path + '.stl')
    nib.freesurfer.io.write_geometry(path, verts, faces)

def evaluate(config, data_usage):
    """Evaluation for cortical surface reconstruction."""
    
    # --------------------------
    # Load configuration
    # --------------------------
    model_dir = config.model_dir
    data_name = config.data_name
    device = config.device
    tag = config.tag
    surf_type = config.surf_type
    surf_hemi = config.surf_hemi
    model_type = config.model_type
    version = config.version
    result_dir = config.parc_init_dir  # Use the specified directory for initial surfaces

    logging.basicConfig(filename=model_dir + 'evaluation_' + data_name + '_' + tag + '.log',
                        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # --------------------------
    # Load dataset
    # --------------------------
    logging.info(f"Load {data_usage} dataset ...")
    dataset = SegDataset(config=config, data_usage=data_usage)  # Use SegDataset for the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # --------------------------
    # Initialize segmentation model to create initial surfaces
    # --------------------------
    logging.info("Initialize segmentation model ...")
    segnet = Unet(c_in=1, c_out=3).to(device)
    segnet.load_state_dict(torch.load(config.segmentation_model_path))  # specify the path to the trained segmentation model

    # --------------------------
    # Initialize surface reconstruction model
    # --------------------------
    logging.info("Initialize surface reconstruction model ...")
    gnn_layers=0
    assert model_type != 'csrf' #used for deformation
    if model_type == 'csrvc':
        gnn_layers=config.gnn_layers
    
    if  model_type == 'csrvc' and version == '3':
        cortexode = CSRFnetV3(dim_h=config.dim_h, kernel_size=config.kernel_size, n_scale=config.n_scale,
                              sf=config.sf,
                              use_pytorch3d_normal=config.use_pytorch3d_normal,
                              gnn_layers=config.gnn_layers,
                              use_gcn=config.gnn == 'gcn',
                              gat_heads=config.gat_heads
                              ).to(device)
    else:
        assert False, "for debugging/safety"
        cortexode = CortexODE(dim_in=3, dim_h=config.dim_h, kernel_size=config.kernel_size, n_scale=config.n_scale).to(device)
    
    model_path = os.path.join(config.model_dir, config.model_file)
    cortexode.load_state_dict(torch.load(model_path))
    
    T = torch.Tensor([0, 1]).to(device)  # integration time interval for ODE
    
    # --------------------------
    # Evaluation
    # --------------------------
    logging.info(f"Start evaluation on {data_usage} dataset ...")
    
    segnet.eval()
    cortexode.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            volume_in, seg_gt, subid, _aff = data
            subid = str(subid[0])
            volume_in = volume_in.to(device)
            seg_gt = seg_gt.to(device)
            
            # Generate initial surface using seg2surf method
            seg_out = segnet(volume_in)
            seg_pred = torch.argmax(seg_out, dim=1)[0]
            
            if surf_hemi == 'lh':
                seg = (seg_pred == 1).cpu().numpy()  # lh
            elif surf_hemi == 'rh':
                seg = (seg_pred == 2).cpu().numpy()  # rh
            else:
                assert False, "unsupported"
            verts, faces = seg2surf(seg, device, data_name=config.data_name)
            verts = torch.tensor(verts, dtype=torch.float32).to(device)
            faces = torch.tensor(faces, dtype=torch.int64).to(device)
            
            verts = verts.unsqueeze(0)
            faces = faces.unsqueeze(0)
            print('verts,faces', verts.shape, faces.shape)
            
            # Process the initial surface with the surface reconstruction model
            if model_type == 'csrvc' and config.gnn == 'gat':
                cortexode.set_data(verts, volume_in, f=faces)
            else:
                cortexode.set_data(verts, volume_in)  # set the input data

            # Integrate using the ODE solver
            v_out = odeint(cortexode, verts, t=T, method=config.solver, options=dict(step_size=config.step_size))[-1]

            # Save the final reconstructed surface
            v_pred = v_out.squeeze()
            f_pred = faces.squeeze()
            v_pred = v_pred.cpu().numpy()
            f_pred = f_pred.cpu().numpy()
            surf_type_name = 'white' if surf_type == 'wm' else 'pial'
            result_filename = f'{subid}_{surf_type}_{surf_hemi}_source{model_type}_gnnlayers{gnn_layers}_prediction'
            result_path = os.path.join(result_dir, result_filename)
            save_mesh(v_pred, f_pred, result_path, data_name)

            logging.info(f'Saved reconstructed surface for {subid} to {result_path}')

if __name__ == '__main__':
    config = load_config()
    #generate all for training
    # for data_usage in ['train', 'valid', 'test']:
    #     evaluate(config, data_usage)
    
    for data_usage in ['test']:
        evaluate(config, data_usage)
