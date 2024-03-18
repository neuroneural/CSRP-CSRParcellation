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
from util.mesh import laplacian_smooth, compute_normal, compute_mesh_distance, check_self_intersect
from util.tca import topology
from model.net import CortexODE, Unet
from config import load_config

import datetime
import time
import nvidia_smi
from csv import writer
import socket

nvidia_smi.nvmlInit()

deviceCount = nvidia_smi.nvmlDeviceGetCount()

hostname = socket.gethostname()

def write_time2csv(model_name, t_sec=None, subj=None, loading=False, memory=False, percentUsed=None, total=None, free=None, used=None):
    base_path = '/speedrun/'

    # Consolidate filename determination
    if loading:
        filename = 'bm.loading'
    else:
        filename = 'bm.events'
        
    if memory:
        filename += '.memory'

    filename += '.csv'
    filename = base_path + filename

    # Define initial list
    List = [model_name, t_sec, hostname, subj]
    
    # If memory flag is true, append memory related info
    if memory:
        List = [model_name, percentUsed, total, free, used, hostname, subj]

    if not os.path.exists(filename):
        # Create the file
        with open(filename, 'w') as file:
            # Perform any initial operations on the file, if needed
            print("File created.")

    with open(filename, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)


# Helper functions
def printModelSize(model):
    # print(dir(model))
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('\n\n\n\n')
    print('model size: {:.3f}MB'.format(size_all_mb))
    print('\n\n\n\n')


def printSpaceUsage(info_flag = False):
    msgs = ""
    for i in range(deviceCount):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        if info_flag:
            return (100*info.free/info.total), info.total, info.free, info.used
        
        msgs += '\n'
        msgs += "Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used)
        nvidia_smi.nvmlShutdown()

    msgs+="\nMax Memory occupied by tensors: "+ str(torch.cuda.max_memory_allocated(device=None))
    msgs+="\nMax Memory Cached: "+ str(torch.cuda.max_memory_cached(device=None))
    msgs+="\nCurrent Memory occupied by tensors: "+ str(torch.cuda.memory_allocated(device=None))
    msgs+="\nCurrent Memory cached occupied by tensors: "+str(torch.cuda.memory_cached(device=None))
    msgs+="\n"
    return str(msgs)

# initialize topology correction
topo_correct = topology()


print(printSpaceUsage())

def seg2surf(seg,
             data_name='hcp',
             sigma=0.5,
             alpha=16,
             level=0.8,
             n_smooth=2):
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
    cc_id = 1 + np.argmax(np.array([np.count_nonzero(cc == i)\
                                    for i in range(1, nc+1)]))
    seg = (cc==cc_id).astype(np.float64)

    # ------ generate signed distance function ------ 
    sdf = -cdt(seg) + cdt(1-seg)
    sdf = sdf.astype(float)
    sdf = gaussian(sdf, sigma=sigma)

     # ------ topology correction ------
    sdf_topo= topo_correct.apply(sdf, threshold=alpha)

    # ------ marching cubes ------
    v_mc, f_mc, _, _ = marching_cubes(-sdf_topo, level=-level, method='lorensen')
    v_mc = v_mc[:,[2,1,0]].copy()
    f_mc = f_mc.copy()
    D1,D2,D3 = sdf_topo.shape
    D = max(D1,D2,D3)
    v_mc = (2*v_mc - [D3, D2, D1]) / D   # rescale to [-1,1]
    
    # ------ bias correction ------
    # Note that this bias is introduced by FreeSurfer.
    # FreeSurfer changed the size of the input MRI, 
    # but the affine matrix of the MRI was not changed.
    # So this bias is caused by the different between 
    # the original and new affine matrix.
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


if __name__ == '__main__':
    a = datetime.datetime.now()

    # log for GPU utilization
    GPU_msgs = []

    ### Set Stage
    stage = '0 - --load configuration--'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')


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
    rho = config.rho # inflation scale


    ### Set Stage
    stage = '0 - --load models--'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')


    # ------ load models ------
    print('before load segnet model')
    torch.cuda.empty_cache()
    print(printSpaceUsage())
    segnet = Unet(c_in=1, c_out=3).to(device)
    segnet.load_state_dict(torch.load(model_dir+'model_seg_'+data_name+'_'+tag+'.pt'))
    print('after load segnet model')
    print(printSpaceUsage())
    print('segnet')
    printModelSize(segnet)
    if test_type == 'pred' or test_type == 'eval':
        T = torch.Tensor([0,1]).to(device)

        torch.cuda.empty_cache()

        print('before cortexode_wm = CortexODE(...')
        print(printSpaceUsage())
        cortexode_wm = CortexODE(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)
        print('after cortexode_wm = CortexODE(...')
        print(printSpaceUsage())

        torch.cuda.empty_cache()
        print('BEFORE cortexode_gm = CortexODE(...')
        print(printSpaceUsage())
        cortexode_gm = CortexODE(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)
        print('after cortexode_gm = CortexODE(...')
        print(printSpaceUsage())
        cortexode_wm.load_state_dict(torch.load(model_dir+'model_wm_'+data_name+'_'+surf_hemi+'_'+tag+'.pt', map_location=device))
        print('after cortexode_wm.load_state_dict(torch...')
        print(printSpaceUsage())
        cortexode_gm.load_state_dict(torch.load(model_dir+'model_gm_'+data_name+'_'+surf_hemi+'_'+tag+'.pt', map_location=device))
        print('after cortexode_gm.load_state_dict(torch...')
        print(printSpaceUsage())
        cortexode_wm.eval()
        print('after cortexode_wm.eval...')
        print(printSpaceUsage())
        cortexode_gm.eval()
        print('after cortexode_gm.eval...')
        print(printSpaceUsage())
    
    print('wm model')
    printModelSize(cortexode_wm)
    print('gm model')
    printModelSize(cortexode_gm)
    ### Set Stage
    stage = '0 - --start testing--'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')

    # ------ start testing ------
    subject_list = sorted(os.listdir(data_dir))
    print('subject_list', subject_list, len(subject_list))
    if test_type == 'eval':
        assd_wm_all = []
        assd_gm_all = []
        hd_wm_all = []
        hd_gm_all = []
        sif_wm_all = []
        sif_gm_all = []
    print("\n\n\nHow any subjects are getting predicted\n\n\n")
    print(len(subject_list))
    b = datetime.datetime.now()
    write_time2csv('CortexODE', t_sec = (b-a).total_seconds(), loading=True)
    percentUsed,total,free,used = printSpaceUsage(info_flag=True)
    write_time2csv('CortexODE',percentUsed=percentUsed, total=total, free=free, used=used,loading=True,memory=True)
    for i in tqdm(range(len(subject_list))):
        a = datetime.datetime.now()
        subid = subject_list[i]

        # ------- load brain MRI ------- 
        if data_name == 'hcp' or data_name == 'adni':
            brain = nib.load(data_dir+subid+'/mri/orig.mgz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 255.).astype(np.float32)
        elif data_name == 'dhcp':
            brain = nib.load(data_dir+subid+'/'+subid+'_T2w.nii.gz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 20).astype(np.float16)
        brain_arr = process_volume(brain_arr, data_name)
        volume_in = torch.Tensor(brain_arr).unsqueeze(0).to(device)

        # ------- predict segmentation -------
        print('before predict segmentation')

        torch.cuda.empty_cache()
        print(printSpaceUsage())
        with torch.no_grad():
            seg_out = segnet(volume_in)
            seg_pred = torch.argmax(seg_out, dim=1)[0]
            if surf_hemi == 'lh':
                seg = (seg_pred==1).cpu().numpy()  # lh
            elif surf_hemi == 'rh':
                seg = (seg_pred==2).cpu().numpy()  # rh

        print('after predict segmentation')
        print(printSpaceUsage())
        # ------- extract initial surface ------- 
        v_in, f_in = seg2surf(seg, data_name, sigma=0.5,
                              alpha=16, level=0.8, n_smooth=2)

        # ------- save initial surface ------- 
        if test_type == 'init':
            mesh_init = trimesh.Trimesh(v_in, f_in)
            mesh_init.export(init_dir+'init_'+data_name+'_'+surf_hemi+'_'+subid+'.obj')

        # ------- predict cortical surfaces ------- 
        if test_type == 'pred' or test_type == 'eval':
            with torch.no_grad():
                torch.cuda.empty_cache()
                print(printSpaceUsage())
                v_in = torch.Tensor(v_in).unsqueeze(0).to(device)
                f_in = torch.LongTensor(f_in).unsqueeze(0).to(device)
                
                # wm surface
                print('before wm predict-------')
                torch.cuda.empty_cache()
                print(printSpaceUsage())
                cortexode_wm.set_data(v_in, volume_in)
                v_wm_pred = odeint(cortexode_wm, v_in, t=T, method=solver,
                                   options=dict(step_size=step_size))[-1]
                print('after wm predict------')
                print(printSpaceUsage())
                v_gm_in = v_wm_pred.clone()
                print("before # inflate and smooth")
                print(printSpaceUsage())

                # inflate and smooth
                for i in range(2):
                    v_gm_in = laplacian_smooth(v_gm_in, f_in, lambd=1.0)
                    n_in = compute_normal(v_gm_in, f_in)
                    v_gm_in += 0.002 * n_in
                
                print("before # pial surface")
                torch.cuda.empty_cache()
                print(printSpaceUsage())
                # pial surface
                cortexode_gm.set_data(v_gm_in, volume_in)
                v_gm_pred = odeint(cortexode_gm, v_gm_in, t=T, method=solver,
                                   options=dict(step_size=step_size/2))[-1]  # divided by 2 to reduce SIFs
                print("after # pial surface")
                print(printSpaceUsage())
            v_wm_pred = v_wm_pred[0].cpu().numpy()
            f_wm_pred = f_in[0].cpu().numpy()
            v_gm_pred = v_gm_pred[0].cpu().numpy()
            f_gm_pred = f_in[0].cpu().numpy()
            # map the surface coordinate from [-1,1] to its original space
            v_wm_pred, f_wm_pred = process_surface_inverse(v_wm_pred, f_wm_pred, data_name)
            v_gm_pred, f_gm_pred = process_surface_inverse(v_gm_pred, f_gm_pred, data_name)

        # ------- save predictde surfaces ------- 
        if test_type == 'pred':
            ### save mesh to .obj or .stl format by Trimesh
            # mesh_wm = trimesh.Trimesh(v_wm_pred, f_wm_pred)
            # mesh_gm = trimesh.Trimesh(v_gm_pred, f_gm_pred)
            # mesh_wm.export(result_dir+'wm_'+data_name+'_'+surf_hemi+'_'+subid+'.stl')
            # mesh_gm.export(result_dir+'gm_'+data_name+'_'+surf_hemi+'_'+subid+'.obj')

            # save the surfaces in FreeSurfer format
            nib.freesurfer.io.write_geometry(result_dir+data_name+'_'+surf_hemi+'_'+subid+'.white',
                                             v_wm_pred, f_wm_pred)
            nib.freesurfer.io.write_geometry(result_dir+data_name+'_'+surf_hemi+'_'+subid+'.pial',
                                             v_gm_pred, f_gm_pred)
            
        # ------- load ground truth surfaces ------- 
        if test_type == 'eval':
            if data_name == 'hcp':
                v_wm_gt, f_wm_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.white.deformed')
                v_gm_gt, f_gm_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.pial.deformed')
            elif data_name == 'adni':
                v_wm_gt, f_wm_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.white')
                v_gm_gt, f_gm_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.pial')
            elif data_name == 'dhcp':
                if surf_hemi == 'lh':
                    surf_wm_gt = nib.load(data_dir+subid+'/'+subid+'_left_wm.surf.gii')
                    surf_gm_gt = nib.load(data_dir+subid+'/'+subid+'_left_pial.surf.gii')
                    v_wm_gt, f_wm_gt = surf_wm_gt.agg_data('pointset'), surf_wm_gt.agg_data('triangle')
                    v_gm_gt, f_gm_gt = surf_gm_gt.agg_data('pointset'), surf_gm_gt.agg_data('triangle')
                elif surf_hemi == 'rh':
                    surf_wm_gt = nib.load(data_dir+subid+'/'+subid+'_right_wm.surf.gii')
                    surf_gm_gt = nib.load(data_dir+subid+'/'+subid+'_right_pial.surf.gii')
                    v_wm_gt, f_wm_gt = surf_wm_gt.agg_data('pointset'), surf_wm_gt.agg_data('triangle')
                    v_gm_gt, f_gm_gt = surf_gm_gt.agg_data('pointset'), surf_gm_gt.agg_data('triangle')

                # apply the affine transformation provided by brain MRI nifti
                v_tmp = np.ones([v_wm_gt.shape[0],4])
                v_tmp[:,:3] = v_wm_gt
                v_wm_gt = v_tmp.dot(np.linalg.inv(brain.affine).T)[:,:3]
                v_tmp = np.ones([v_gm_gt.shape[0],4])
                v_tmp[:,:3] = v_gm_gt
                v_gm_gt = v_tmp.dot(np.linalg.inv(brain.affine).T)[:,:3]

        # ------- evaluation -------
        if test_type == 'eval':
            v_wm_pred = torch.Tensor(v_wm_pred).unsqueeze(0).to(device)
            f_wm_pred = torch.LongTensor(f_wm_pred).unsqueeze(0).to(device)
            v_gm_pred = torch.Tensor(v_gm_pred).unsqueeze(0).to(device)
            f_gm_pred = torch.LongTensor(f_gm_pred).unsqueeze(0).to(device)

            v_wm_gt = torch.Tensor(v_wm_gt).unsqueeze(0).to(device)
            f_wm_gt = torch.LongTensor(f_wm_gt.astype(np.float32)).unsqueeze(0).to(device)
            v_gm_gt = torch.Tensor(v_gm_gt).unsqueeze(0).to(device)
            f_gm_gt = torch.LongTensor(f_gm_gt.astype(np.float32)).unsqueeze(0).to(device)

            # compute ASSD and HD
            assd_wm, hd_wm = compute_mesh_distance(v_wm_pred, v_wm_gt, f_wm_pred, f_wm_gt)
            assd_gm, hd_gm = compute_mesh_distance(v_gm_pred, v_gm_gt, f_gm_pred, f_gm_gt)
            if data_name == 'dhcp':  # the resolution is 0.7
                assd_wm = 0.7*assd_wm
                assd_gm = 0.7*assd_gm
                hd_wm = 0.7*hd_wm
                hd_gm = 0.7*hd_gm
            assd_wm_all.append(assd_wm)
            assd_gm_all.append(assd_gm)
            hd_wm_all.append(hd_wm)
            hd_gm_all.append(hd_gm)

            ### compute percentage of self-intersecting faces
            ### uncomment below if you have installed torch-mesh-isect
            ### https://github.com/vchoutas/torch-mesh-isect
            # sif_wm_all.append(check_self_intersect(v_wm_pred, f_wm_pred, collisions=20))
            # sif_gm_all.append(check_self_intersect(v_gm_pred, f_gm_pred, collisions=20))
            sif_wm_all.append(0)
            sif_gm_all.append(0)


        ### Set Stage
        stage = '0 - --END--'
        msgs = printSpaceUsage()
        GPU_msgs.append(stage + msgs + '\n\n\n')

        b = datetime.datetime.now()
        write_time2csv('CortexODE', t_sec = (b-a).total_seconds())
        percentUsed,total,free,used = printSpaceUsage(info_flag=True)
        write_time2csv('CortexODE',memory=True, percentUsed=percentUsed,total=total,free=free,used=used)
        #---------------writing time to a file-----------------
        # List that we want to add as a new row
        #write_time2csv("CortexODE", t_sec, loading = False)
        # Open our existing CSV file in append mode
        # Create a file object for this file
        


        #---------------end of writing time to a file-----------------

        for msg in GPU_msgs:
            print(msg)
        #exit()



    # ------- report the final results ------- 
    if test_type == 'eval':
        print('======== wm ========')
        print('assd mean:', np.mean(assd_wm_all))
        print('assd std:', np.std(assd_wm_all))
        print('hd mean:', np.mean(hd_wm_all))
        print('hd std:', np.std(hd_wm_all))
        print('sif mean:', np.mean(sif_wm_all))
        print('sif std:', np.std(sif_wm_all))
        print('======== gm ========')
        print('assd mean:', np.mean(assd_gm_all))
        print('assd std:', np.std(assd_gm_all))
        print('hd mean:', np.mean(hd_gm_all))
        print('hd std:', np.std(hd_gm_all))
        print('sif mean:', np.mean(sif_gm_all))
        print('sif std:', np.std(sif_gm_all))
