import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
import trimesh
from data.preprocess import process_volume, process_surface
from util.mesh import laplacian_smooth, compute_normal

from pytorch3d.structures import Meshes

# ----------------------------
#  for segmentation
# ----------------------------

# ----------------------------
#  for surface reconstruction
# ----------------------------

class BrainData():
    """
    v_in: vertices of input surface
    f_in: faces of ground truth surface
    v_gt: vertices of input surface
    f_gt: faces of ground truth surface
    """
    def __init__(self, volume, v_in, v_gt, f_in, f_gt):
        self.v_in = torch.Tensor(v_in)
        self.f_in = torch.LongTensor(f_in)
        self.v_gt = torch.Tensor(v_gt)
        self.f_gt = torch.LongTensor(f_gt)
        self.volume = torch.from_numpy(volume)
    
    def getBrain(self):
        return self.volume, self.v_in, self.v_gt, self.f_in, self.f_gt
        
        
class FSAVGBrainDataset(Dataset):
    def __init__(self, config, data_usage='train'):
        """
        Initializes the dataset with configurations and prepares
        a list of subject IDs for lazy loading.
        """
        self.data_dir = os.path.join(config.data_dir, data_usage)
        # Assuming self.data_dir is the directory path containing subject folders and possibly other files
        self.subject_list = sorted([item for item in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, item))])

        self.config = config
        self.data_usage = data_usage
        self.mse_threshold = config.mse_threshold
        
        
    def __len__(self):
        return len(self.subject_list)
    
    def __getitem__(self, idx):
        subid = self.subject_list[idx]
        # Load data for the subject identified by `subid`
        # Similar to the logic you've written inside your loop
        # but adjusted to load data for one subject only
        # For example:
        brain_arr, v_in, v_gt, f_in, f_gt = self._load_surf_data_for_subject(subid,self.config,self.data_usage)
        
        return brain_arr, v_in, v_gt, f_in, f_gt, subid
    
    # Function to calculate MSE
    def _calculate_mse(self,original, noisy):
        return 1000.0*np.mean((original - noisy) ** 2)

    def _load_surf_data_for_subject_baseline(self, subid,config,data_usage):
        """
        data_dir: the directory of your dataset
        init_dir: the directory of created initial surfaces
        data_name: [hcp, adni, ...]
        data_usage: [train, valid, test]
        surf_type: [wm, gm]
        surf_hemi: [lh, rh]
        lambd: weight for Laplacian smoothing
        """
        
        data_dir = config.data_dir
        data_dir = data_dir + data_usage + '/'
        data_name = config.data_name
        init_dir = "v2cc/templates/fsaverage/"
        surf_type = config.surf_type
        surf_hemi = config.surf_hemi
        device = config.device

        rho = config.rho    # 0.002
        lambd = config.lambd
        
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
        
        # ------- wm surface reconstruction ------- 
        if surf_type == 'wm':
            # ------- load input surface ------- 
            # inputs is the initial surface
            
            v_in, f_in = nib.freesurfer.io.read_geometry(os.path.join(init_dir,f'{surf_hemi}.white'))
            
            # ------- load gt surface ------- 
            if data_name == 'hcp':
                # depends on your FreeSurfer version
                v_gt, f_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.white')
            elif data_name == 'adni':
                v_gt, f_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.white')
            elif data_name == 'dhcp':
                if surf_hemi == 'lh':
                    surf_gt = nib.load(data_dir+subid+'/'+subid+'_left_wm.surf.gii')
                    v_gt, f_gt = surf_gt.agg_data('pointset'), surf_gt.agg_data('triangle')
                elif surf_hemi == 'rh':
                    surf_gt = nib.load(data_dir+subid+'/'+subid+'_right_wm.surf.gii')
                    v_gt, f_gt = surf_gt.agg_data('pointset'), surf_gt.agg_data('triangle')
                # apply the affine transformation provided by brain MRI nifti
                v_tmp = np.ones([v_gt.shape[0],4])
                v_tmp[:,:3] = v_gt
                v_gt = v_tmp.dot(np.linalg.inv(brain.affine).T)[:,:3]
            v_gt, f_gt = process_surface(v_gt, f_gt, data_name)
        
        # ------- pial surface reconstruction ------- 
        elif surf_type == 'gm':
            # ------- load input surface ------- 
            # input is fsaverage template surface
            v_in, f_in = nib.freesurfer.io.read_geometry(os.path.join(init_dir,f'{surf_hemi}.pial'))
            
            # ------- load gt surface ------- 
            if data_name == 'hcp':
                v_gt, f_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.pial')
            elif data_name == 'adni':
                v_gt, f_gt = nib.freesurfer.io.read_geometry(data_dir+subid+'/surf/'+surf_hemi+'.pial')
            elif data_name == 'dhcp':
                if surf_hemi == 'lh':
                    surf_gt = nib.load(data_dir+subid+'/'+subid+'_left_pial.surf.gii')
                    v_gt, f_gt = surf_gt.agg_data('pointset'), surf_gt.agg_data('triangle')
                elif surf_hemi == 'rh':
                    surf_gt = nib.load(data_dir+subid+'/'+subid+'_right_pial.surf.gii')
                    v_gt, f_gt = surf_gt.agg_data('pointset'), surf_gt.agg_data('triangle')
                v_tmp = np.ones([v_gt.shape[0],4])
                v_tmp[:,:3] = v_gt
                v_gt = v_tmp.dot(np.linalg.inv(brain.affine).T)[:,:3]
            v_gt, f_gt = process_surface(v_gt, f_gt, data_name)
        
        v_in, f_in = process_surface(v_in, f_in, data_name)
        return BrainData(volume=brain_arr, v_in=v_in, v_gt=v_gt,
                              f_in=f_in, f_gt=f_gt).getBrain()


    def _load_surf_data_for_subject(self, subid,config,data_usage):
        if config.train_type == 'surf':
            return self._load_surf_data_for_subject_baseline(subid,config,data_usage)
        else:
            assert False