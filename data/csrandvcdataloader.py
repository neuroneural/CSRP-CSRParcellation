import torch
from torch.utils.data import Dataset
import re
import os
import numpy as np
import nibabel as nib
from data.preprocess import process_volume, process_surface
from pytorch3d.structures import Meshes
import trimesh

# **Add the necessary imports for smoothing and normals**
from util.mesh import laplacian_smooth, compute_normal

class SegData():
    def __init__(self, vol, seg, subj_id,aff):
        self.vol = torch.Tensor(vol)
        self.seg = torch.Tensor(seg)
        self.subj_id = subj_id
        self.aff = aff
    
    def getSeg(self):
        return self.vol, self.seg, self.subj_id, self.aff

class SegDataset(Dataset):
    def __init__(self, config, data_usage='train'):
        """
        Initializes the dataset with configurations for lazy loading.
        Args:
            config: Configuration object containing dataset parameters.
            data_usage: Specifies the dataset split to use ('train', 'valid', 'test').
        """
        self.config = config
        self.data_usage = data_usage
        self.data_dir = os.path.join(config.data_dir, data_usage)
        # Assuming self.data_dir is the directory path containing subject folders and possibly other files
        self.subject_list = sorted([re.sub(r'\D', '',str(item)) for item in os.listdir(self.data_dir) if len(re.sub(r'\D', '',str(item)))>1 and os.path.isdir(os.path.join(self.data_dir, item))])
        
    def __len__(self):
        return len(self.subject_list)
    
    def __getitem__(self, idx):
        subid = f'{self.subject_list[idx]}'
        subid = re.sub(r'\D', '', subid)

        vol, seg,_,aff = self._load_seg_data_for_subject(subid,self.config,self.data_usage)
        assert subid == _
        return vol, seg, subid, aff
    
    def _load_seg_data_for_subject(self, subid,config,data_usage):
        """
        data_dir: the directory of your dataset
        data_name: [hcp, adni, dhcp, ...]
        data_usage: [train, valid, test]
        """
        
        data_name = config.data_name
        data_dir = config.data_dir
        data_dir = data_dir + data_usage + '/'

        subject_list = sorted(os.listdir(data_dir))
        
        if data_name == 'hcp' or data_name == 'adni':
            brain = nib.load(data_dir+subid+'/mri/orig.mgz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 255.).astype(np.float32)
            brain_arr = process_volume(brain_arr, data_name)
            aff=brain.affine
            
            seg = nib.load(data_dir+subid+'/mri/ribbon.mgz')
            seg_arr = seg.get_fdata()
            seg_arr = process_volume(seg_arr, data_name)[0]
            seg_left = (seg_arr == 2).astype(int)    # left wm
            seg_right = (seg_arr == 41).astype(int)  # right wm

            seg_arr = np.zeros_like(seg_left, dtype=int)  # final label
            seg_arr += 1 * seg_left
            seg_arr += 2 * seg_right
        elif data_name == 'dhcp':
            brain = nib.load(data_dir+subid+'/'+subid+'_T2w.nii.gz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 20).astype(np.float32)
            brain_arr = process_volume(brain_arr, data_name)
            
            # wm_label is the generated segmentation by projecting surface into the volume
            seg_arr = np.load(data_dir+subid+'/'+subid+'_wm_label.npy', allow_pickle=True)
            seg_arr = process_volume(seg_arr, data_name)[0]
            
        return SegData(vol=brain_arr, seg=seg_arr,subj_id=subid,aff=aff).getSeg()


class BrainData():
    """
    Data class to hold all necessary components for training.
    """
    def __init__(self, volume, v_in, v_gt, f_in, f_gt, labels, aff=None, ctab=None, nearest_labels=None, mask=None):
        self.volume = torch.Tensor(volume)
        self.v_gt = torch.Tensor(v_gt)
        self.f_gt = torch.from_numpy(f_gt.astype(np.float32)).long()  # Convert to float32 then to long to handle endian issues
        self.labels = torch.from_numpy(labels.astype(np.float32)).long()  # Convert to float32 then to long to handle endian issues
        self.aff = aff
        self.ctab = torch.from_numpy(ctab.astype(np.float32)).long() if ctab is not None else None  # Convert ctab if provided
        self.v_in = torch.Tensor(v_in) if v_in is not None else None
        self.f_in = torch.from_numpy(f_in.astype(np.float32)).long() if f_in is not None else None  # Convert to float32 then to long to handle endian issues
        self.nearest_labels = torch.from_numpy(nearest_labels.astype(np.float32)).long() if nearest_labels is not None else None
        self.mask = torch.Tensor(mask) if mask is not None else None

    def getBrain(self):
        return self.volume, self.v_in, self.v_gt, self.f_in, self.f_gt, self.labels, self.aff

class BrainDataset(Dataset):
    def __init__(self, config, data_usage='train',affCtab=False):
        """
        Initializes the dataset with configurations and prepares
        a list of subject IDs for lazy loading.
        """
        self.data_dir = os.path.join(config.data_dir, data_usage)
        self.subject_list = sorted([
            item for item in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, item))
        ])
        self.config = config
        self.data_usage = data_usage
        self.mse_threshold = config.mse_threshold
        self.surf_hemi = config.surf_hemi
        self.surf_type = config.surf_type
        self.device = config.device
        self.affCtab = affCtab
    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, idx):
        subid = self.subject_list[idx]

        # Load data for the subject identified by `subid`
        brain_arr, v_in, v_gt, f_in, f_gt, labels, aff, ctab = self._load_surf_data_for_subject(
            subid, self.config, self.data_usage
        )
        if not self.affCtab:
            return torch.tensor(brain_arr).float(), torch.tensor(v_in).float(), torch.tensor(v_gt).float(), torch.tensor(f_in).long(), torch.tensor(f_gt).long(), torch.tensor(labels).long()
        else:
            return torch.tensor(brain_arr).float(), torch.tensor(v_in).float(), torch.tensor(v_gt).float(), torch.tensor(f_in).long(), torch.tensor(f_gt).long(), torch.tensor(labels).long(), torch.tensor(aff).float(), torch.tensor(ctab).long(), subid

    def _load_surf_data_for_subject(self, subid, config, data_usage):
        data_dir = os.path.join(config.data_dir, data_usage)
        data_name = config.data_name
        init_dir = os.path.join(config.init_dir, data_usage)
        surf_type = config.surf_type
        surf_hemi = config.surf_hemi
        device = self.device

        # Load MRI volume
        if data_name in ['hcp', 'adni']:
            brain = nib.load(os.path.join(data_dir, subid, 'mri', 'orig.mgz'))
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 255.).astype(np.float32)
            aff = brain.affine
        elif data_name == 'dhcp':
            brain = nib.load(os.path.join(data_dir, subid, f'{subid}_T2w.nii.gz'))
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 20).astype(np.float32)
            aff = brain.affine
        else:
            raise ValueError(f"Unsupported data_name: {data_name}")
        brain_arr = process_volume(brain_arr, data_name)

        # Load ground truth surface
        v_gt, f_gt = self._load_ground_truth_surface(data_dir, subid, data_name, surf_hemi, surf_type, aff)

        # Load input surface
        v_in, f_in = self._load_input_surface(data_dir, subid, data_name, init_dir, surf_hemi, surf_type, aff)

        # Load labels
        labels, ctab = self._load_vertex_labels(data_dir, subid, config.atlas, surf_hemi)

        # Ensure labels are valid
        if labels is None or len(labels) != v_gt.shape[0]:
            raise ValueError(f"Labels not loaded correctly for subject {subid}")

        return brain_arr, v_in, v_gt, f_in, f_gt, labels, aff, ctab

    def _load_ground_truth_surface(self, data_dir, subid, data_name, surf_hemi, surf_type, aff):
        if data_name == 'hcp':
            if surf_type == 'wm':
                v_gt, f_gt = nib.freesurfer.io.read_geometry(
                    os.path.join(data_dir, subid, 'surf', f'{surf_hemi}.white.deformed')
                )
            elif surf_type == 'gm':
                v_gt, f_gt = nib.freesurfer.io.read_geometry(
                    os.path.join(data_dir, subid, 'surf', f'{surf_hemi}.pial.deformed')
                )
            else:
                raise ValueError(f"Unsupported surf_type: {surf_type}")
        elif data_name == 'adni':
            if surf_type == 'wm':
                v_gt, f_gt = nib.freesurfer.io.read_geometry(
                    os.path.join(data_dir, subid, 'surf', f'{surf_hemi}.white')
                )
            elif surf_type == 'gm':
                v_gt, f_gt = nib.freesurfer.io.read_geometry(
                    os.path.join(data_dir, subid, 'surf', f'{surf_hemi}.pial')
                )
            else:
                raise ValueError(f"Unsupported surf_type: {surf_type}")
        elif data_name == 'dhcp':
            if surf_type == 'wm':
                surf_gt = nib.load(
                    os.path.join(data_dir, subid, f'{subid}_{surf_hemi}_wm.surf.gii')
                )
            elif surf_type == 'gm':
                surf_gt = nib.load(
                    os.path.join(data_dir, subid, f'{subid}_{surf_hemi}_pial.surf.gii')
                )
            else:
                raise ValueError(f"Unsupported surf_type: {surf_type}")
            v_gt, f_gt = surf_gt.agg_data('pointset'), surf_gt.agg_data('triangle')
            # Apply affine transformation
            v_tmp = np.ones([v_gt.shape[0], 4])
            v_tmp[:, :3] = v_gt
            v_gt = v_tmp.dot(np.linalg.inv(aff).T)[:, :3]
        else:
            raise ValueError(f"Unsupported data_name: {data_name}")
        v_gt, f_gt = process_surface(v_gt, f_gt, data_name)
        return v_gt, f_gt

    def _load_input_surface(self, data_dir, subid, data_name, init_dir, surf_hemi, surf_type, aff):
        if surf_type == 'wm':
            # Load initial mesh from init_dir
            init_mesh_path = os.path.join(init_dir, f'init_{data_name}_{surf_hemi}_{subid}.obj')
            # print('init_mesh_path',init_mesh_path)
            if not os.path.isfile(init_mesh_path):
                raise FileNotFoundError(f"Initial mesh not found for subject {subid}")
            mesh_in = trimesh.load(init_mesh_path)
            v_in, f_in = mesh_in.vertices, mesh_in.faces
        elif surf_type == 'gm':
            # Use white matter surface as input
            if data_name == 'hcp':
                v_in, f_in = nib.freesurfer.io.read_geometry(
                    os.path.join(data_dir, subid, 'surf', f'{surf_hemi}.white.deformed')
                )
            elif data_name == 'adni':
                v_in, f_in = nib.freesurfer.io.read_geometry(
                    os.path.join(data_dir, subid, 'surf', f'{surf_hemi}.white')
                )
            elif data_name == 'dhcp':
                surf_in = nib.load(
                    os.path.join(data_dir, subid, f'{subid}_{surf_hemi}_wm.surf.gii')
                )
                v_in, f_in = surf_in.agg_data('pointset'), surf_in.agg_data('triangle')
                # Apply affine transformation
                v_tmp = np.ones([v_in.shape[0], 4])
                v_tmp[:, :3] = v_in
                v_in = v_tmp.dot(np.linalg.inv(aff).T)[:, :3]
            else:
                raise ValueError(f"Unsupported data_name: {data_name}")
            v_in, f_in = process_surface(v_in, f_in, data_name)

            # **Perform inflation and smoothing**
            v_in, f_in = self._inflate_and_smooth(v_in, f_in)
        else:
            raise ValueError(f"Unsupported surf_type: {surf_type}")
        return v_in, f_in

    def _inflate_and_smooth(self, v_in, f_in):
        n_inflate = self.config.n_inflate
        rho = self.config.rho
        lambd = self.config.lambd
        device = self.device
        
        v_in_tensor = torch.Tensor(v_in).unsqueeze(0).to(device)
        f_in_tensor = torch.LongTensor(f_in).unsqueeze(0).to(device)
        for _ in range(n_inflate):
            # **Apply Laplacian smoothing**
            v_in_tensor = laplacian_smooth(v_in_tensor, f_in_tensor, lambd=lambd)
            # **Compute vertex normals**
            n_in = compute_normal(v_in_tensor, f_in_tensor)
            # **Inflate along the normal direction**
            v_in_tensor += rho * n_in
        v_in = v_in_tensor.cpu().numpy()[0]
        f_in = f_in_tensor.cpu().numpy()[0]
        return v_in, f_in

    def _load_vertex_labels(self, data_dir, subid, atlas, surf_hemi):
        atlas_dir = os.path.join(data_dir, subid, 'label')
        try:
            if atlas == 'aparc':
                annot_file = os.path.join(atlas_dir, f'{surf_hemi}.{atlas}.annot')
            elif atlas == 'DKTatlas40':
                annot_file = os.path.join(atlas_dir, f'{surf_hemi}.aparc.{atlas}.annot')
            else:
                raise ValueError("Label mapping not supported yet.")
            labels, ctab, _ = nib.freesurfer.io.read_annot(annot_file)
            # Process labels if necessary
            labels[labels == -1] = 4  # Non-cortex label
            labels = labels.astype(np.float32)  # Convert labels to float32 to handle endian issues
            ctab = ctab.astype(np.float32)  # Convert ctab to float32 to handle endian issues
            return labels, ctab
        except Exception as e:
            print(f"Error loading vertex labels for subject {subid}: {e}")
            return None, None
