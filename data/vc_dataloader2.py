import torch
from torch.utils.data import Dataset

import os
import numpy as np
import nibabel as nib
from data.preprocess import process_volume, process_surface
import matplotlib.pyplot as plt
import re

class VertexData:
    def __init__(self, brain_arr, v, f, labels, subid, ctab,names):
        self.brain_arr = torch.Tensor(brain_arr)
        self.v = torch.Tensor(v)
        self.f = torch.LongTensor(f)
        self.labels = torch.from_numpy(labels.astype(np.float32)).long()  # converting to float is a workaround for some endian problems
        self.subid = subid
        self.ctab = torch.from_numpy(ctab.astype(np.float32)).long() #doesn't need gpu
        # print('names',names)
        # print('type',type(names))
        # self.names = torch.tensor(names) #doesn't need gpu
    
    def get_data(self):
        return self.brain_arr, self.v, self.f, self.labels, self.subid, self.ctab#, self.names


class CSRVertexLabeledDataset(Dataset):
    def __init__(self, config, data_usage='train', num_classes=37):
        self.num_classes = num_classes
        self.config = config
        self.data_usage = data_usage
        self.data_dir = os.path.join(config.data_dir, data_usage)
        self.subject_list = sorted([re.sub(r'\D', '', str(item)) for item in os.listdir(self.data_dir) if len(re.sub(r'\D', '', str(item))) > 1 and os.path.isdir(os.path.join(self.data_dir, item))])

    def __len__(self):
        return len(self.subject_list)
    
    def __getitem__(self, idx):
        subid = self.subject_list[idx]
        brain_arr, v, f, labels, ctab,names = self._load_vertex_labeled_data_for_subject(subid, self.config, self.data_usage)
        return VertexData(brain_arr, v, f, labels, subid, ctab,names).get_data()
    
    # def _generate_color_map(self, subid):
    #     """
    #     Generate a color map based on the specified atlas' .ctab file.
    #     Args:
    #         subid (str): Subject ID to locate the specific .ctab file.
    #     Returns:
    #         np.array: An array of shape (num_classes, 3) with RGB colors.
    #     """
    #     atlas = self.config.atlas
    #     ctab_file = None
    #     if atlas == 'aparc':
    #         ctab_file = os.path.join(self.data_dir, subid, 'label', f'aparc.annot.ctab')
    #     elif atlas == 'aparc.a2009s':
    #         ctab_file = os.path.join(self.data_dir, subid, 'label', f'aparc.annot.a2009s.ctab')
    #     elif atlas == 'aparc.DKTatlas40':
    #         ctab_file = os.path.join(self.data_dir, subid, 'label', f'aparc.annot.DKTatlas40.ctab')
    #     elif atlas == 'BA':
    #         ctab_file = os.path.join(self.data_dir, subid, 'label', f'BA.ctab')
    #     else:
    #         raise ValueError("Atlas coloring not supported")

    #     # Read color map from the .ctab file
    #     colors = []
    #     with open(ctab_file, 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             if not line.startswith('#'):
    #                 parts = line.split()
    #                 if len(parts) >= 5:
    #                     r, g, b = int(parts[2]), int(parts[3]), int(parts[4])
    #                     colors.append([r, g, b])
    #     colors = np.array(colors, dtype=np.int32)
    #     if colors.shape[0] < self.num_classes:
    #         raise ValueError(f"Colormap in {ctab_file} does not have enough colors for {self.num_classes} classes.")
    #     return colors[:self.num_classes]
    
    def _load_vertex_labeled_data_for_subject(self, subid, config, data_usage):
        data_dir = os.path.join(config.data_dir, data_usage)
        data_name = config.data_name
        surf_type = 'gm'
        surf_hemi = config.surf_hemi
        atlas_dir = os.path.join(config.data_dir, data_usage, subid, 'label')

        if data_name == 'hcp' or data_name == 'adni':
            brain = nib.load(os.path.join(data_dir, subid, 'mri', 'orig.mgz'))
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 255.).astype(np.float32)
        elif data_name == 'dhcp':
            brain = nib.load(os.path.join(data_dir, subid, f'{subid}_T2w.nii.gz'))
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 20).astype(np.float16)
        brain_arr = process_volume(brain_arr, data_name)

        if data_name == 'hcp':
            v, f = nib.freesurfer.io.read_geometry(os.path.join(data_dir, subid, 'surf', f'{surf_hemi}.pial.deformed'))
        elif data_name == 'adni':
            v, f = nib.freesurfer.io.read_geometry(os.path.join(data_dir, subid, 'surf', f'{surf_hemi}.pial'))
        elif data_name == 'dhcp':
            surf_gt = nib.load(os.path.join(data_dir, subid, f'{subid}_{surf_hemi}_pial.surf.gii'))
            v, f = surf_gt.agg_data('pointset'), surf_gt.agg_data('triangle')
            v_tmp = np.ones([v.shape[0], 4])
            v_tmp[:, :3] = v
            v = v_tmp.dot(np.linalg.inv(brain.affine).T)[:, :3]
        v, f = process_surface(v, f, data_name)

        labels, ctab,names = self._load_vertex_labels(atlas_dir, surf_hemi, config.atlas)

        return brain_arr, v, f, labels, ctab,names
    

    def _load_vertex_labels(self, atlas_dir, surf_hemi, atlas):
        if self.config.atlas == 'aparc':
            annot_file = os.path.join(atlas_dir, f'{surf_hemi}.{atlas}.annot')
        elif self.config.atlas == 'DKTatlas40':
            annot_file = os.path.join(atlas_dir, f'{surf_hemi}.aparc.{atlas}.annot')
        else:
            assert False, "label mapping not supported yet"
        labels, ctab, names = nib.freesurfer.io.read_annot(annot_file)
        if self.config.atlas == 'aparc' or self.config.atlas == 'DKTatlas40':
            labels[labels == -1] = 4 #non cortex, see ctab file
        else:
            assert False, "label mapping not supported yet"# Convert ctab to RGB format, excluding the -1 label color if necessary
        # color_map = ctab[:, :3]
        
        # # Ensure there are enough colors for the classes excluding -1
        # if color_map.shape[0] < len(_names): 
        #     raise ValueError(f"Colormap does not have enough colors for the classes.")

        return labels, ctab, names


