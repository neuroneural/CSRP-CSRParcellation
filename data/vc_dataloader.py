import torch
from torch.utils.data import Dataset

import os
import numpy as np
import nibabel as nib
from data.preprocess import process_volume, process_surface
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
#  for vertex classification
# ----------------------------

class VertexData:
    def __init__(self, brain_arr, v, f, labels, subid, color_map):
        self.brain_arr = torch.Tensor(brain_arr)
        self.v = torch.Tensor(v)
        self.f = torch.LongTensor(f)
        # Convert labels to float32 to avoid byte order issues
        self.labels = torch.from_numpy(labels.astype(np.float32)).long()#converting to float is a workaround for some endian problems
        self.subid = subid
        self.color_map = torch.Tensor(color_map)
    
    def get_data(self):
        return self.brain_arr, self.v, self.f, self.labels, self.subid, self.color_map


class CSRVertexLabeledDataset(Dataset):
    def __init__(self, config, data_usage='train',num_classes=37):
        """
        Initializes the dataset with configurations for lazy loading.
        Args:
            config: Configuration object containing dataset parameters.
            data_usage: Specifies the dataset split to use ('train', 'valid', 'test').
        """
        self.num_classes = num_classes
        self.config = config
        self.data_usage = data_usage
        self.data_dir = os.path.join(config.data_dir, data_usage)
        self.subject_list = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.subject_list)
    
    def __getitem__(self, idx):
        subid = self.subject_list[idx]
        brain_arr, v, f, labels = self._load_vertex_labeled_data_for_subject(subid, self.config, self.data_usage)
        color_map = self._generate_color_map()
        return VertexData(brain_arr, v, f, labels, subid, color_map).get_data()
    
    def _generate_color_map(self):
        """
        Generate a unique color map for the given number of classes using Matplotlib's color maps.
        Args:
            num_classes (int): The number of unique classes.
        Returns:
            np.array: An array of shape (num_classes, 3) with RGB colors.
        """
        cmap = plt.get_cmap('tab10')  # 'tab10' is a good starting point for distinct colors
        colors = cmap(np.linspace(0, 1, self.num_classes))[:, :3]  # Get RGB parts
        colors = (colors * 255).astype(int)  # Scale to 0-255 for RGB
        return colors

    def _load_vertex_labeled_data_for_subject(self, subid, config, data_usage):
        """
        Load brain MRI data, surface meshes, and vertex annotations for a given subject.
        """
        data_dir = os.path.join(config.data_dir, data_usage)
        data_name = config.data_name
        surf_type = 'gm'  # Only handle grey matter
        surf_hemi = config.surf_hemi
        atlas_dir = os.path.join(config.data_dir, data_usage, subid, 'label')  # Directory where the FreeSurfer atlas files are stored
        
        # Load brain MRI
        if data_name == 'hcp' or data_name == 'adni':
            brain = nib.load(os.path.join(data_dir, subid, 'mri', 'orig.mgz'))
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 255.).astype(np.float32)
        elif data_name == 'dhcp':
            brain = nib.load(os.path.join(data_dir, subid, f'{subid}_T2w.nii.gz'))
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 20).astype(np.float16)
        brain_arr = process_volume(brain_arr, data_name)
        
        # Load gt surface for gm (grey matter)
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

        # Load vertex annotations from the FreeSurfer atlas
        labels = self._load_vertex_labels(subid, atlas_dir, surf_hemi, config.atlas)
        
        return brain_arr, v, f, labels
    
    def _load_vertex_labels(self, subid, atlas_dir, surf_hemi, atlas):
        """
        Load vertex annotations from the FreeSurfer atlas for the given subject.
        Args:
            subid: Subject ID.
            atlas_dir: Directory where the FreeSurfer atlas files are stored.
            surf_hemi: Hemisphere (e.g., 'lh' or 'rh').
            atlas: The selected atlas for the annotations.
        """
        annot_file = os.path.join(atlas_dir, f'{surf_hemi}.{atlas}.annot')
        labels, _, _ = nib.freesurfer.io.read_annot(annot_file)
        print('max labels',np.max(labels))
        print('min labels',np.min(labels))
        labels = labels + 1#added to shift from -1 to 0
        return labels
