import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
from data.preprocess import process_volume, process_surface
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import pytorch3d.ops
import pytorch3d.structures
import re

class VertexData:
    def __init__(self, brain_arr, v, f, labels, subid, color_map, nearest_labels=None, mask=None):
        self.brain_arr = torch.Tensor(brain_arr)
        self.v = torch.Tensor(v)
        self.f = torch.LongTensor(f)
        self.labels = torch.from_numpy(labels.astype(np.float32)).long()
        self.subid = subid
        self.color_map = torch.from_numpy(color_map.astype(np.float32)).long()
        self.nearest_labels = torch.from_numpy(nearest_labels.astype(np.float32)).long() if nearest_labels is not None else None
        self.mask = torch.Tensor(mask) if mask is not None else None
    
    def get_data(self):
        return self.brain_arr, self.v, self.f, self.labels, self.subid, self.color_map, self.nearest_labels, self.mask


class CSRVertexLabeledDataset(Dataset):
    def __init__(self, config, data_usage='train', num_classes=37, new_format=False):
        self.num_classes = num_classes
        self.config = config
        self.data_usage = data_usage
        self.data_dir = os.path.join(config.data_dir, data_usage)
        self.subject_list = sorted([re.sub(r'\D', '',str(item)) for item in os.listdir(self.data_dir) if len(re.sub(r'\D', '',str(item)))>1 and os.path.isdir(os.path.join(self.data_dir, item))])
        self.new_format = new_format

    def __len__(self):
        return len(self.subject_list)
    
    def __getitem__(self, idx):
        subid = self.subject_list[idx]
        brain_arr, v, f, labels, color_map = self._load_vertex_labeled_data_for_subject(subid, self.config, self.data_usage)
        if self.new_format:
            gt_v, gt_f, gt_labels = self._load_ground_truth(subid)
            normals = self._calculate_normals(v, f)
            gt_normals = self._calculate_normals(gt_v, gt_f)
            kdtree = KDTree(gt_v)
            distances, indices = kdtree.query(v, k=1)
            nearest_labels = gt_labels[indices.flatten()]
            mask = self._create_normal_mask(normals, gt_normals[indices.flatten()])
            return VertexData(brain_arr, v, f, labels, subid, color_map, nearest_labels, mask).get_data()
        else:
            return VertexData(brain_arr, v, f, labels, subid, color_map).get_data()

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

        labels, color_map = self._load_vertex_labels(atlas_dir, surf_hemi, config.atlas)

        return brain_arr, v, f, labels, color_map

    def _load_vertex_labels(self, atlas_dir, surf_hemi, atlas):
        annot_file = os.path.join(atlas_dir, f'{surf_hemi}.{atlas}.annot')
        labels, ctab, _names = nib.freesurfer.io.read_annot(annot_file)
        if self.config.atlas=='aparc':
            labels[labels == -1] = 4
        else:
            assert False, "label mapping not supported yet"
        color_map = ctab[:, :3]
        if color_map.shape[0] < len(_names): 
            raise ValueError(f"Colormap does not have enough colors for the classes.")
        return labels, color_map

    def _load_ground_truth(self, subid):
        data_dir = os.path.join(self.config.data_dir, 'ground_truth')
        gt_v, gt_f = nib.freesurfer.io.read_geometry(os.path.join(data_dir, subid, 'surf', 'pial'))
        gt_labels, _, _ = nib.freesurfer.io.read_annot(os.path.join(data_dir, subid, 'label', 'aparc.annot'))
        return gt_v, gt_f, gt_labels

    def _calculate_normals(self, v, f):
        verts = torch.tensor(v, dtype=torch.float32)
        faces = torch.tensor(f, dtype=torch.int64)
        mesh = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
        normals = pytorch3d.ops.verts_normals(mesh)[0]
        return normals.numpy()

    def _create_normal_mask(self, normals, gt_normals, threshold=60):
        cos_sim = np.einsum('ij,ij->i', normals, gt_normals) / (
                    np.linalg.norm(normals, axis=1) * np.linalg.norm(gt_normals, axis=1))
        angles = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * (180 / np.pi)
        mask = (angles <= threshold).astype(np.float32)
        return mask
