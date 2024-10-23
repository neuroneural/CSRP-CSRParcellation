import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
from data.preprocess import process_volume, process_surface, process_surface_inverse
from sklearn.neighbors import KDTree
import pytorch3d.structures
import pytorch3d
import re
import random

from scipy.spatial import cKDTree


def chamfer_distance(v1, v2):
    
    kdtree1 = cKDTree(v1)
    kdtree2 = cKDTree(v2)

    distances1, _ = kdtree1.query(v2)
    distances2, _ = kdtree2.query(v1)

    return np.mean(distances1) + np.mean(distances2)


class VertexData:
    def __init__(self, brain_arr, v_gt, f_gt, labels, subid, ctab, v_in=None, f_in=None, nearest_labels=None, mask=None):
        # Add print statements and assertions
        assert brain_arr is not None, "brain_arr is None"
        assert v_gt is not None, "v_gt is None"
        assert f_gt is not None, "f_gt is None"
        assert labels is not None, "labels are None"
        assert subid is not None, "subid is None"
        assert ctab is not None, "ctab is None"
        
        self.brain_arr = torch.Tensor(brain_arr)
        self.v_gt = torch.Tensor(v_gt)
        self.f_gt = torch.from_numpy(f_gt.astype(np.float32)).long()# converting to float is a workaround for some endian problems
        self.labels = torch.from_numpy(labels.astype(np.float32)).long()
        self.subid = subid
        self.ctab = torch.from_numpy(ctab.astype(np.float32)).long()
        self.v_in = torch.Tensor(v_in) if v_in is not None else None
        self.f_in = torch.from_numpy(f_in.astype(np.float32)).long() if f_in is not None else None
        self.nearest_labels = torch.from_numpy(nearest_labels.astype(np.float32)).long() if nearest_labels is not None else None
        self.mask = torch.Tensor(mask) if mask is not None else None
    def get_data(self):
        return self.brain_arr, self.v_gt, self.f_gt, self.labels, self.subid, self.ctab, self.v_in, self.f_in, self.nearest_labels, self.mask

class CSRVertexLabeledDatasetV3(Dataset):
    def __init__(self, config, data_usage='train', num_classes=37):
        self.num_classes = num_classes
        self.config = config
        self.data_usage = data_usage
        self.data_dir = os.path.join(config.data_dir, data_usage)
        self.subject_list = sorted([re.sub(r'\D', '', str(item)) for item in os.listdir(self.data_dir) if len(re.sub(r'\D', '', str(item))) > 1 and os.path.isdir(os.path.join(self.data_dir, item))])
    def __len__(self):
        return len(self.subject_list)
    
    def __getitem__(self, idx):
        subid = f'{self.subject_list[idx]}'
        subid = re.sub(r'\D', '', subid)
        brain_arr, v_gt, f_gt, labels, ctab = self._load_vertex_labeled_data_for_subject(subid, self.config, self.data_usage)
        v_in, f_in = self._load_input_mesh(subid)
        
        assert v_in is not None, f"v_in is None,{self.config.parc_init_dir}"
        assert f_in is not None, "f_in is None"
        
        normals = self._calculate_normals(v_in, f_in)
        gt_normals = self._calculate_normals(v_gt, f_gt)
        
        assert v_gt.ndim == 2,f'v_gt.shape'
        kdtree = KDTree(v_gt)
        assert v_in.ndim == 2,f'v_in.shape'
        distances, indices = kdtree.query(v_in, k=1)
        assert indices.shape[0]==v_in.shape[0],f'{indices.shape},{v_in.shape}'
        nearest_labels = labels[indices.flatten()]
        assert nearest_labels.shape[0] == v_in.shape[0],f'{nearest_label.shape},{v_in.shape}'
        # mask = self._create_normal_mask(normals, gt_normals[indices.flatten()])
        mask = None
        # Add print statements and assertions for new format variables
        assert brain_arr is not None, "brain_arr is None"
        assert v_gt is not None, "v_gt is None"
        assert f_gt is not None, "f_gt is None"
        assert labels is not None, "labels are None"
        assert subid is not None, "subid is None"
        assert ctab is not None, "ctab is None"
        assert v_in is not None, "v_in is None"
        assert f_in is not None, "f_in is None"
        assert nearest_labels is not None, "nearest_labels are None"
        # assert mask is not None, "mask is None"
        
        return VertexData(brain_arr, v_gt, f_gt, labels, subid, ctab, v_in, f_in, nearest_labels, mask).get_data()
    
    def _load_vertex_labeled_data_for_subject(self, subid, config, data_usage):
        data_dir = os.path.join(config.data_dir, data_usage)
        data_name = config.data_name
        surf_type = config.surf_type  # surf_type from config
        surf_hemi = config.surf_hemi  # surf_hemi from config
        atlas_dir = os.path.join(config.data_dir, data_usage, subid, 'label')
        try:
            if data_name == 'hcp' or data_name == 'adni':
                brain = nib.load(os.path.join(data_dir, subid, 'mri', 'orig.mgz'))
                brain_arr = brain.get_fdata()
                brain_arr = (brain_arr / 255.).astype(np.float32)
            elif data_name == 'dhcp':
                brain = nib.load(os.path.join(data_dir, subid, f'{subid}_T2w.nii.gz'))
                brain_arr = brain.get_fdata()
                brain_arr = (brain_arr / 20).astype(np.float16)
            brain_arr = process_volume(brain_arr, data_name)
                
            assert brain_arr is not None, f"Failed to load brain_arr for subject {subid}"
            #TODO: NEED TO ADD LOADING FOR WHITE SURFACES
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
            
            v, f = process_surface(v_gt, f_gt, data_name)
            assert v is not None and f is not None, f"Failed to load vertices or faces for subject {subid}"
        
            labels, ctab = self._load_vertex_labels(atlas_dir, surf_hemi, config.atlas)
            assert labels is not None, f"Failed to load labels for subject {subid}"
            assert ctab is not None, f"Failed to load ctab for subject {subid}"
            return brain_arr, v, f, labels, ctab
        except Exception as e:
            print(f"Error loading data for subject {subid}: {e}")
            return None, None, None, None, None


    def _load_vertex_labels(self, atlas_dir, surf_hemi, atlas):
        try:
            if self.config.atlas == 'aparc':
                annot_file = os.path.join(atlas_dir, f'{surf_hemi}.{atlas}.annot')
            elif self.config.atlas == 'DKTatlas40':
                annot_file = os.path.join(atlas_dir, f'{surf_hemi}.aparc.{atlas}.annot')
            else:
                assert False, "label mapping not supported yet"
            labels, ctab, _names = nib.freesurfer.io.read_annot(annot_file)
            if self.config.atlas == 'aparc' or self.config.atlas == 'DKTatlas40':
                labels[labels == -1] = 4 #non cortex, see ctab file
            else:
                assert False, "label mapping not supported yet"
            if ctab.shape[0] < len(_names): 
                raise ValueError(f"Colormap does not have enough colors for the classes.")
            return labels, ctab
        except Exception as e:
            print(f"Error loading vertex labels: {e}")
            return None, None

    def _load_input_mesh(self, subid):
        try:
            surf_type = self.config.surf_type  # surf_type from config
            surf_hemi = self.config.surf_hemi  # surf_hemi from config
            if self.config.model_type in ['csrvc']:
                input_mesh_dir = os.path.join(self.config.parc_init_dir,self.data_usage,surf_type,surf_hemi)
            else:
                input_mesh_dir = self.config.parc_init_dir

            # Get all available GNN layer files for the subject
            if self.config.model_type in ['csrvc']:
                pattern = f'{self.config.data_name}_{surf_hemi}_{subid}_gnnlayers\d_{surf_type}_pred_epoch\d'
            elif self.config.model_type2 !='baseline':#legacy
                pattern = f'{subid}_{surf_type}_{surf_hemi}_source{self.config.model_type2}_gnnlayers\d_prediction'
            else:
                self.config.gnn_layers=0 
                pattern = f'{subid}_{surf_type}_{surf_hemi}_source{self.config.model_type2}_gnnlayers\d_prediction'
            available_files = [f for f in os.listdir(input_mesh_dir) if re.match(pattern, f)]
            if not available_files:
                raise FileNotFoundError(f"No input mesh files found for subject {subid} with surf_type {surf_type} and surf_hemi {surf_hemi}.")

            # Extract the layer numbers and randomly select one
            gnn_layers = [int(re.search(r'gnnlayers(\d)', f).group(1)) for f in available_files]#This requires one hyperparam per folder!
            epoch = [int(re.search(r'epoch(\d+)', f).group(1)) for f in available_files]#This requires one hyperparam per folder!
            selected_layer = random.choice(gnn_layers)
            selected_epoch = random.choice(epoch)
            if self.config.model_type in ['csrvc']:
                filename = f'{self.config.data_name}_{surf_hemi}_{subid}_gnnlayers{selected_layer}_{surf_type}_pred_epoch{selected_epoch}.surf'
            else:    
                filename = f'{subid}_{surf_type}_{surf_hemi}_source{self.config.model_type2}_gnnlayers{selected_layer}_prediction'
            input_mesh_path = os.path.join(input_mesh_dir, filename)
            
            v_in, f_in = nib.freesurfer.io.read_geometry(input_mesh_path)
            v_in, f_in = process_surface(v_in,f_in,self.config.data_name)
            assert v_in is not None, f"input path: {input_mesh_path}"
            return v_in, torch.from_numpy(f_in.astype(np.float32)).long().numpy()  # needed workaround.
        except Exception as e:
            print(f"Error loading input mesh for subject {subid}: {e}")
            return None, None

    def _calculate_normals(self, v, f):
        verts = torch.tensor(v, dtype=torch.float32)
        faces = torch.tensor(f, dtype=torch.int64)
        mesh = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
        normals = mesh.verts_normals_packed()  # Correct function
        return normals.numpy()

    def _create_normal_mask(self, normals, gt_normals, threshold=60):
        cos_sim = np.einsum('ij,ij->i', normals, gt_normals) / (
                    np.linalg.norm(normals, axis=1) * np.linalg.norm(gt_normals, axis=1))
        angles = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * (180 / np.pi)
        mask = (angles <= threshold).astype(np.float32)
        return mask
