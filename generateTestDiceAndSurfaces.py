import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from data.vc_dataloader2 import CSRVertexLabeledDataset  # Ensure this matches your data loader path
from model.csrvertexclassification import CSRVCNet

import logging
import os
import csv
import torch.multiprocessing as mp
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from config import load_config
from plyfile import PlyData, PlyElement
from data.preprocess import process_surface_inverse
from scipy.spatial import cKDTree
import nibabel as nib

def chamfer_distance(v1, v2):
    kdtree1 = cKDTree(v1)
    kdtree2 = cKDTree(v2)

    distances1, _ = kdtree1.query(v2)
    distances2, _ = kdtree2.query(v1)

    return np.mean(distances1) + np.mean(distances2)

def get_num_classes(atlas):
    atlas_num_classes = {
        'aparc': 36,            # 34 regions + 1 for unknown +1 for corpus callosum mapped to 4 from -1
        'a2009s': 83,     # 82 regions + 1 for unknown
        'DKTatlas40': 36, # 40 regions + 1 for unknown
        'BA': 53,               # 52 regions + 1 for unknown
        # Add more atlases as needed
    }
    return atlas_num_classes.get(atlas, 0)

def save_mesh_with_annotations(verts, faces, labels, save_path_fs, data_name='hcp'):
    verts = verts.squeeze().cpu().numpy()
    faces = faces.squeeze().squeeze().long().cpu().numpy()
    verts, faces = process_surface_inverse(verts, faces, data_name)

    labels = labels.squeeze().long().cpu().numpy()
    
    nib.freesurfer.write_geometry(save_path_fs + '.surf', verts, faces)
    nib.freesurfer.write_annot(save_path_fs + '.annot', labels)

def compute_dice(pred, target, num_classes, exclude_classes=[]):
    dice_scores = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for i in range(num_classes):
        if i in exclude_classes:
            continue
        
        pred_i = (pred == i)
        target_i = (target == i)
        
        intersection = np.sum(pred_i & target_i)
        union = np.sum(pred_i) + np.sum(target_i)
        
        if union == 0:
            dice_score = 1.0
        else:
            dice_score = 2. * intersection / union
        dice_scores.append(dice_score)
        
    return np.mean(dice_scores)

def evaluate_model(config):
    """
    Evaluation script for CSRVCNet for vertex classification.
    """

    # --------------------------
    # load configuration
    # --------------------------
    model_dir = config.model_dir
    data_name = config.data_name
    surf_type = config.surf_type
    surf_hemi = config.surf_hemi
    device = config.device
    tag = config.tag
    result_dir = config.result_dir  # Directory to save annotations and surfaces

    n_epochs = config.n_epochs
    start_epoch = config.start_epoch
    lr = config.lr
    
    C = config.dim_h     # hidden dimension of features
    K = config.kernel_size    # kernel / cube size
    Q = config.n_scale    # multi-scale input

    # Get number of classes based on atlas
    num_classes = get_num_classes(config.atlas)
    if num_classes == 0:
        raise ValueError(f"Unsupported atlas: {config.atlas}")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # --------------------------
    # initialize models
    # --------------------------
    logging.info("initialize model ...")
    
    use_pytorch3d_normal = config.use_pytorch3d_normal != 'no'
    
    if config.model_type == 'csrvc':
        csrvcnet = CSRVCNet(dim_h=C, kernel_size=K, n_scale=Q,
                       gnn_layers=config.gnn_layers,
                       use_gcn=config.gnn == 'gcn',
                       gat_heads=config.gat_heads,
                       num_classes=num_classes,
                       use_pytorch3d=use_pytorch3d_normal
                       ).to(device)
    else:
        assert False, "your config arguments don't match this file."
    
    model_path = os.path.join(config.model_dir, config.model_file)
    print('model_path',model_path)
    if os.path.isfile(model_path):
        print('device', config.device)
        csrvcnet.load_state_dict(torch.load(model_path, map_location=torch.device(config.device)))
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    csrvcnet.eval()  # Set model to evaluation mode

    # --------------------------
    # load test dataset
    # --------------------------
    logging.info("load test dataset ...")
    testset = CSRVertexLabeledDataset(config, 'test')  # Ensure your data loader is correct
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    
    # --------------------------
    # evaluation
    # --------------------------
    logging.info("start evaluation ...")
    with torch.no_grad():
        test_dice_scores = []
        exclude_classes = [-1,4]

        for idx, data in enumerate(testloader):
            volume_in, v_in, f_in, labels, subid, color_map = data  # Ensure this matches your data loader output

            volume_in = volume_in.to(device).float()
            v_in = v_in.to(device)
            f_in = f_in.to(device)
            labels = labels.to(device)  # Ensure labels are moved to the device

            csrvcnet.set_data(v_in, volume_in, f=f_in)  # Set the input data

            logits = csrvcnet(v_in)  # Forward pass
            logits = logits.permute(0, 2, 1)  # Reshape logits

            if torch.any(labels < 0) or torch.any(labels >= num_classes):
                print(f"Invalid label detected in test batch {idx}")
                print(f"Labels range: {labels.min()} to {labels.max()}")
                continue  # Skip this batch

            preds = torch.argmax(logits, dim=1)  # Get predicted labels
            
            dice_score = compute_dice(preds, labels, num_classes, exclude_classes)
            test_dice_scores.append(dice_score)

            # Save the predicted and ground truth annotated meshes in both PLY and FreeSurfer formats
            
            pred_save_path_fs = os.path.join(result_dir, f"annotated_mesh_pred_{subid[0]}_{surf_hemi}_{surf_type}_layers{config.gnn_layers}")
            gt_save_path_fs = os.path.join(result_dir, f"annotated_mesh_gt_{subid[0]}_{surf_hemi}_{surf_type}_layers{config.gnn_layers}")
                                        
            save_mesh_with_annotations(v_in,f_in, preds, pred_save_path_fs, data_name='hcp')
            save_mesh_with_annotations(v_in,f_in, labels, gt_save_path_fs, data_name='hcp')
            
            print(f"Saved predicted annotated mesh for subject {subid[0]} to {pred_save_path_fs} and {pred_save_path_fs}.surf/.annot")
            print(f"Saved ground truth annotated mesh for subject {subid[0]} to {pred_save_path_fs} and {gt_save_path_fs}.surf/.annot")

        avg_test_dice_score = np.mean(test_dice_scores)
        print(f"Average test Dice score: {avg_test_dice_score}")

        # Save results to CSV
        csv_log_path = os.path.join(model_dir, f"test_results_vertex_classification_{tag}.csv")
        fieldnames = ['subject_id', 'gnn_layers', 'surf_hemi', 'surf_type', 'test_dice_score']

        if not os.path.exists(csv_log_path):
            with open(csv_log_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(csv_log_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for idx, data in enumerate(testloader):
                volume_in, v_in, f_in, labels, subid, color_map = data  # Ensure this matches your data loader output
                writer.writerow({
                    'subject_id': subid[0],
                    'gnn_layers': config.gnn_layers,
                    'surf_hemi': surf_hemi,
                    'surf_type': surf_type,
                    'test_dice_score': test_dice_scores[idx]
                })

if __name__ == '__main__':
    mp.set_start_method('spawn')
    config = load_config()
    evaluate_model(config)
