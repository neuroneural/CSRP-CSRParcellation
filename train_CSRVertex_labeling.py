import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from data.vc_dataloader2 import CSRVertexLabeledDataset  # Ensure this matches your data loader path
from model.csrvertexclassification import CSRVCNet

from util.mesh import compute_dice

import logging
import os
import csv
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from config import load_config
from plyfile import PlyData, PlyElement
from data.preprocess import process_surface_inverse
from scipy.spatial import cKDTree

from model.csrvcv3 import CSRVCV3  # Updated import

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

def save_mesh_with_annotations(mesh, labels, save_path, color_map, data_name='hcp'):
    
    verts, faces = process_surface_inverse(mesh.verts_packed().squeeze().cpu().numpy(), mesh.faces_packed().squeeze().cpu().numpy(), data_name)
    assert verts.shape[0] != 1 
    assert faces.shape[0] != 1 
    invalid_mask = (labels < 0) | (labels >= color_map.size(1))
    
    if invalid_mask.any():
        print(f"Invalid labels found: {labels[invalid_mask]}")
        labels[invalid_mask] = 0  # Assign a default valid label

    labels = labels.squeeze().long().cpu().numpy()
    assert labels.shape[0] != 1
    vertex_colors = color_map[0, labels, :].cpu().numpy()
    vertex_colors = vertex_colors.squeeze()

    vertices = np.array([(*verts[i], *vertex_colors[i]) for i in range(len(verts))],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    faces = np.array([(faces[i],) for i in range(len(faces))],
                      dtype=[('vertex_indices', 'i4', (3,))])

    vertex_element = PlyElement.describe(vertices, 'vertex')
    face_element = PlyElement.describe(faces, 'face')

    PlyData([vertex_element, face_element], text=True).write(save_path)#replace with freeview compatible stl file and annotation file.

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

def visualize_and_save_mesh(csrvcnet, dataloader, result_dir, device, config, epoch):
    for idx, data in enumerate(dataloader):
        volume_in, v_gt, f_gt, labels, subid, color_map = data
        
        volume_in = volume_in.to(device).float()
        v_gt = v_gt.to(device)
        f_gt = f_gt.to(device)
        labels = labels.to(device)
        
        csrvcnet.set_data(v_gt, volume_in, f=f_gt)
        if config.model_type == 'csrvc' and config.version == '3':
                # No initial_state or features_in needed
                # Integrate over time
                _ = csrvcnet(None, v_gt) #deformation not being trained here. 
                logits = csrvcnet.get_class_logits()
                logits = logits.unsqueeze(0)
        else:    
            logits = csrvcnet(v_gt)
        
        assert logits.ndim == 3, f"Expected 3 dimensions, but got {logits.ndim} dimensions."
        assert logits.shape[0] == 1, f"Expected 1 patient {logits.shape} shape."

        preds = torch.argmax(logits, dim=2)
        # print('preds.shape',preds.shape)
        assert preds.ndim == 2, f"Expected 3 dimensions, but got {preds.ndim} dimensions."
        assert preds.shape[0] == 1, f"Expected 1 patient {preds.shape} shape."

        preds = preds.squeeze(0)
        mesh = Meshes(verts=v_gt, faces=f_gt)
        save_path = os.path.join(result_dir, f"annotated_mesh_gtpred_{subid[0]}_{config.surf_hemi}_{config.surf_type}_layers{config.gnn_layers}_epoch{epoch}.ply")
        save_mesh_with_annotations(mesh, preds, save_path, color_map)
        print(f"Saved predicted annotated mesh for subject {subid[0]} to {save_path}")
        save_path = os.path.join(result_dir, f"annotated_mesh_gtfs_{subid[0]}_{config.surf_hemi}_{config.surf_type}_layers{config.gnn_layers}_epoch{epoch}.ply")
        save_mesh_with_annotations(mesh, labels, save_path, color_map)
        print(f"Saved freesurfer gt annotated mesh for subject {subid[0]} to {save_path}") 

def train_surfvc(config):
    """
    Training script for CSRVCNet for vertex classification.
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
    visualize = config.visualize.lower() == 'yes'
    
    n_epochs = config.n_epochs
    start_epoch = config.start_epoch
    n_samples = config.n_samples
    lr = config.lr
    
    C = config.dim_h     # hidden dimension of features
    K = config.kernel_size    # kernel / cube size
    Q = config.n_scale    # multi-scale input

    # Get number of classes based on atlas
    num_classes = get_num_classes(config.atlas)
    if num_classes == 0:
        raise ValueError(f"Unsupported atlas: {config.atlas}")

    # create log file
    log_filename = f"{model_dir}/model_vertex_classification_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}"

    if config.gnn == 'gat':
        use_gcn = False
        log_filename += f"_heads{config.gat_heads}"
    elif config.gnn == 'gcn':
        use_gcn = True

    log_filename += ".log"

    logging.basicConfig(filename=log_filename, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s')

    # --------------------------
    # initialize models
    # --------------------------
    logging.info("initialize model ...")
    
    print('csrvc version ', config.version)
    
    use_pytorch3d_normal = config.use_pytorch3d_normal != 'no'
    
    print("config.model_type, config.version")
    print(config.model_type,config.version)
    if config.model_type == 'csrvc' and config.version == '3':
        csrvcnet = CSRVCV3(dim_h=C,
                            kernel_size=K,
                            n_scale=Q,
                            sf=config.sf,
                            gnn_layers=config.gnn_layers,
                            use_gcn=use_gcn,
                            gat_heads=config.gat_heads,
                            num_classes=num_classes).to(device)
    elif config.model_type == 'csrvc':
        assert False, "sanity check"
        csrvcnet = CSRVCNet(dim_h=C, kernel_size=K, n_scale=Q,
                       gnn_layers=config.gnn_layers,
                       use_gcn=use_gcn,
                       gat_heads=config.gat_heads,
                       num_classes=num_classes,
                       use_pytorch3d=use_pytorch3d_normal
                       ).to(device)
    else:
        assert False, "your config arguments don't match this file."
    
    model_path = None
    
    if config.model_file:
        print('loading model', config.model_file)
        print('hemi', config.surf_hemi)
        print('surftype', config.surf_type)
        
        start_epoch = int(config.start_epoch)
        model_path = os.path.join(config.model_dir, config.model_file)
    
    if model_path and os.path.isfile(model_path):
        print('device', config.device)
        csrvcnet.load_state_dict(torch.load(model_path, map_location=torch.device(config.device)))
        print(f"Model loaded from {model_path}")
    else:
        print("No model file provided or file does not exist. Starting from scratch.")
    
    print('start epoch', start_epoch)
    optimizer = optim.Adam(csrvcnet.parameters(), lr=lr)
    patience = 0
    if config.patience != "standard":
        try:
            patience = int(config.patience)
        except:
            print("patience should either be standard (no scheduler) or an int >=0")
    else:
        print("scheduler is standard and will never step")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)

    # --------------------------
    # load dataset
    # --------------------------
    logging.info("load dataset ...")
    trainset = CSRVertexLabeledDataset(config, 'train')  # Ensure your data loader is correct
    validset = CSRVertexLabeledDataset(config, 'valid')  # Ensure your data loader is correct

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)
    
    # --------------------------
    # training
    # --------------------------
    
    logging.info("start training ...")
    for epoch in tqdm(range(start_epoch, n_epochs + 1)):
        avg_loss = []
        subs = 0
        for idx, data in enumerate(trainloader):
            break
            volume_in, v_in, f_in, labels, subid, color_map = data  # Ensure this matches your data loader output

            optimizer.zero_grad()

            volume_in = volume_in.to(device).float()
            v_in = v_in.to(device)
            f_in = f_in.to(device)
            labels = labels.to(device)  # Ensure labels are moved to the device
            csrvcnet.set_data(v_in, volume_in, f=f_in)  # Set the input data
                
            if config.model_type == 'csrvc' and config.version == '3':
                # No initial_state or features_in needed
                # Integrate over time
                _ = csrvcnet(None, v_in) #deformation not being trained here. 
                logits = csrvcnet.get_class_logits()
                logits = logits.unsqueeze(0)#it appears i'm missing a dimension of logits, probably a trivial one representing the batch that is never used
            else:
                logits = csrvcnet(v_in)  # Forward pass
            print('logits.shape',logits.shape)
            assert logits.ndim == 3, f"Expected 3 dimensions, but got {logits.ndim} dimensions."
            assert logits.shape[0] == 1, f"Expected 1 patient {logits.shape} shape."

            # Reshape logits to match the shape required for CrossEntropyLoss
            logits = logits.permute(0, 2, 1)  # [batch_size, num_vertices, num_classes] -> [batch_size, num_classes, num_vertices]
            
            # Ensure labels are within the valid range
            if torch.any(labels < 0) or torch.any(labels >= num_classes):
                print(f"Invalid label detected in batch {idx} of epoch {epoch}")
                print(f"Labels range: {labels.min()} to {labels.max()}")
                continue  # Skip this batch

            loss = nn.CrossEntropyLoss()(logits, labels)  # Calculate classification loss
            
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        logging.info('epoch:{}, loss:{}'.format(epoch, np.mean(avg_loss)))
        
        if epoch == start_epoch or epoch == n_epochs or epoch % 10 == 0:
            logging.info('-------------validation--------------')
            with torch.no_grad():
                valid_error = []
                valid_dice_scores = []  # List to store dice scores
                exclude_classes = [4] if config.atlas == 'aparc'or config.atlas == 'DKTatlas40' else [] #exclude non cortex, but include medial wall

                for idx, data in enumerate(validloader):
                    volume_in, v_in, f_in, labels, subid, color_map = data  # Ensure this matches your data loader output

                    volume_in = volume_in.to(device).float()
                    v_in = v_in.to(device)
                    f_in = f_in.to(device)
                    labels = labels.to(device)  # Ensure labels are moved to the device

                    csrvcnet.set_data(v_in, volume_in, f=f_in)  # Set the input data
                    if config.model_type == 'csrvc' and config.version == '3':
                        # No initial_state or features_in needed
                        # Integrate over time
                        _ = csrvcnet(None, v_in) #deformation not being trained here. 
                        logits = csrvcnet.get_class_logits()
                        logits = logits.unsqueeze(0)#it appears i'm missing a dimension of logits, probably a trivial one representing the batch that is never used
                    else:
                        logits = csrvcnet(v_in)  # Forward pass
                    print('logits.shape',logits.shape)
            
                    assert logits.ndim == 3, f"Expected 3 dimensions, but got {logits.ndim} dimensions."
                    assert logits.shape[0] == 1, f"Expected 1 patient {logits.shape} shape."

                    logits = logits.permute(0, 2, 1)  # Reshape logits

                    if torch.any(labels < 0) or torch.any(labels >= num_classes):
                        print(f"Invalid label detected in validation batch {idx} of epoch {epoch}")
                        print(f"Labels range: {labels.min()} to {labels.max()}")
                        continue  # Skip this batch

                    valid_loss = nn.CrossEntropyLoss()(logits, labels).item()
                    valid_error.append(valid_loss)
                    
                    # Calculate Dice score
                    preds = torch.argmax(logits, dim=1)  # Get predicted labels
                    dice_score = compute_dice(preds, labels, num_classes, exclude_classes)
                    valid_dice_scores.append(dice_score)

                if epoch > 1 and epoch % 10 == 0 and config.patience != 'standard':
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(np.mean(valid_error).item())
                    new_lr = optimizer.param_groups[0]['lr']
                    if old_lr != new_lr:
                        print(f"Learning rate was adjusted from {old_lr} to {new_lr}")
                    else:
                        print("Learning rate was not adjusted.")
                
                logging.info('epoch:{}, validation error:{}, validation dice:{}'.format(epoch, np.mean(valid_error), np.mean(valid_dice_scores)))
                logging.info('-------------------------------------')
                # Log to CSV
                csv_log_path = os.path.join(model_dir, f"training_log_vertex_classification_{tag}.csv")
                fieldnames = ['surf_hemi', 'surf_type', 'version', 'epoch', 'training_loss', 'validation_error', 'validation_dice', 'gnn', 'gnn_layers', 'gat_heads']

                if not os.path.exists(csv_log_path):
                    with open(csv_log_path, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                avg_training_loss = np.mean(avg_loss)
                with open(csv_log_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if config.gnn == 'gat':
                        writer.writerow({
                            'surf_hemi': surf_hemi,
                            'surf_type': surf_type,
                            'version': config.version,
                            'epoch': epoch,
                            'training_loss': avg_training_loss,  # Include training loss here
                            'validation_error': np.mean(valid_error),
                            'validation_dice': np.mean(valid_dice_scores),
                            'gnn': config.gnn,
                            'gnn_layers': config.gnn_layers,
                            'gat_heads': config.gat_heads
                        })
                    elif config.gnn == 'gcn':
                        writer.writerow({
                            'surf_hemi': surf_hemi,
                            'surf_type': surf_type,
                            'version': config.version,
                            'epoch': epoch,
                            'training_loss': avg_training_loss,  # Include training loss here
                            'validation_error': np.mean(valid_error),
                            'validation_dice': np.mean(valid_dice_scores),
                            'gnn': config.gnn,
                            'gnn_layers': config.gnn_layers,
                            'gat_heads': 'NA'
                        })

                # Call the visualization method if needed
                if visualize:
                    visualize_and_save_mesh(csrvcnet, validloader, config.result_dir, device, config, epoch)

        # save model checkpoints 
        if epoch == start_epoch or epoch == n_epochs or epoch % 10 == 0:
            if config.gnn == 'gat':
                model_filename = f"model_vertex_classification_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}_heads{config.gat_heads}_{epoch}epochs.pt"
            elif config.gnn == 'gcn':
                model_filename = f"model_vertex_classification_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}_{epoch}epochs.pt"
            elif config.gnn == 'baseline':
                model_filename = f"model_vertex_classification_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_{epoch}epochs.pt"
            else:
                assert False, 'update naming conventions for model file name'
            
            torch.save(csrvcnet.state_dict(), os.path.join(model_dir, model_filename))
    
    if config.gnn == 'gat':
        final_model_filename = f"model_vertex_classification_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}_heads{config.gat_heads}.pt"
    elif config.gnn == 'gcn':
        final_model_filename = f"model_vertex_classification_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}.pt"
    elif config.gnn == 'baseline':
        final_model_filename = f"model_vertex_classification_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}.pt"
    else:
        assert False, 'update naming conventions for model file name'
    
    torch.save(csrvcnet.state_dict(), os.path.join(model_dir, final_model_filename))
    print('saving meshes')
    # Save visualizations if needed
            
if __name__ == '__main__':
    mp.set_start_method('spawn')
    config = load_config()
    if config.train_type == 'surfvc':
        train_surfvc(config)
