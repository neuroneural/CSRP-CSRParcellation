import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from data.vc_dataloader import CSRVertexLabeledDataset  # Ensure this matches your data loader path
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


def get_num_classes(atlas):
    """
    Get the number of classes for the specified atlas.
    Args:
        atlas (str): The name of the atlas.
    Returns:
        int: The number of classes for the atlas.
    """
    atlas_num_classes = {
        'aparc': 37,            # 34 regions + 1 for unknown
        'aparc.a2009s': 83,     # 82 regions + 1 for unknown
        'aparc.DKTatlas40': 41, # 40 regions + 1 for unknown
        'BA': 53,               # 52 regions + 1 for unknown
        # Add more atlases as needed
    }
    return atlas_num_classes.get(atlas, 0)  # Default to 0 if atlas not found

def save_mesh_with_annotations(mesh, labels, save_path, color_map):
    verts = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()
    
    invalid_mask = (labels < 0) | (labels >= color_map.size(1))
    if invalid_mask.any():
        print(f"Invalid labels found: {labels[invalid_mask]}")
        labels[invalid_mask] = 0  # Assign a default valid label

    labels = labels.long().cpu().numpy()

    vertex_colors = color_map[0, labels, :].cpu().numpy()  # Index color_map with labels
    vertex_colors = vertex_colors.squeeze()

    vertices = np.array([(*verts[i], *vertex_colors[i]) for i in range(len(verts))],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    faces = np.array([(faces[i],) for i in range(len(faces))],
                      dtype=[('vertex_indices', 'i4', (3,))])

    vertex_element = PlyElement.describe(vertices, 'vertex')
    face_element = PlyElement.describe(faces, 'face')

    PlyData([vertex_element, face_element], text=True).write(save_path)

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
    
    if config.model_type == 'csrvc':
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
            volume_in, v_in, f_in, labels, subid, color_map = data  # Ensure this matches your data loader output

            optimizer.zero_grad()

            volume_in = volume_in.to(device).float()
            v_in = v_in.to(device)
            f_in = f_in.to(device)
            labels = labels.to(device)  # Ensure labels are moved to the device

            csrvcnet.set_data(v_in, volume_in, f=f_in)  # Set the input data

            logits = csrvcnet(v_in)  # Forward pass

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
                for idx, data in enumerate(validloader):
                    volume_in, v_in, f_in, labels, subid, color_map = data  # Ensure this matches your data loader output

                    volume_in = volume_in.to(device).float()
                    v_in = v_in.to(device)
                    f_in = f_in.to(device)
                    labels = labels.to(device)  # Ensure labels are moved to the device

                    csrvcnet.set_data(v_in, volume_in, f=f_in)  # Set the input data

                    logits = csrvcnet(v_in)  # Forward pass
                    logits = logits.permute(0, 2, 1)  # Reshape logits

                    if torch.any(labels < 0) or torch.any(labels >= num_classes):
                        print(f"Invalid label detected in validation batch {idx} of epoch {epoch}")
                        print(f"Labels range: {labels.min()} to {labels.max()}")
                        continue  # Skip this batch

                    valid_loss = nn.CrossEntropyLoss()(logits, labels).item()
                    valid_error.append(valid_loss)

                if epoch > 1 and epoch % 10 == 0 and config.patience != 'standard':
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(np.mean(valid_error).item())
                    new_lr = optimizer.param_groups[0]['lr']
                    if old_lr != new_lr:
                        print(f"Learning rate was adjusted from {old_lr} to {new_lr}")
                    else:
                        print("Learning rate was not adjusted.")
                
                logging.info('epoch:{}, validation error:{}'.format(epoch, np.mean(valid_error)))
                logging.info('-------------------------------------')
                # Log to CSV
                csv_log_path = os.path.join(model_dir, f"training_log_vertex_classification_{tag}.csv")
                fieldnames = ['surf_hemi', 'surf_type', 'version', 'epoch', 'training_loss', 'validation_error', 'gnn', 'gnn_layers', 'gat_heads']

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
                            'gnn': config.gnn,
                            'gnn_layers': config.gnn_layers,
                            'gat_heads': 'NA'
                        })

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
    if visualize:
        for idx, data in enumerate(validloader):
            volume_in, v_in, f_in, labels, subid, color_map = data
            volume_in = volume_in.to(device).float()
            v_in = v_in.to(device)
            f_in = f_in.to(device)
            labels = labels.to(device)
            csrvcnet.set_data(v_in, volume_in, f=f_in)
            logits = csrvcnet(v_in)
            print('logits.shape', logits.shape)
            preds = torch.argmax(logits, dim=2)  # Ensure correct dimension for argmax
            preds = preds.squeeze(0)  # Remove the batch dimension if present
            mesh = Meshes(verts=v_in, faces=f_in)
            save_path = os.path.join(model_dir, f"annotated_mesh_{subid[0]}.ply")
            save_mesh_with_annotations(mesh, preds, save_path, color_map)
            print(f"Saved annotated mesh for subject {subid[0]} to {save_path}")
            
if __name__ == '__main__':
    mp.set_start_method('spawn')
    config = load_config()
    if config.train_type == 'surfvc':
        train_surfvc(config)
