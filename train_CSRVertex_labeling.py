import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from data.vc_dataloader2 import CSRVertexLabeledDataset  # Ensure this matches your data loader path
from model.csrvertexclassification import CSRVCNet
from data.datautil import decode_names

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
import nibabel as nib
from model.csrvcv3 import CSRVCV3  # Updated import

import torch.nn.functional as F

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


def save_mesh_with_annotations(verts, faces, labels, ctab, save_path_fs, data_name='hcp'):
    # Convert tensors to numpy arrays and ensure correct shapes
    verts = verts.squeeze()
    faces = faces.squeeze()
    print("Original verts shape:", verts.shape)
    # Assert that verts should have shape (V, 3)
    assert verts.dim() == 2 and verts.shape[1] == 3, "Verts should have shape (V, 3)"
    verts = verts.squeeze().cpu().numpy()
    print("Processed verts shape after squeeze and numpy conversion:", verts.shape)
    # Assert that verts now have shape (V, 3)
    assert verts.ndim == 2 and verts.shape[1] == 3, "Processed verts should have shape (V, 3)"
    
    print("Original faces shape:", faces.shape)
    # Assert that faces should have shape (F, 3)
    assert faces.dim() == 2 and faces.shape[1] == 3, "Faces should have shape (F, 3)"
    faces = faces.squeeze().long().cpu().numpy()
    print("Processed faces shape after squeeze, long, and numpy conversion:", faces.shape)
    # Assert that faces now have shape (F, 3)
    assert faces.ndim == 2 and faces.shape[1] == 3, "Processed faces should have shape (F, 3)"
    
    # Process the surface if needed
    verts, faces = process_surface_inverse(verts, faces, data_name)
    print("Verts shape after process_surface_inverse:", verts.shape)
    print("Faces shape after process_surface_inverse:", faces.shape)
    # Assert that verts and faces still have correct shapes
    assert verts.ndim == 2 and verts.shape[1] == 3, "Verts after processing should have shape (V, 3)"
    assert faces.ndim == 2 and faces.shape[1] == 3, "Faces after processing should have shape (F, 3)"
    
    # Process labels
    print("Original labels shape:", labels.shape)
    # Assert that labels should be 1D or 2D with one column
    assert labels.dim() in [1, 2], "Labels should be 1D or 2D tensor"
    labels = labels.squeeze().long().cpu().numpy()
    print("Processed labels shape after squeeze, long, and numpy conversion:", labels.shape)
    # Assert that labels now have shape (V,)
    assert labels.ndim == 1 and labels.shape[0] == verts.shape[0], "Labels should have shape (V,)"
    
    # Remap labels of class 4 to -1
    labels[labels == 4] = -1
    print("Labels after remapping class 4 to -1:", np.unique(labels))
    
    # Ensure color table (ctab) is correctly sized
    print("Original ctab shape:", ctab.shape)
    # Assert that ctab should have shape (1, N, 5)
    assert ctab.dim() == 3 and ctab.shape[2] == 5, "ctab should have shape (1, N, 5)"
    ctab = ctab.squeeze().long().cpu().numpy()
    print("Processed ctab shape after squeeze, long, and numpy conversion:", ctab.shape)
    # Assert that ctab now has shape (N, 5)
    assert ctab.ndim == 2 and ctab.shape[1] == 5, "ctab should have shape (N, 5)"
    
    # Decode names for the annotation file
    names = decode_names()
    print("Names decoded for annotation file:", names)
    # Assert that the number of names matches the number of labels in ctab
    assert len(names) == ctab.shape[0], "Number of names must match number of labels in ctab"
    
    # Save the surface geometry
    nib.freesurfer.write_geometry(save_path_fs + '.surf', verts, faces)
    print(f"Saved surface geometry to {save_path_fs}.surf")
    
    # Save the annotation file
    nib.freesurfer.write_annot(save_path_fs + '.annot', 
                               labels,
                               ctab,
                               names, fill_ctab=False)
    print(f"Saved annotation file to {save_path_fs}.annot")


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
    # Turn off gradients for inference and visualization
    with torch.no_grad():  # This disables gradient tracking, saving memory
        for idx, data in enumerate(dataloader):
            volume_in, v_gt, f_gt, labels, subid, color_map = data

            # Move data to device (CPU or GPU)
            volume_in = volume_in.to(device).float()
            v_gt = v_gt.to(device)
            f_gt = f_gt.to(device)
            labels = labels.to(device)

            # Set data for the network
            csrvcnet.set_data(v_gt, volume_in, f=f_gt)

            # Get predictions/logits
            if config.model_type == 'csrvc' and config.version == '3':
                # Deformation not being trained here
                _ = csrvcnet(None, v_gt)  # Perform the forward pass
                logits = csrvcnet.get_class_logits()
                logits = torch.nn.functional.log_softmax(logits, dim=1)
                logits = logits.unsqueeze(0)  # Adjust shape to add batch dimension
            else:
                logits = csrvcnet(v_gt)

            # Ensure the logits have the correct shape
            assert logits.ndim == 3, f"Expected 3 dimensions, but got {logits.ndim} dimensions."
            assert logits.shape[0] == 1, f"Expected 1 sample in the batch, but got shape {logits.shape[0]}."

            # Get the predicted classes (argmax over classes)
            preds = torch.argmax(logits, dim=2)

            # Ensure the predictions have the correct shape
            assert preds.ndim == 2, f"Expected 2 dimensions for predictions, but got {preds.ndim} dimensions."
            assert preds.shape[0] == 1, f"Expected 1 sample in the batch, but got shape {preds.shape[0]}."

            # Squeeze the batch dimension
            preds = preds.squeeze(0)

            # Create mesh object for saving
            mesh = Meshes(verts=v_gt, faces=f_gt)

            # Save predicted annotated mesh
            save_path = os.path.join(result_dir, f"annotated_mesh_gtpred_{subid[0]}_{config.surf_hemi}_{config.surf_type}_layers{config.gnn_layers}_epoch{epoch}.ply")
            save_mesh_with_annotations(v_gt, f_gt, preds, color_map, save_path, data_name='hcp')
            print(f"Saved predicted annotated mesh for subject {subid[0]} to {save_path}")

            # Save ground truth annotated mesh
            save_path = os.path.join(result_dir, f"annotated_mesh_gtfs_{subid[0]}_{config.surf_hemi}_{config.surf_type}_layers{config.gnn_layers}_epoch{epoch}.ply")
            save_mesh_with_annotations(v_gt, f_gt, labels, color_map, save_path, data_name='hcp')
            print(f"Saved FreeSurfer ground truth annotated mesh for subject {subid[0]} to {save_path}")
 

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
    #visualize = config.visualize.lower() == 'yes'
    visualize = False
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
        if epoch == start_epoch or epoch == n_epochs or epoch % 50 == 0: 
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
            
if __name__ == '__main__':
    mp.set_start_method('spawn')
    config = load_config()
    if config.train_type == 'surfvc':
        train_surfvc(config)
