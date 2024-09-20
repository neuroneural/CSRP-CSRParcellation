import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from data.csrandvcdataloader import BrainDataset
from model.tcsrvc import TCSRVC  # Updated import
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes

import logging
import os
import torch.multiprocessing as mp
from config import load_config
from scipy.spatial import cKDTree

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

def train_surf(config):
    """
    Training TCSRVC for cortical surface reconstruction and classification.
    """
    
    # --------------------------
    # Load configuration
    # --------------------------
    model_dir = config.model_dir
    data_name = config.data_name
    surf_type = config.surf_type
    surf_hemi = config.surf_hemi
    device = config.device
    tag = config.tag
    print('surf_type', surf_type)
    print('surf_hemi', surf_hemi)
    n_epochs = config.n_epochs
    start_epoch = config.start_epoch
    lr = config.lr
    
    C = config.dim_h     # Hidden dimension of features
    K = config.kernel_size    # Kernel / cube size
    Q = config.n_scale    # Multi-scale input
    
    num_classes = config.num_classes  # Number of classes for classification
    
    # Threshold for starting classification loss computation
    classification_loss_threshold = config.classification_loss_threshold  # e.g., 0.04
    
    # Loss weight for classification loss
    classification_loss_weight = config.classification_loss_weight  # e.g., 1.0
    
    # Create log file
    log_filename = os.path.join(model_dir, f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_csrvc_layers{config.gnn_layers}_sf{config.sf}")
    print('log_filename', log_filename)
    
    log_filename += ".log"

    # Configure logging
    logging.basicConfig(filename=log_filename, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s')

    # --------------------------
    # Initialize model
    # --------------------------
    logging.info("Initialize model ...")

    # Initialize the model
    cortexode = TCSRVC(dim_h=C,
                        kernel_size=K,
                        n_scale=Q,
                        sf=config.sf,
                        num_classes=num_classes).to(device)

    # Load model state if a model path is provided
    if config.model_file:
        model_path = os.path.join(config.model_dir, config.model_file)
        if os.path.isfile(model_path):
            cortexode.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            print("No model file provided or file does not exist. Starting from scratch.")

    optimizer = optim.Adam(cortexode.parameters(), lr=lr)

    # --------------------------
    # Load dataset
    # --------------------------
    logging.info("Load dataset ...")
    trainset = BrainDataset(config, 'train')  # Should include labels
    validset = BrainDataset(config, 'valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)

    # Flag to indicate whether to compute classification loss
    compute_classification_loss = False

    # --------------------------
    # Training
    # --------------------------

    logging.info("Start training ...")
    num_time_steps = 10  # Define the number of time steps

    for epoch in tqdm(range(start_epoch, n_epochs + 1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            # Unpack data
            volume_in, v_in, v_gt, f_in, f_gt, labels = data

            optimizer.zero_grad()

            # Move data to device and ensure float dtype
            volume_in = volume_in.to(device).float()
            v_in = v_in.to(device).float()
            v_gt = v_gt.to(device).float()
            f_in = f_in.to(device).long()
            f_gt = f_gt.to(device).long()
            labels = labels.to(device).long()

            # Set initial state and data
            t = torch.arange(num_time_steps).to(device)
            cortexode.set_data(v_in, volume_in, f=f_in, t=t)
            # No initial_state or features_in needed

            # Initialize variables
            v_out = v_in.clone()
            hidden_states = None

            for step in range(num_time_steps):
                # Convert step to tensor
                t_step = torch.tensor([step], device=device, dtype=torch.float32)

                # Forward pass
                dx, class_logits, hidden_states = cortexode(v_out, t=t_step, hidden_states=hidden_states)
                v_out = v_out + dx

                # Update the memory with the latest interactions
                cortexode.block1.memory.update_state(
                    cortexode.block1.src,
                    cortexode.block1.dst,
                    t_edge=t_step.repeat(cortexode.block1.src.size(0)),
                    raw_msg=cortexode.block1.edge_attr,
                )

                # Retain hidden states for the next step
                hidden_states = cortexode.block1.memory.memory.clone()


            # Reconstruction Loss
            if surf_type == 'wm':
                # Chamfer Distance Loss on vertices
                chamfer_loss = 1e3 * chamfer_distance(v_out, v_gt)[0]
                reconstruction_loss = chamfer_loss
            elif surf_type == 'gm':
                # MSE Loss between vertex positions
                mse_loss = 1e3 * nn.MSELoss()(v_out, v_gt)
                reconstruction_loss = mse_loss

            total_loss = reconstruction_loss

            # Check if reconstruction loss is below threshold
            if not compute_classification_loss and reconstruction_loss.item() < classification_loss_threshold:
                compute_classification_loss = True
                print(f"Reconstruction loss below threshold at epoch {epoch}, batch {idx}. Starting classification loss computation.")

            if compute_classification_loss:
                # Perform KDTree nearest neighbor search to assign labels
                v_out_np = v_out.squeeze(0).detach().cpu().numpy()
                v_gt_np = v_gt.squeeze(0).cpu().numpy()

                # Build KDTree from ground truth surface vertices
                if surf_type == 'wm':
                    kd_tree = cKDTree(v_gt_np)

                    # For each predicted vertex, find nearest ground truth vertex
                    distances, indices = kd_tree.query(v_out_np)

                    # Obtain labels from ground truth labels
                    labels_np = labels.squeeze(0).cpu().numpy()
                    predicted_labels = labels_np[indices]
                else:
                    predicted_labels = labels.squeeze(0).cpu().numpy()  # Correspondence can be exploited.

                # Convert labels to tensor
                predicted_labels = torch.from_numpy(predicted_labels).long().to(device)

                # Retrieve classification logits
                class_logits = cortexode.get_class_logits()

                # Compute classification loss
                classification_loss = nn.NLLLoss()(class_logits, predicted_labels)

                # Total Loss (add classification loss)
                total_loss += classification_loss_weight * classification_loss

            avg_loss.append(total_loss.item())
            total_loss.backward()
            optimizer.step()

        logging.info('epoch:{}, loss:{}'.format(epoch, np.mean(avg_loss)))

        if epoch == start_epoch or epoch == n_epochs or epoch % 10 == 0:
            logging.info('-------------validation--------------')
            with torch.no_grad():
                total_valid_error = []
                recon_valid_error = []
                dice_valid_error = []
                classification_valid_error = []
                for idx, data in enumerate(validloader):
                    volume_in, v_in, v_gt, f_in, f_gt, labels = data

                    # Move data to device and ensure float dtype
                    volume_in = volume_in.to(device).float()
                    v_in = v_in.to(device).float()
                    v_gt = v_gt.to(device).float()
                    f_in = f_in.to(device).long()
                    f_gt = f_gt.to(device).long()
                    labels = labels.to(device).long()

                    # Set initial state and data
                    t = torch.arange(num_time_steps).to(device)
                    cortexode.set_data(v_in, volume_in, f=f_in, t=t)

                    # Initialize variables
                    v_out = v_in.clone()
                    hidden_states = None

                    for step in range(num_time_steps):
                        # Convert step to tensor
                        t_step = torch.tensor([step], device=device, dtype=torch.float32)

                        # Forward pass
                        dx, class_logits, hidden_states = cortexode(v_out, t=t_step, hidden_states=hidden_states)
                        v_out = v_out + dx

                        # Update the memory with the latest interactions
                        cortexode.block1.memory.update_state(
                            cortexode.block1.src,
                            cortexode.block1.dst,
                            t_edge=t_step.repeat(cortexode.block1.src.size(0)),
                            raw_msg=cortexode.block1.edge_attr,
                        )

                        # Retain hidden states for the next step
                        hidden_states = cortexode.block1.memory.memory.clone()


                    # Reconstruction Loss
                    if surf_type == 'wm':
                        # Chamfer Distance Loss on vertices
                        chamfer_loss = 1e3 * chamfer_distance(v_out, v_gt)[0]
                        reconstruction_loss = chamfer_loss
                    elif surf_type == 'gm':
                        # MSE Loss between vertex positions
                        mse_loss = 1e3 * nn.MSELoss()(v_out, v_gt)
                        reconstruction_loss = mse_loss

                    total_valid_loss = reconstruction_loss.item()
                    recon_valid_loss = reconstruction_loss.item()
                    dice_score = -1.0  # Unless calculated, make negative since higher is better.
                    if compute_classification_loss:
                        # Perform KDTree nearest neighbor search to assign labels
                        v_out_np = v_out.squeeze(0).detach().cpu().numpy()
                        v_gt_np = v_gt.squeeze(0).cpu().numpy()
                        labels_np = labels.squeeze(0).cpu().numpy()

                        if surf_type == 'wm':
                            kd_tree = cKDTree(v_gt_np)

                            # For each predicted vertex, find nearest ground truth vertex
                            distances, indices = kd_tree.query(v_out_np)

                            # Obtain labels from ground truth labels
                            labels_np = labels.squeeze(0).cpu().numpy()
                            predicted_labels = labels_np[indices]
                        else:
                            predicted_labels = labels.squeeze(0).cpu().numpy()  # Correspondence can be exploited.

                        # Convert labels to tensor
                        predicted_labels = torch.from_numpy(predicted_labels).long().to(device)

                        # Retrieve classification logits
                        class_logits = cortexode.get_class_logits()

                        # Compute classification loss
                        classification_loss = nn.NLLLoss()(class_logits, predicted_labels)
                        exclude_classes = [4] if config.atlas == 'aparc' or config.atlas == 'DKTatlas40' else []
                        predicted_classes = torch.argmax(class_logits, dim=1)
                        dice_score = compute_dice(predicted_classes, predicted_labels, num_classes, exclude_classes)

                        total_valid_loss += classification_loss_weight * classification_loss.item()
                        dice_valid_error.append(dice_score)
                        classification_valid_error.append(classification_loss_weight * classification_loss.item())
                    total_valid_error.append(total_valid_loss)
                    recon_valid_error.append(recon_valid_loss)

                logging.info('epoch:{}, total validation error:{}'.format(epoch, np.mean(total_valid_error)))
                logging.info('epoch:{}, reconstruction validation error:{}'.format(epoch, np.mean(recon_valid_error)))
                logging.info('epoch:{}, dice validation error:{}'.format(epoch, np.mean(dice_valid_error)))
                logging.info('epoch:{}, classification validation error:{}'.format(epoch, np.mean(classification_valid_error)))
                logging.info('-------------------------------------')

        # Save model checkpoints
        if epoch == start_epoch or epoch == n_epochs or epoch % 10 == 0:
            model_filename = f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_csrvc_layers{config.gnn_layers}_sf{config.sf}_{epoch}epochs.pt"

            torch.save(cortexode.state_dict(), os.path.join(model_dir, model_filename))

    final_model_filename = f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_csrvc_layers{config.gnn_layers}_sf{config.sf}.pt"

    # Save the final model
    torch.save(cortexode.state_dict(), os.path.join(model_dir, final_model_filename))

if __name__ == '__main__':
    mp.set_start_method('spawn')
    config = load_config()
    if config.train_type == 'surfandseg':
        train_surf(config)
    else:
        raise ValueError("Unsupported training type.")
