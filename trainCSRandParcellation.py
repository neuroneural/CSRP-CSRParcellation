import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from data.csrandvcdataloader import BrainDataset
from model.csrvcv2 import CSRVCV2  # Updated import
from model.csrvcv3 import CSRVCV3  # Updated import
from model.csrvcSplitGnn import CSRVCSPLITGNN  # Updated import
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes

import logging
from torchdiffeq import odeint_adjoint as odeint
from config import load_config
import re
import os
import csv
import torch.multiprocessing as mp

from scipy.spatial import cKDTree

import torch.nn.functional as F

import random

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
    Training CSRVCV2 for cortical surface reconstruction and classification.
    """
    rand_num = random.randint(100000, 999999)
    # --------------------------
    # Load configuration
    # --------------------------
    model_dir = config.model_dir
    data_name = config.data_name
    surf_type = config.surf_type
    surf_hemi = config.surf_hemi
    device = config.device
    tag = config.tag
    print('surf_type',surf_type)
    print('surf_hemi',surf_hemi)
    n_epochs = config.n_epochs
    start_epoch = config.start_epoch
    lr = config.lr
    
    C = config.dim_h     # Hidden dimension of features
    K = config.kernel_size    # Kernel / cube size
    Q = config.n_scale    # Multi-scale input
    
    step_size = config.step_size    # Step size of integration
    solver = config.solver    # ODE solver
    
    num_classes = config.num_classes  # Number of classes for classification
    
    # Threshold for starting classification loss computation (removed as per the modifications)
    # classification_loss_threshold = config.classification_loss_threshold  # e.g., 0.04
    
    # Loss weight for classification loss
    classification_loss_weight = config.classification_loss_weight  # e.g., 1.0
    
    # Add configuration flags to control loss computation
    compute_reconstruction_loss = config.compute_reconstruction_loss == 'yes'  # True or False
    compute_classification_loss = config.compute_classification_loss == 'yes'  # True or False
    
    # Convert boolean values to strings for filename
    recon_loss_str = 'recon' if compute_reconstruction_loss else 'norecon'
    class_loss_str = 'class' if compute_classification_loss else 'noclass'


    # Create log file
    log_filename = os.path.join(
        model_dir,
        f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_csrvc_layers"
        f"{config.gnn_layers}_sf{config.sf}_{solver}_{recon_loss_str}_{class_loss_str}_{rand_num}"
    )

    print('log_filename',log_filename)
    if config.gnn == 'gat':
        use_gcn = False
        log_filename += f"_heads{config.gat_heads}"
    elif config.gnn == 'gcn':
        use_gcn = True
    else:
        use_gcn = False  # default to False if not specified

    log_filename += ".log"

    # Configure logging
    logging.basicConfig(filename=log_filename, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s')

    # --------------------------
    # Initialize model
    # --------------------------
    logging.info("Initialize model ...")

    # Initialize the model
    if config.model_type == 'csrvc' and config.version == '2':
        assert False,'sanity check'
        cortexode = CSRVCV2(dim_h=C,
                            kernel_size=K,
                            n_scale=Q,
                            sf=config.sf,
                            gnn_layers=config.gnn_layers,
                            use_gcn=use_gcn,
                            gat_heads=config.gat_heads,
                            num_classes=num_classes).to(device)
    elif config.model_type == 'csrvc' and config.version == '3':
        cortexode = CSRVCV3(dim_h=C,
                            kernel_size=K,
                            n_scale=Q,
                            sf=config.sf,
                            gnn_layers=config.gnn_layers,
                            use_gcn=use_gcn,
                            gat_heads=config.gat_heads,
                            num_classes=num_classes).to(device)
    elif config.model_type == "csrvc" and config.version == '4':
        assert False,'sanity check'
        
        cortexode = CSRVCSPLITGNN(dim_h=C,
                            kernel_size=K,
                            n_scale=Q,
                            sf=config.sf,
                            gnn_layers=config.gnn_layers,
                            use_gcn=use_gcn,
                            gat_heads=config.gat_heads,
                            num_classes=num_classes).to(device) 
    else:
        raise ValueError("Unsupported model type or version.")

    # Load model state if a model path is provided
    if config.model_file:
        model_path = os.path.join(config.model_dir, config.model_file)
        if os.path.isfile(model_path):
            cortexode.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            print("No model file provided or file does not exist. Starting from scratch.")

    optimizer = optim.Adam(cortexode.parameters(), lr=lr)

    T = torch.Tensor([0, 1]).to(device)    # Integration time interval for ODE

    # --------------------------
    # Load dataset
    # --------------------------
    logging.info("Load dataset ...")
    trainset = BrainDataset(config, 'train')  # Should include labels
    validset = BrainDataset(config, 'valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)

    # --------------------------
    # Training
    # --------------------------

    logging.info("Start training ...")
    for epoch in tqdm(range(start_epoch, n_epochs + 1)):
        avg_recon_loss = []
        avg_classification_loss = []
        for idx, data in enumerate(trainloader):
            # Unpack data
            volume_in, v_in, v_gt, f_in, f_gt, labels = data

            optimizer.zero_grad()

            # Move data to device
            volume_in = volume_in.to(device).float()
            v_in = v_in.to(device).float()
            v_gt = v_gt.to(device).float()
            f_in = f_in.to(device).long()
            f_gt = f_gt.to(device).long()
            labels = labels.to(device).long()

            
            # Reconstruction Loss
            if compute_reconstruction_loss:
                # Set initial state and data
                cortexode.set_data(v_in, volume_in, f=f_in)

                # Integrate over time
                v_out = odeint(cortexode, v_in, t=T, method=solver, options=dict(step_size=step_size))[-1]

                # Compute reconstruction loss as before
                if surf_type == 'wm':
                    chamfer_loss = 1e3 * chamfer_distance(v_out, v_gt)[0]
                    reconstruction_loss = chamfer_loss
                elif surf_type == 'gm':
                    mse_loss = 1e3 * nn.MSELoss()(v_out, v_gt)
                    reconstruction_loss = mse_loss

                reconstruction_loss.backward()
                optimizer.step()
                avg_recon_loss.append(reconstruction_loss.item())
                        
            # Classification Loss
            if compute_classification_loss:
                optimizer.zero_grad()
                cortexode.set_data(v_gt, volume_in, f=f_gt)

                # Perform forward pass to get class logits without ODE integration
                _ = cortexode(None, v_gt)
                
                class_logits = cortexode.get_class_logits()
                class_logits = class_logits.unsqueeze(0)
                class_logits = class_logits.permute(0, 2, 1)  # Reshape logits

                # Ensure labels are within valid range
                if torch.any(labels < 0) or torch.any(labels >= num_classes):
                    print(f"Invalid label detected in batch {idx} of epoch {epoch}")
                    print(f"Labels range: {labels.min()} to {labels.max()}")
                    continue  # Skip this batch
                    
                # Compute classification loss
                classification_loss = nn.CrossEntropyLoss()(class_logits, labels)
                classification_loss.backward()
                optimizer.step()
                avg_classification_loss.append(classification_loss.item())

            
            
        logging.info('epoch:{}, recon loss:{}'.format(epoch, np.mean(avg_recon_loss)))
        logging.info('epoch:{}, classification loss:{}'.format(epoch, np.mean(avg_classification_loss)))

        if epoch == start_epoch or epoch == n_epochs or epoch % 10 == 0:
            logging.info('-------------validation--------------')
            with torch.no_grad():
                
                recon_valid_error = []
                dice_valid_error = []
                classification_valid_error = []
                for idx, data in enumerate(validloader):
                    volume_in, v_in, v_gt, f_in, f_gt, labels = data

                    # Move data to device
                    volume_in = volume_in.to(device).float()
                    v_in = v_in.to(device).float()
                    v_gt = v_gt.to(device).float()
                    f_in = f_in.to(device).long()
                    f_gt = f_gt.to(device).long()
                    labels = labels.to(device).long()
                    
                    recon_valid_loss = 0

                    if compute_reconstruction_loss:
                        # Set initial state and data
                        cortexode.set_data(v_in, volume_in, f=f_in)

                        # Integrate over time
                        v_out = odeint(cortexode, v_in, t=T, method=solver, options=dict(step_size=step_size))[-1]

                        # Compute reconstruction loss
                        if surf_type == 'wm':
                            chamfer_loss = 1e3 * chamfer_distance(v_out, v_gt)[0]
                            reconstruction_loss = chamfer_loss
                        elif surf_type == 'gm':
                            mse_loss = 1e3 * nn.MSELoss()(v_out, v_gt)
                            reconstruction_loss = mse_loss

                        recon_valid_loss = reconstruction_loss.item()
                    
                    if compute_classification_loss:
                        # Set data for classification
                        cortexode.set_data(v_gt, volume_in, f=f_gt)

                        # Perform forward pass to get class logits without ODE integration
                        _ = cortexode(None, v_gt)

                        class_logits = cortexode.get_class_logits()
                        class_logits = F.log_softmax(class_logits, dim=1)
                        class_logits = class_logits.unsqueeze(0)
                        class_logits = class_logits.permute(0, 2, 1)

                        # Ensure labels are within valid range
                        if torch.any(labels < 0) or torch.any(labels >= num_classes):
                            print(f"Invalid label detected in validation batch {idx} of epoch {epoch}")
                            print(f"Labels range: {labels.min()} to {labels.max()}")
                            continue  # Skip this batch

                        # Compute classification loss
                        classification_loss = nn.CrossEntropyLoss()(class_logits, labels)
                        classification_valid_error.append(classification_loss.item())

                        # Compute Dice score
                        predicted_classes = torch.argmax(class_logits, dim=1)
                        exclude_classes = [4] if config.atlas in ['aparc', 'DKTatlas40'] else []
                        dice_score = compute_dice(predicted_classes, labels, num_classes, exclude_classes)
                        dice_valid_error.append(dice_score)

                    recon_valid_error.append(recon_valid_loss)

                logging.info('epoch:{}, reconstruction validation error:{}'.format(epoch, np.mean(recon_valid_error)))
                logging.info('epoch:{}, dice validation error:{}'.format(epoch, np.mean(dice_valid_error)))
                logging.info('epoch:{}, classification validation error:{}'.format(epoch, np.mean(classification_valid_error)))
                logging.info('-------------------------------------')

        if epoch == start_epoch or epoch == n_epochs or epoch % 10 == 0:
            if config.gnn == 'gat':
                model_filename = (
                    f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_csrvc_layers"
                    f"{config.gnn_layers}_sf{config.sf}_heads{config.gat_heads}_{epoch}epochs_{solver}_"
                    f"{recon_loss_str}_{class_loss_str}_{rand_num}.pt"
                )
            elif config.gnn == 'gcn':
                model_filename = (
                    f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_csrvc_layers"
                    f"{config.gnn_layers}_sf{config.sf}_{epoch}epochs_{solver}_{recon_loss_str}_{class_loss_str}_{rand_num}.pt"
                )
            else:
                assert False, 'Update naming conventions for model file name'

            torch.save(cortexode.state_dict(), os.path.join(model_dir, model_filename))

    # Save the final model
    if config.gnn == 'gat':
        final_model_filename = (
            f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_csrvc_layers"
            f"{config.gnn_layers}_sf{config.sf}_heads{config.gat_heads}_{solver}_"
            f"{recon_loss_str}_{class_loss_str}_{rand_num}.pt"
        )
    elif config.gnn == 'gcn':
        final_model_filename = (
            f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_csrvc_layers"
            f"{config.gnn_layers}_sf{config.sf}_{solver}_{recon_loss_str}_{class_loss_str}_{rand_num}.pt"
        )
    else:
        assert False, 'Update naming conventions for model file name'

    torch.save(cortexode.state_dict(), os.path.join(model_dir, final_model_filename))


if __name__ == '__main__':
    mp.set_start_method('spawn')
    config = load_config()
    if config.train_type == 'surfandseg':
        # Add default values for new config options if they are not set
        if not hasattr(config, 'compute_reconstruction_loss'):
            config.compute_reconstruction_loss = True  # Default to True
        if not hasattr(config, 'compute_classification_loss'):
            config.compute_classification_loss = True  # Default to True
        train_surf(config)
    else:
        raise ValueError("Unsupported training type.")
