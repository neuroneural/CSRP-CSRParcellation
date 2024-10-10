import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from data.csrandvcdataloader import BrainDataset
from model.csrvcv2 import CSRVCV2  # Updated import
from model.csrvcv3 import CSRVCV3  # Updated import
from model.csrvcv4 import CSRVCV4  # Updated import
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

from scipy.spatial import cKDTree  # cKDTree import
from sklearn.neighbors import KDTree  # KDTree import

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

def extract_rand_num_from_filename(filename):
    # Extract the rand_num from the filename
    # Assuming the rand_num is the last group of digits before .pt or _final.pt
    match = re.search(r'_(\d+)(?:\.pt|_final\.pt)$', filename)
    if match:
        return int(match.group(1))
    else:
        return None

# Custom logging handler that flushes after every emit
class FlushFileHandler(logging.FileHandler):
    """
    Custom FileHandler that flushes the buffer after every emit.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logger(log_filename):
    """
    Sets up a logger that writes to the specified log file with immediate flushing.
    """
    logger = logging.getLogger('train_surf_logger')
    logger.setLevel(logging.INFO)
    
    # Prevent adding multiple handlers if the logger is already set up
    if not logger.handlers:
        # Create custom handler that flushes after each message
        handler = FlushFileHandler(log_filename, mode='a')
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    return logger

def train_surf(config):
    """
    Training CSRVCV2 for cortical surface reconstruction and classification.
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
    print('atlas', config.atlas)
    n_epochs = config.n_epochs
    lr = config.lr

    C = config.dim_h     # Hidden dimension of features
    K = config.kernel_size    # Kernel / cube size
    Q = config.n_scale    # Multi-scale input

    step_size = config.step_size    # Step size of integration
    solver = config.solver    # ODE solver

    num_classes = config.num_classes  # Number of classes for classification

    # Loss weight for classification loss
    classification_loss_weight = config.classification_loss_weight  # e.g., 1.0

    # Add configuration flags to control loss computation
    compute_reconstruction_loss = config.compute_reconstruction_loss == 'yes'  # True or False
    compute_classification_loss = config.compute_classification_loss == 'yes'  # True or False

    # Convert boolean values to strings for filename
    recon_loss_str = 'recon' if compute_reconstruction_loss else 'norecon'
    class_loss_str = 'class' if compute_classification_loss else 'noclass'

    # Initialize use_gcn based on config
    if config.gnn == 'gat':
        use_gcn = False
    elif config.gnn == 'gcn':
        use_gcn = True
    else:
        use_gcn = False  # default to False if not specified

    # --------------------------
    # Initialize rand_num and log filename
    # --------------------------
    # Initialize rand_num to None
    rand_num = None

    # Load model_file to potentially extract rand_num
    if config.model_file:
        rand_num = extract_rand_num_from_filename(config.model_file)
        if rand_num is None:
            rand_num = random.randint(100000, 999999)
            print(f"Could not extract rand_num from filename. Generated new rand_num: {rand_num}")
        else:
            print(f"Extracted rand_num {rand_num} from model filename.")
    else:
        rand_num = random.randint(100000, 999999)

    # Create log file
    log_filename = os.path.join(
        model_dir,
        f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_csrvc_layers"
        f"{config.gnn_layers}_sf{config.sf}_{solver}_{recon_loss_str}_{class_loss_str}_{rand_num}"
    )

    if config.gnn == 'gat':
        log_filename += f"_heads{config.gat_heads}"
    elif config.gnn == 'gcn':
        pass  # No change needed for gcn

    log_filename += ".log"
    print('log_filename', log_filename)

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # --------------------------
    # Set up logger
    # --------------------------
    logger = setup_logger(log_filename)

    # Now, we can use logger.info instead of logging.info
    logger.info("Initialize model ...")

    T = torch.Tensor([0, 1]).to(device)    # Integration time interval for ODE

    # --------------------------
    # Initialize model
    # --------------------------
    # Initialize the model
    if config.model_type == 'csrvc' and config.version == '2':
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
        cortexode = CSRVCV4(dim_h=C,
                            kernel_size=K,
                            n_scale=Q,
                            sf=config.sf,
                            gnn_layers=config.gnn_layers,
                            use_gcn=use_gcn,
                            gat_heads=config.gat_heads,
                            num_classes=num_classes).to(device)
    else:
        raise ValueError("Unsupported model type or version.")

    optimizer = optim.Adam(cortexode.parameters(), lr=lr)

    # Load model state if a model path is provided
    if config.model_file:
        model_path = os.path.join(config.model_dir, config.model_file)
        if os.path.isfile(model_path):
            # Load model state dict directly
            cortexode.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Model state loaded from {model_path}. Starting from epoch {config.start_epoch}")
            start_epoch = config.start_epoch  # Since we're not loading epoch from checkpoint
        else:
            logger.info("No model file provided or file does not exist. Starting from scratch.")
            start_epoch = config.start_epoch
    else:
        start_epoch = config.start_epoch  # If not resuming, start from config.start_epoch

    # --------------------------
    # Load dataset
    # --------------------------
    logger.info("Load dataset ...")
    trainset = BrainDataset(config, 'train')  # Should include labels
    validset = BrainDataset(config, 'valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)

    # --------------------------
    # Training
    # --------------------------
    logger.info("Start training ...")
    for epoch in tqdm(range(start_epoch, n_epochs + 1)):
        avg_recon_loss = []
        avg_classification_loss = []
        in_dist_avg_classification_loss = []
        cortexode.train()
        
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
                    chamfer_loss = 1e3 * chamfer_distance(v_out, v_gt)[0]
                    mse_loss = 1e3 * nn.MSELoss()(v_out, v_gt)
                    reconstruction_loss = mse_loss + chamfer_loss

                reconstruction_loss.backward()
                optimizer.step()
                avg_recon_loss.append(reconstruction_loss.item())
                if compute_classification_loss:
                    # In-distribution approximate classification loss
                    v_out_np = v_out.detach().cpu().numpy()[0]
                    v_gt_np = v_gt.detach().cpu().numpy()[0]
                    labels_np = labels.detach().cpu().numpy()[0]
                    kdtree = KDTree(v_gt_np)
                    distances, indices = kdtree.query(v_out_np, k=1)
                    indices = torch.from_numpy(indices.flatten()).long().to(device)
                    nearest_gt_labels = torch.from_numpy(labels_np[indices.cpu().numpy()]).long().to(device)
                    cortexode.set_data(v_out, volume_in, f=f_in)
                    # Perform forward pass to get class logits without ODE integration
                    _ = cortexode(None, v_out)
                    class_logits = cortexode.get_class_logits()
                    class_logits = class_logits.permute(1, 0)
                    
                    # Ensure labels are within valid range
                    if torch.any(nearest_gt_labels < 0) or torch.any(nearest_gt_labels >= num_classes):
                        print(f"Invalid label detected in batch {idx} of epoch {epoch}")
                        print(f"Labels range: {nearest_gt_labels.min()} to {nearest_gt_labels.max()}")
                        continue  # Skip this batch
                    
                    # Compute classification loss
                    classification_loss = nn.CrossEntropyLoss()(class_logits.squeeze(), nearest_gt_labels.squeeze())
                    in_dist_avg_classification_loss.append(classification_loss.item())
    
            # Classification Loss
            if compute_classification_loss:
                optimizer.zero_grad()
                cortexode.set_data(v_gt, volume_in, f=f_gt)

                # Perform forward pass to get class logits without ODE integration
                _ = cortexode(None, v_gt)

                class_logits = cortexode.get_class_logits()
                class_logits = class_logits.permute(1, 0)  # Reshape logits
                
                # Ensure labels are within valid range
                if torch.any(labels < 0) or torch.any(labels >= num_classes):
                    print(f"Invalid label detected in batch {idx} of epoch {epoch}")
                    print(f"Labels range: {labels.min()} to {labels.max()}")
                    continue  # Skip this batch

                # Compute classification loss
                classification_loss = nn.CrossEntropyLoss()(class_logits.squeeze(), labels.squeeze())
                classification_loss.backward()
                optimizer.step()
                avg_classification_loss.append(classification_loss.item())

        logger.info('epoch:{}, recon loss:{}'.format(epoch, np.mean(avg_recon_loss)))
        logger.info('epoch:{}, classification loss:{}'.format(epoch, np.mean(avg_classification_loss)))
        logger.info('epoch:{}, in_dist_classification loss:{}'.format(epoch, np.mean(in_dist_avg_classification_loss)))

        if epoch == start_epoch or epoch == n_epochs or epoch % 10 == 0:
            logger.info('-------------validation--------------')
            with torch.no_grad():
                cortexode.eval()
                recon_valid_error = []
                chamfer_valid_error = []
                mse_valid_error = []
                dice_valid_error = []
                in_dist_dice_valid_error = []
                in_dist_classification_valid_error = []
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
                            chamfer_loss = 1e3 * chamfer_distance(v_out, v_gt)[0]
                            reconstruction_loss = mse_loss + chamfer_loss

                        recon_valid_loss = reconstruction_loss.item()
                        chamfer_valid_loss = chamfer_loss.item()
                        mse_valid_loss  = mse_loss.item()
                        if compute_classification_loss:
                            # In-distribution approximate classification loss
                            v_out_np = v_out.detach().cpu().numpy()[0]
                            v_gt_np = v_gt.detach().cpu().numpy()[0]
                            labels_np = labels.detach().cpu().numpy()[0]
                            kdtree = KDTree(v_gt_np)
                            distances, indices = kdtree.query(v_out_np, k=1)
                            indices = torch.from_numpy(indices.flatten()).long().to(device)
                            nearest_gt_labels = torch.from_numpy(labels_np[indices.cpu().numpy()]).long().to(device)
                            cortexode.set_data(v_out, volume_in, f=f_in)
                            # Perform forward pass to get class logits without ODE integration
                            _ = cortexode(None, v_out)
                            class_logits = cortexode.get_class_logits()
                            class_logits = class_logits.permute(1, 0)
                            
                            # Ensure labels are within valid range
                            if torch.any(nearest_gt_labels < 0) or torch.any(nearest_gt_labels >= num_classes):
                                print(f"Invalid label detected in validation batch {idx} of epoch {epoch}")
                                print(f"Labels range: {nearest_gt_labels.min()} to {nearest_gt_labels.max()}")
                                continue  # Skip this batch
                            
                            # Compute classification loss
                            classification_loss = nn.CrossEntropyLoss()(class_logits.squeeze(), nearest_gt_labels.squeeze())
                            in_dist_classification_valid_error.append(classification_loss.item())
                            
                            class_logits = class_logits.unsqueeze(0)
                            class_logits = F.log_softmax(class_logits, dim=1)
                            
                            # Compute Dice score
                            predicted_classes = torch.argmax(class_logits, dim=1)
                            exclude_classes = [-1,4] if config.atlas in ['aparc', 'DKTatlas40'] else []
                            in_dist_dice_score = compute_dice(predicted_classes, nearest_gt_labels.unsqueeze(0), num_classes, exclude_classes)
                            in_dist_dice_valid_error.append(in_dist_dice_score)
                            
                    if compute_classification_loss:
                        # Set data for classification
                        cortexode.set_data(v_gt, volume_in, f=f_gt)

                        # Perform forward pass to get class logits without ODE integration
                        _ = cortexode(None, v_gt)

                        class_logits = cortexode.get_class_logits()
                        class_logits = class_logits.permute(1, 0)
                        
                        # Ensure labels are within valid range
                        if torch.any(labels < 0) or torch.any(labels >= num_classes):
                            print(f"Invalid label detected in validation batch {idx} of epoch {epoch}")
                            print(f"Labels range: {labels.min()} to {labels.max()}")
                            continue  # Skip this batch
                        
                        # Compute classification loss
                        classification_loss = nn.CrossEntropyLoss()(class_logits.squeeze(), labels.squeeze())
                        classification_valid_error.append(classification_loss.item())
                        
                        # Compute Dice score
                        class_logits = class_logits.unsqueeze(0)#assert now how 3 dim
                        class_logits = F.log_softmax(class_logits, dim=1)
                        
                        predicted_classes = torch.argmax(class_logits, dim=1)
                        exclude_classes = [-1,4] if config.atlas in ['aparc', 'DKTatlas40'] else []
                        dice_score = compute_dice(predicted_classes, labels, num_classes, exclude_classes)
                        dice_valid_error.append(dice_score)
                         

                    recon_valid_error.append(recon_valid_loss)
                    chamfer_valid_error.append(chamfer_valid_loss)
                    mse_valid_error.append(mse_valid_loss)

                logger.info('epoch:{}, reconstruction validation error:{}'.format(epoch, np.mean(recon_valid_error)))
                logger.info('epoch:{}, chamfer validation error:{}'.format(epoch, np.mean(chamfer_valid_error)))
                logger.info('epoch:{}, mse validation error:{}'.format(epoch, np.mean(mse_valid_error)))
                logger.info('epoch:{}, dice validation error:{}'.format(epoch, np.mean(dice_valid_error)))
                logger.info('epoch:{}, in_dist_dice validation error:{}'.format(epoch, np.mean(in_dist_dice_valid_error)))
                logger.info('epoch:{}, classification validation error:{}'.format(epoch, np.mean(classification_valid_error)))
                logger.info('epoch:{}, in_dist_classification validation error:{}'.format(epoch, np.mean(in_dist_classification_valid_error)))
                logger.info('-------------------------------------')

        if epoch == start_epoch or epoch == n_epochs or epoch % 10 == 0:
            # Create model filename based on existing naming conventions
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
                raise ValueError('Update naming conventions for model file name')

            # Save only the model's state_dict
            torch.save(cortexode.state_dict(), os.path.join(model_dir, model_filename))

    # Save the final model
    if config.gnn == 'gat':
        final_model_filename = (
            f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_csrvc_layers"
            f"{config.gnn_layers}_sf{config.sf}_heads{config.gat_heads}_{n_epochs}epochs_{solver}_"
            f"{recon_loss_str}_{class_loss_str}_{rand_num}_final.pt"
        )
    elif config.gnn == 'gcn':
        final_model_filename = (
            f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_csrvc_layers"
            f"{config.gnn_layers}_sf{config.sf}_{n_epochs}epochs_{solver}_{recon_loss_str}_{class_loss_str}_{rand_num}_final.pt"
        )
    else:
        raise ValueError('Update naming conventions for model file name')

    # Save final model's state_dict
    torch.save(cortexode.state_dict(), os.path.join(model_dir, final_model_filename))

if __name__ == '__main__':
    mp.set_start_method('spawn')
    config = load_config()
    if config.train_type == 'surfandseg':
        # Add default values for new config options if they are not set
        if not hasattr(config, 'compute_reconstruction_loss'):
            config.compute_reconstruction_loss = 'yes'  # Changed to string to match earlier comparison
        if not hasattr(config, 'compute_classification_loss'):
            config.compute_classification_loss = 'yes'  # Changed to string to match earlier comparison
        train_surf(config)
    else:
        raise ValueError("Unsupported training type.")
