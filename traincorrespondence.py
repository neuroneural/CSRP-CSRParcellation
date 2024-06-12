import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import monai.networks.nets as nets

import numpy as np
from tqdm import tqdm
from data.v2ccdataloader import FSAVGBrainDataset
from model.net import CortexODE, Unet
from model.csrfusionnetv3 import CSRFnetV3
from model.csrfunetv1 import CSRFUnetV1
from util.mesh import compute_dice

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

import logging
from torchdiffeq import odeint_adjoint as odeint
from config import load_config
import re
import os
import csv
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_surf(config):
    """
    Training CortexODE for cortical surface reconstruction
    using adjoint sensitivity method proposed in neural ODE
    
    For original neural ODE paper please see:
    - Chen et al. Neural ordinary differential equations. NeurIPS, 2018.
      Paper: https://arxiv.org/abs/1806.07366v5
      Code: https://github.com/rtqichen/torchdiffeq
    
    Note: using seminorm in adjoint method can accelerate the training, but it
    will cause exploding gradients for explicit methods in our experiments.

    For seminorm please see:
    - Patrick et al. Hey, that's not an ODE: Faster ODE Adjoints via Seminorms. ICML, 2021.
      Paper: https://arxiv.org/abs/2009.09457
      Code: https://github.com/patrick-kidger/FasterNeuralDiffEq

    Configurations (see config.py):
    model_dir: directory to save your checkpoints
    data_name: [hcp, adni, ...]
    surf_type: [wm, gm]
    surf_hemi: [lh, rh]
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
    
    n_epochs = config.n_epochs
    start_epoch = config.start_epoch
    n_samples = config.n_samples
    lr = config.lr
    
    C = config.dim_h     # hidden dimension of features
    K = config.kernel_size    # kernel / cube size
    Q = config.n_scale    # multi-scale input
    
    step_size = config.step_size    # step size of integration
    solver = config.solver    # ODE solver
    
    # create log file
    if config.version=="0":
        log_filename = f"{model_dir}/model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_sf{config.sf}_{solver}"
    else:
        log_filename = f"{model_dir}/model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}_sf{config.sf}_{solver}"

    # If the GNN type is 'gat', include `gat_heads` in the filename
    if config.gnn == 'gat':
        use_gcn = False
        log_filename += f"_heads{config.gat_heads}"
    elif config.gnn == 'gcn':
        use_gcn = True
    # Complete the filename by appending '.log'
    log_filename += ".log"

    # Configure logging to append to the specified log file, with the desired format and level
    logging.basicConfig(filename=log_filename, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s')

        
    # --------------------------
    # initialize models
    # --------------------------
    logging.info("initalize model ...")
    
    print('csrf version ', config.version)
        
    print('version 3 is loading')
    if config.model_type == 'csrfunet' and config.version=='1':
        cortexode = CSRFUnetV1(dim_h=C, kernel_size=K, n_scale=Q,
                       sf=config.sf,
                       gnn_layers=config.gnn_layers,
                       use_gcn=False,
                       gat_heads=config.gat_heads
                      ).to(device)
    else:
        assert False
        cortexode = CSRFnetV3(dim_h=C, kernel_size=K, n_scale=Q,
                        sf=config.sf,
                        gnn_layers=config.gnn_layers,
                        use_gcn=False,
                        gat_heads=config.gat_heads
                        ).to(device)
    
    model_path = None
    
    if config.model_file:
        print('loading model',config.model_file)
        print('hemi', config.surf_hemi)
        print('surftype', config.surf_type)
        
        start_epoch = int(config.start_epoch)
        model_path = os.path.join(config.model_dir, config.model_file)
    
    # Load model state if a model path is provided
    if model_path and os.path.isfile(model_path):
        print('device', config.device)
        cortexode.load_state_dict(torch.load(model_path, map_location=torch.device(config.device)))
        print(f"Model loaded from {model_path}")
    else:
        print("No model file provided or file does not exist. Starting from scratch.")
    
    print('start epoch',start_epoch)
    optimizer = optim.Adam(cortexode.parameters(), lr=lr)
    patience=0
    if config.patience != "standard":
        try:
            patience= int(config.patience)
        except:
            print("patience should either be standard (no scheduler) or an int >=0")
    else:
        print("scheduler is standard and will never step")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)

    T = torch.Tensor([0,1]).to(device)    # integration time interval for ODE

    # --------------------------
    # load dataset
    # --------------------------
    logging.info("load dataset ...")
    trainset = FSAVGBrainDataset(config, 'train')
    validset = FSAVGBrainDataset(config, 'valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)
    
    # --------------------------
    # training
    # --------------------------
    
    logging.info("start training ...")
    for epoch in tqdm(range(start_epoch, n_epochs + 1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            volume_in, v_in, v_gt, f_in, f_gt, _subj = data

            optimizer.zero_grad()

            volume_in = volume_in.to(device).float()
            v_in = v_in.to(device)
            f_in = f_in.to(device)
            v_gt = v_gt.to(device)
            f_gt = f_gt.to(device)
            
            if config.model_type == 'csrf' or  config.model_type == 'csrfunet' :
                cortexode.set_data(v_in, volume_in, f=f_in)
            else:
                cortexode.set_data(v_in, volume_in)    # set the input data

            print("torch.max v_in", torch.max(v_in))
            v_out = odeint(cortexode, v_in, t=T, method=solver,
                            options=dict(step_size=step_size))[-1]
            loss = 1e3 * nn.L1Loss()(v_out, v_gt)

            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        logging.info('epoch:{}, loss:{}'.format(epoch, np.mean(avg_loss)))
        
        if epoch == start_epoch or epoch == n_epochs or epoch%10==0:
            logging.info('-------------validation--------------')
            with torch.no_grad():
                valid_error = []
                for idx, data in enumerate(validloader):
                    volume_in, v_in, v_gt, f_in, f_gt, _subj = data

                    optimizer.zero_grad()

                    volume_in = volume_in.to(device).float()
                    v_in = v_in.to(device)
                    f_in = f_in.to(device)
                    v_gt = v_gt.to(device)
                    f_gt = f_gt.to(device)

                    if config.model_type == 'csrf':
                        cortexode.set_data(v_in, volume_in, f=f_in)
                    else:
                        cortexode.set_data(v_in, volume_in)    # set the input data
                    
                    v_out = odeint(cortexode, v_in, t=T, method=solver,
                                   options=dict(step_size=step_size))[-1]
                    valid_error.append(1e3 * chamfer_distance(v_out, v_gt)[0].item())

                if epoch > 1 and epoch %10 ==0 and config.patience != 'standard':
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(np.mean(valid_error).item())
                    new_lr = optimizer.param_groups[0]['lr']
                    if old_lr != new_lr:
                        print("Learning rate was adjusted from {} to {}".format(old_lr, new_lr))
                    else:
                        print("Learning rate was not adjusted.")
                
                                         

                logging.info('epoch:{}, validation error:{}'.format(epoch, np.mean(valid_error)))
                logging.info('-------------------------------------')
                # Log to CSV
                csv_log_path = os.path.join(model_dir, f"training_log_{tag}_{solver}.csv")
                fieldnames = ['surf_hemi', 'surf_type', 'version', 'epoch', 'training_loss', 'validation_error', 'gnn', 'gnn_layers', 'sf', 'gat_heads','solver']

                if not os.path.exists(csv_log_path):
                    with open(csv_log_path, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                avg_training_loss = np.mean(avg_loss)
                with open(csv_log_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if config.gnn=='gat':
                        writer.writerow({
                            'surf_hemi' : surf_hemi,
                            'surf_type' : surf_type,
                            'version' : config.version,
                            'epoch': epoch,
                            'training_loss': avg_training_loss,  # Include training loss here
                            'validation_error': np.mean(valid_error),
                            'gnn': config.gnn,
                            'gnn_layers': config.gnn_layers,
                            'sf': config.sf,
                            'gat_heads': config.gat_heads,
                            'solver':solver
                        })
                    elif config.gnn=='gcn':
                        writer.writerow({
                            'surf_hemi' : surf_hemi,
                            'surf_type' : surf_type,
                            'version' : config.version,
                            'epoch': epoch,
                            'training_loss': avg_training_loss,  # Include training loss here
                            'validation_error': np.mean(valid_error),
                            'gnn': config.gnn,
                            'gnn_layers': config.gnn_layers,
                            'sf': config.sf,
                            'gat_heads': 'NA',
                            'solver':solver
                        })

        
        # save model checkpoints 
        if epoch == start_epoch or epoch == n_epochs or epoch%10==0:
            if config.gnn=='gat':
                model_filename = f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}_sf{config.sf}_heads{config.gat_heads}_{epoch}epochs_{solver}.pt"
            elif config.gnn =='gcn':
                model_filename = f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}_sf{config.sf}_{epoch}epochs_{solver}.pt"
            elif config.gnn =='baseline':
                model_filename = f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_sf{config.sf}_{epoch}epochs_{solver}.pt"
            else:
                assert False,'update naming conventions for model file name'
            
            torch.save(cortexode.state_dict(), os.path.join(model_dir, model_filename))
    
    if config.gnn=='gat':
        final_model_filename = f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}_sf{config.sf}_heads{config.gat_heads}_{solver}.pt"
    elif config.gnn =='gcn':
        final_model_filename = f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}_sf{config.sf}_{solver}.pt"
    elif config.gnn =='baseline':
        final_model_filename = f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_sf{config.sf}_{solver}.pt"
    else:
        assert False,'update naming conventions for model file name'
    
    # save the final model
    torch.save(cortexode.state_dict(), os.path.join(model_dir, final_model_filename))

if __name__ == '__main__':
    mp.set_start_method('spawn')
    config = load_config()
    if config.train_type == 'surf':
        train_surf(config)
