import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import monai.networks.nets as nets

import numpy as np
from tqdm import tqdm
from data.dataloader import SegDataset, BrainDataset
from model.net import CortexODE, Unet
from model.csrfusionnet import CSRFnet
from model.csrfusionnetv2 import CSRFnetV2
from model.csrfusionnetv3 import CSRFnetV3
from model.segmenterfactory import SegmenterFactory
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

def train_seg(config):
    """training WM segmentation"""
    
    # --------------------------
    # load configuration
    # --------------------------
    model_dir = config.model_dir   # the directory to save the checkpoints
    data_name = config.data_name
    device = config.device
    tag = config.tag
    n_epochs = config.n_epochs
    start_epoch = config.start_epoch
    lr = config.lr
    
    # start training logging
    logging.basicConfig(filename=model_dir+'model_seg_'+data_name+'_'+tag+'.log',
                        filemode='a',
                        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # --------------------------
    # load dataset
    # --------------------------
    logging.info("load dataset ...")
    
    #Updated
    trainset = SegDataset(config=config, data_usage='train')
    validset = SegDataset(config=config, data_usage='valid')

    #Updated
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)
    
    # --------------------------
    # initialize model
    # --------------------------
    logging.info("initalize model ...")
    #segnet = Unet(c_in=1, c_out=3).to(device)#renamed SegUnet
    print('config.seg_model_type',config.seg_model_type)

    if config.seg_model_type == "SwinUNETR":
        segnet = SegmenterFactory.get_segmenter("SwinUNETR",device)
    elif config.seg_model_type == "MonaiUnet":
        segnet = SegmenterFactory.get_segmenter("MonaiUnet",device)    
    elif config.seg_model_type == "SegUnet":
        segnet = SegmenterFactory.get_segmenter("SegUnet",device)
    else:
        assert False, "Config model name is incorrect"
    
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
        segnet.load_state_dict(torch.load(model_path, map_location=torch.device(config.device)))
        print(f"Model loaded from {model_path}")
    else:
        print("No model file provided or file does not exist. Starting from scratch.")
    
    print('start epoch',start_epoch)
    
    
    optimizer = optim.Adam(segnet.parameters(), lr=lr)
    # in case you need to load a checkpoint
    # segnet.load_state_dict(torch.load(model_dir+'model_seg_'+data_name+'_'+tag+'_XXepochs.pt'))
    # segnet.load_state_dict(torch.load('./ckpts/pretrained/adni/model_seg_adni_pretrained.pt'))

    # --------------------------
    # training model
    # --------------------------
    logging.info("start training ...")
        
    for epoch in tqdm(range(start_epoch, n_epochs + 1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            volume_in, seg_gt = data

            optimizer.zero_grad()
            volume_in = volume_in.to(device)
            seg_gt = seg_gt.long().to(device)

            # Check if the model requires padding and cropping
            if config.seg_model_type == "SwinUNETR":
                original_size = [1, 1, 176, 208, 176]  # Original input size
                target_size = [1, 1, 192, 224, 192]  # Target size after padding

                # Step 1: Pad the volume to the target size
                padded_volume = SegmenterFactory.pad_to_size(volume_in, target_size)
                
                # Process padded volume through the network
                seg_out = segnet(padded_volume)

                # Ensure the output size matches the padded input, with the channel size adjusted for segmentation
                pvshapemod = list(padded_volume.size())
                pvshapemod[2:] = target_size[2:]  # Only spatial dimensions are padded, channel dimension stays
                assert list(seg_out.size())[2:] == pvshapemod[2:], "Segmentation output size mismatch after padding"

                # Step 2: Crop the output back to the original input size
                original_size_seg = [1, 3, 176, 208, 176]  # Adjust for the number of segmentation classes
                cropped_volume = SegmenterFactory.crop_to_original_size(seg_out, original_size_seg)
                seg_out = cropped_volume
            else:
                # For models that don't require padding/cropping, process directly
                seg_out = segnet(volume_in)

            # Assertions for sanity checks
            if config.seg_model_type == "SwinUNETR":
                assert list(seg_out.size()) == original_size_seg, "Segmentation output size mismatch after cropping"
            else:
                original_size_no_padding = [1, 3, 176, 208, 176]  # Adjust for segmentation classes without padding
                assert list(seg_out.size()) == original_size_no_padding, "Segmentation output size mismatch without padding/cropping"

            # Compute the loss
            loss = nn.CrossEntropyLoss()(seg_out, seg_gt)
            
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()


        logging.info("epoch:{}, loss:{}".format(epoch,np.mean(avg_loss)))
        
        if epoch == start_epoch or epoch == n_epochs or epoch%10==0:
            logging.info('-------------validation--------------')
            with torch.no_grad():
                avg_error = []
                avg_dice = []
                for idx, data in enumerate(validloader):
                    volume_in, seg_gt = data
                    volume_in = volume_in.to(device)
                    seg_gt = seg_gt.long().to(device)

                    # Check if the model requires padding and cropping
                    if config.seg_model_type == "SwinUNETR":
                        # Same as the updated training loop
                        original_size = [1, 1, 176, 208, 176]
                        target_size = [1, 1, 192, 224, 192]

                        padded_volume = SegmenterFactory.pad_to_size(volume_in, target_size)
                        seg_out = segnet(padded_volume)

                        # Ensure the output size matches the padded input
                        assert list(seg_out.size())[2:] == target_size[2:]

                        original_size = [1, 3, 176, 208, 176]
                        cropped_volume = SegmenterFactory.crop_to_original_size(seg_out, original_size)
                        seg_out = cropped_volume
                    else:
                        # Original validation code for models that do not require padding
                        seg_out = segnet(volume_in)

                    avg_error.append(nn.CrossEntropyLoss()(seg_out, seg_gt).item())

                    # Adjust the following lines to work with or without padding/cropping as needed
                    seg_out = torch.argmax(seg_out, dim=1)
                    seg_out = F.one_hot(seg_out, num_classes=3).permute(0,4,1,2,3)[:,1:]
                    seg_gt = F.one_hot(seg_gt, num_classes=3).permute(0,4,1,2,3)[:,1:]
                    dice = compute_dice(seg_out, seg_gt, '3d')
                    avg_dice.append(dice)

                logging.info("epoch:{}, validation error:{}".format(epoch, np.mean(avg_error)))
                logging.info("Dice score:{}".format(np.mean(avg_dice)))
                logging.info('-------------------------------------')
        # save model checkpoints
        if epoch == start_epoch or epoch == n_epochs or epoch%10==0:
            torch.save(segnet.state_dict(),
                       model_dir+'model_seg_'+data_name+'_'+tag+'_'+str(epoch)+'epochs.pt')
    # save final model
    torch.save(segnet.state_dict(),
               model_dir+'model_seg_'+data_name+'_'+tag+'.pt')


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
    log_filename = f"{model_dir}/model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}_sf{config.sf}_{solver}"

    # If the GNN type is 'gat', include `gat_heads` in the filename
    if config.gnn == 'gat':
        log_filename += f"_heads{config.gat_heads}"

    # Complete the filename by appending '.log'
    log_filename += ".log"

    # Configure logging to append to the specified log file, with the desired format and level
    logging.basicConfig(filename=log_filename, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s')

    if config.gnn=='gcn':
        use_gcn=True
        gnnVersion=2
    elif config.gnn == 'gat':
        use_gcn=False
        gnnVersion=1
    else:
        assert False, f"unsupported gnn configuration {config.gnn}"    
    
    
    # --------------------------
    # initialize models
    # --------------------------
    logging.info("initalize model ...")
    
    print('csrf version ', config.version)
        
    if config.model_type == 'csrf' and config.version=='1':
        cortexode = CSRFnet(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q,
                       sf=config.sf,
                       gnn_layers=config.gnn_layers,
                       gnnVersion=gnnVersion,
                       gat_heads=config.gat_heads).to(device)
    elif config.model_type == 'csrf' and config.version=='2':
        cortexode = CSRFnetV2(dim_h=C, kernel_size=K, n_scale=Q,
                       sf=config.sf,
                       gnn_layers=config.gnn_layers,
                       use_gcn=use_gcn,
                       gat_heads=config.gat_heads).to(device)
    elif config.model_type == 'csrf' and config.version=='2L':
        cortexode = CSRFnetV2(dim_h=C, kernel_size=K, n_scale=Q,
                       sf=config.sf,
                       gnn_layers=config.gnn_layers,
                       use_gcn=use_gcn,
                       gat_heads=config.gat_heads).to(device)
    elif config.model_type == 'csrf' and config.version=='3':
        assert False, 'currently unsupported'
        cortexode = CSRFnetV3(dim_h=C, kernel_size=K, n_scale=Q,
                       sf=config.sf,
                       gnn_layers=config.gnn_layers,
                       use_gcn=use_gcn,
                       use_layernorm = use_layernorm,
                       gat_heads=config.gat_heads).to(device)
    else:
        cortexode = CortexODE(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)
    
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
    T = torch.Tensor([0,1]).to(device)    # integration time interval for ODE

    # --------------------------
    # load dataset
    # --------------------------
    logging.info("load dataset ...")
    #Updated
    trainset = BrainDataset(config, 'train')
    validset = BrainDataset(config, 'valid')

    #Updated
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)
    
    # --------------------------
    # training
    # --------------------------
    
    logging.info("start training ...")
    for epoch in tqdm(range(start_epoch, n_epochs + 1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            volume_in, v_in, v_gt, f_in, f_gt = data

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

            if surf_type == 'wm':    # training with randomly sampled points
                ### integration using seminorm (not recommended)
                # v_out = odeint(cortexode, v_in, t=T, method=solver,
                #                options=dict(step_size=step_size), adjoint_options=dict(norm='seminorm'))[-1]
                
                ### integration without seminorm
                v_out = odeint(cortexode, v_in, t=T, method=solver,
                               options=dict(step_size=step_size))[-1]
                
                mesh_out = Meshes(verts=v_out, faces=f_in)
                mesh_gt = Meshes(verts=v_gt, faces=f_gt)
                v_out = sample_points_from_meshes(mesh_out, n_samples)
                v_gt = sample_points_from_meshes(mesh_gt, n_samples)
                
                # scale by 1e3 since the coordinates are rescaled to [-1,1]
                loss = 1e3 * chamfer_distance(v_out, v_gt)[0]    # chamfer loss
                
            elif surf_type == 'gm':    # training with vertices
                v_out = odeint(cortexode, v_in, t=T, method=solver,
                               options=dict(step_size=step_size))[-1]
                loss = 1e3 * nn.MSELoss()(v_out, v_gt)

            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        logging.info('epoch:{}, loss:{}'.format(epoch, np.mean(avg_loss)))
        
        if epoch == start_epoch or epoch == n_epochs or epoch%10==0:
            logging.info('-------------validation--------------')
            with torch.no_grad():
                valid_error = []
                for idx, data in enumerate(validloader):
                    volume_in, v_in, v_gt, f_in, f_gt = data

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
            else:
                assert False,'update naming conventions for model file name'
            
            torch.save(cortexode.state_dict(), os.path.join(model_dir, model_filename))
    
    if config.gnn=='gat':
        final_model_filename = f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}_sf{config.sf}_heads{config.gat_heads}_{solver}.pt"
    elif config.gnn =='gcn':
        final_model_filename = f"model_{surf_type}_{data_name}_{surf_hemi}_{tag}_v{config.version}_gnn{config.gnn}_layers{config.gnn_layers}_sf{config.sf}_{solver}.pt"
    else:
        assert False,'update naming conventions for model file name'
    
    # save the final model
    torch.save(cortexode.state_dict(), os.path.join(model_dir, final_model_filename))


if __name__ == '__main__':
    mp.set_start_method('spawn')
    config = load_config()
    if config.train_type == 'surf':
        train_surf(config)
    elif config.train_type == 'seg':
        train_seg(config)
