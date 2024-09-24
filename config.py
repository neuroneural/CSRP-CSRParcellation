import argparse
import torch

def load_config():

    # args
    parser = argparse.ArgumentParser(description="CortexODE")
    
    # for training
    parser.add_argument('--data_dir', default="../dataset/hcp600/", type=str, help="directory of the dataset")
    parser.add_argument('--model_dir', default="./ckpts/model/", type=str, help="directory for saving the models")
    parser.add_argument('--model_type', default="csrf", type=str, help="either: [csrf, cortexode]")
    parser.add_argument('--model_type2', default="baseline", type=str, help="either: [csrf, cortexode]")
    parser.add_argument('--segmentation_model_path', default="", type=str, help="path for segmentation model.")
    
    parser.add_argument('--compute_reconstruction_loss', type=str, default='yes',
                        help='Flag to compute reconstruction loss (default: yes).')
    parser.add_argument('--compute_classification_loss', type=str, default='no',
                        help='Flag to compute classification loss (default: no).')
    
    parser.add_argument('--seg_model_type', default="SegUnet", type=str, help="either: [SwinUNETR, MonaiUnet, SegUnet]")
    parser.add_argument('--model_file', default=None, type=str, help="File for resuming models.")
    parser.add_argument('--gm_model_file', default=None, type=str, help="File for resuming models.")
    parser.add_argument('--wm_model_file', default=None, type=str, help="File for resuming models.")
    parser.add_argument('--seg_model_file', default=None, type=str, help="File for resuming models.")
    parser.add_argument('--num_time_steps', default=10, type=int, help="Number of timesteps for tgnn.")
    parser.add_argument('--continue', default="no", type=str, help="continue training [yes, no].")
    parser.add_argument('--start_epoch', default=1, type=int, help="start counting at 1 for start epochs.")
    parser.add_argument('--data_name', default="hcp", type=str, help="name of the dataset")
    parser.add_argument('--train_type', default="surf", type=str, help="type of training: [seg, surf, self_surf]")
    parser.add_argument('--surf_type', default="wm", type=str, help="type of the surface: [wm, gm]")
    parser.add_argument('--surf_hemi', default="lh", type=str, help="left or right hemisphere: [lh, rh]")
    parser.add_argument('--device', default="gpu", type=str, help="gpu or cpu")
    parser.add_argument('--tag', default='0000', type=str, help="identity for experiments")

    # Reconstruction loss thresholds
    parser.add_argument('--mse_threshold', default=0.01, type=float, help="target amount of error to start from")
    parser.add_argument('--loss_threshold', default=0.01, type=float, help="good enough to double target")
    parser.add_argument('--count_thresh', default=5, type=int, help="number of validations to fail before quit")
    parser.add_argument('--max_mse_thresh', default=0.06, type=float, help="quit when error is this big")
    
    parser.add_argument('--use_pytorch3d_normal', default='yes', type=str, help="[yes, no] normal computation flag")
    
    parser.add_argument('--version', default="1", type=str, help="either: [1, 2, 3]")
    
    # Version 1 parameters
    parser.add_argument('--gnn', default="gcn", type=str, help="either: [gcn, gat]")
    parser.add_argument('--gnn_layers', default=2, type=int, help="num of gnn layers [2, 3, 4, 5, 6, 7, 8]")
    parser.add_argument('--sf', default=0.1, type=float, help="scaling factor for final layer output")  # pairs with nonlinearity
    parser.add_argument('--gat_heads', default=8, type=int, help="num of GAT heads [1 recommended]")
    
    # Version 2 parameters (ignored in version 1)
    parser.add_argument('--use_layernorm', default='no', type=str, help="use layer norm: [yes, no]")
    parser.add_argument('--patience', default='0', type=str, help="scheduler patience standard or [0, 1, 2, ...]")
    
    parser.add_argument('--atlas', default="DKTatlas40", type=str, help="choose an atlas [DKTatlas40, aparc]")
    parser.add_argument('--visualize', default="no", type=str, help="[yes, no]")
    
    parser.add_argument('--solver', default='euler', type=str, help="ODE solver: [euler, midpoint, rk4]")
    parser.add_argument('--step_size', default=0.1, type=float, help="step size of the ODE solver")
    parser.add_argument('--lambd', default=1.0, type=float, help="Laplacian smooth weight")
    parser.add_argument('--n_inflate', default=2, type=int, help="num of iterations of Laplacian smoothing and inflating")
    parser.add_argument('--rho', default=0.002, type=float, help="inflation scale of normals")
    
    parser.add_argument('--n_epochs', default=100, type=int, help="num of training epochs")
    parser.add_argument('--n_samples', default=150000, type=int, help="num of sampled points for training")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    
    parser.add_argument('--kernel_size', default=5, type=int, help="kernel size of conv layers")
    parser.add_argument('--dim_h', default=128, type=int, help="dimension of hidden layers")
    parser.add_argument('--n_scale', default=3, type=int, help="num of scales for multi-scale inputs")
    
    parser.add_argument('--deform_model_dir', default="", type=str, help="deform model directory")
    parser.add_argument('--classification_model_dir', default="", type=str, help="classification model directory")
    parser.add_argument('--deform_model_file', default="", type=str, help="deform model file")
    parser.add_argument('--classify_model_file', default="", type=str, help="classification model file")
    parser.add_argument('--tagdeform', default="tagname", type=str, help="deformation tag")
    parser.add_argument('--tagclassification', default="tagname", type=str, help="classification tag")
    
    # For testing
    parser.add_argument('--test_type', default="pred", type=str, help="type of testing: [init, pred, eval]")
    parser.add_argument('--init_dir', default="./ckpts/init/", type=str, help="directory for saving the initial surfaces")
    parser.add_argument('--parc_init_dir', default=None, type=str, help="directory for saving the initial surfaces")
    parser.add_argument('--result_dir', default="./ckpts/result/", type=str, help="directory for saving the predicted surfaces")
    
    # --------------------------
    # Additional parameters to include
    # --------------------------
    
    # Number of classes for classification
    parser.add_argument('--num_classes', default=36, type=int, help="number of classes in the atlas")
    
    # Classification loss threshold (for starting classification loss computation)
    parser.add_argument('--classification_loss_threshold', default=0.04, type=float, help="threshold for starting classification loss")
    
    # Classification loss weight (to balance reconstruction and classification losses)
    parser.add_argument('--classification_loss_weight', default=1.0, type=float, help="weight for classification loss")
    
    # Number of neighbors for KDTree (if needed)
    # parser.add_argument('--k_neighbors', default=1, type=int, help="number of neighbors for KDTree search")
    
    config = parser.parse_args()
    
    # Device configuration
    if config.device == "gpu":
        config.device = torch.device("cuda")
    elif config.device == "cpu":
        config.device = torch.device("cpu")
    else:
        config.device = torch.device(config.device)
    
    # Ensure device is set correctly
    config.device = torch.device(config.device)
    
    return config
