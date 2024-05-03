import argparse
import torch
    
def load_config():

    # args
    parser = argparse.ArgumentParser(description="CortexODE")
    
    # for training
    parser.add_argument('--data_dir', default="../dataset/hcp600/", type=str, help="directory of the dataset")
    parser.add_argument('--model_dir', default="./ckpts/model/", type=str, help="directory for saving the models")
    parser.add_argument('--model_type', default="csrf", type=str, help="either: [csrf, cortexode]")
    parser.add_argument('--seg_model_type', default="SegUnet", type=str, help="either: [SwinUNETR,MonaiUnet,SegUnet]")
    parser.add_argument('--model_file', default=None, type=str, help="File for resuming models.")
    parser.add_argument('--start_epoch', default=1, type=int, help="start counting at 1 for start epochs.")
    parser.add_argument('--data_name', default="hcp", type=str, help="name of the dataset")
    parser.add_argument('--train_type', default="surf", type=str, help="type of training: [seg, surf,self_surf]")
    parser.add_argument('--surf_type', default="wm", type=str, help="type of the surface: [wm, gm]")
    parser.add_argument('--surf_hemi', default="lh", type=str, help="left or right hemisphere: [lh, rh]")
    parser.add_argument('--device', default="gpu", type=str, help="gpu or cpu")
    parser.add_argument('--tag', default='0000', type=str, help="identity for experiments")

    #parser.add_argument('--mse_threshold', default=0.036, type=float, help="scaling factor for tanh nonlinearity [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0] ")
    parser.add_argument('--mse_threshold', default=0.036, type=float, help="scaling factor for tanh nonlinearity [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0] ")
    parser.add_argument('--loss_threshold', default=0.01, type=float, help="scaling factor for tanh nonlinearity [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0] ")
    parser.add_argument('--count_thresh', default=5, type=int, help="scaling factor for tanh nonlinearity [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0] ")
    parser.add_argument('--max_mse_thresh', default=0.2, type=float, help="scaling factor for tanh nonlinearity [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0] ")
    
    parser.add_argument('--version', default="1", type=str, help="either: [1,2,3]")
    
    #version 1 params:
    parser.add_argument('--gnn', default="gcn", type=str, help="either: [gcn,gat]")
    parser.add_argument('--gnn_layers', default=2, type=int, help="num of gnn layers [2,3,4,5,6,7,8]")
    parser.add_argument('--sf', default=0.1, type=float, help="scaling factor for tanh nonlinearity [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0] ")
    parser.add_argument('--gat_heads', default=8, type=int, help="num of gnn heads [1 recommended]")
    
    #version 2 params, don't permute these for search in version 1 (wastes time, will be ignored in version 1):
    parser.add_argument('--use_layernorm', default='no', type=str, help="use layer norm:[yes,no]")
    
    parser.add_argument('--solver', default='euler', type=str, help="ODE solver: [euler, midpoint, rk4]")
    parser.add_argument('--step_size', default=0.1, type=float, help="step size of the ODE solver")
    parser.add_argument('--lambd', default=1.0, type=float, help="Laplacian smooth weight")
    parser.add_argument('--n_inflate', default=2, type=int, help="num of interations of Laplacian smoothing and inflating")
    parser.add_argument('--rho', default=0.002, type=float, help="inflation scale of normals")
    
    parser.add_argument('--n_epochs', default=100, type=int, help="num of training epochs")
    parser.add_argument('--n_samples', default=150000, type=int, help="num of sampled points for training")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    
    parser.add_argument('--kernel_size', default=5, type=int, help="kernel size of conv layers")
    parser.add_argument('--dim_h', default=128, type=int, help="dimension of hidden layers")
    parser.add_argument('--n_scale', default=3, type=int, help="num of scales for multi-scale inputs")
    
    # for testing
    parser.add_argument('--test_type', default="pred", type=str, help="type of testing: [init, pred, eval]")
    parser.add_argument('--init_dir', default="./ckpts/init/", type=str, help="directory for saving the initial surfaces")
    parser.add_argument('--result_dir', default="./ckpts/result/", type=str, help="directory for saving the predicted surfaces")


    config = parser.parse_args()
    
    if config.device == "gpu":
        config.device = torch.device("cuda")
    elif config.device == "cpu":
        config.device = torch.device("cpu")
    else:
        config.device = torch.device(config.device)
    
    config.device = torch.device(config.device)
    return config