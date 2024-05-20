import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import compute_normal
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATConv

from util.mesh import compute_normal

from model.deformationgnn import DeformationGNN
from pytorch3d.structures import Meshes

class NodeFeatureNet(nn.Module):
    def __init__(self, C=128, K=5, n_scale=1,use_pytorch3d=True):
        super(NodeFeatureNet, self).__init__()
        # mlp layers
        self.use_pytorch3d = use_pytorch3d
        self.fc1 = nn.Linear(6, C)
        self.fc2 = nn.Linear(2*C, C*4)
        self.fc3 = nn.Linear(C*4, C*2)
        
        # for local convolution operation
        self.localconv = nn.Conv3d(n_scale, C, (K, K, K))
        self.localfc = nn.Linear(C, C)
        
        Q = n_scale      # number of scales
        
        self.n_scale = n_scale
        self.K = K        
        self.C = C
        self.Q = Q
        # for cube sampling
        self.initialized = False
        grid = np.linspace(-K//2, K//2, K)
        grid_3d = np.stack(np.meshgrid(grid, grid, grid), axis=0).transpose(2,1,3,0)
        self.x_shift = torch.Tensor(grid_3d.copy()).view(-1,3)
        self.cubes = torch.zeros([1, self.Q, self.K, self.K, self.K])

    def forward(self, v):
        
        z_local = self.cube_sampling(v)
        z_local = self.localconv(z_local)
        z_local = F.leaky_relu(z_local,0.2)#New relu
        z_local = z_local.view(-1, self.m, self.C)
        z_local = self.localfc(z_local)
        z_local = F.leaky_relu(z_local,0.2)#New relu
        # point feature
        if not self.use_pytorch3d:
            normal = compute_normal(v,self.f)#depricate this
        else:
            mesh = Meshes(verts=v, faces=self.f)
            normal = mesh.verts_normals_packed()
            normal = normal.unsqueeze(0)
        x = torch.cat([torch.zeros_like(v), normal], 2)
        z_point = F.leaky_relu(self.fc1(x), 0.2)
        
        # feature fusion
        z = torch.cat([z_point, z_local], 2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        
        return z    # node features
    def _initialize(self, V):
        # initialize coordinates shift and cubes
        self.x_shift = self.x_shift.to(V.device)
        self.cubes = self.cubes.to(V.device)
        self.initialized = True
    def set_data(self, x, V,f=None,edge_list=None):
        # x: coordinats
        # V: input brain MRI volume
        if not self.initialized:
            self._initialize(V)
        self.f = f
        self.edge_list=edge_list
        # set the shape of the volume
        D1,D2,D3 = V[0,0].shape
        D = max([D1,D2,D3])
        # rescale for grid sampling
        self.rescale = torch.Tensor([D3/D, D2/D, D1/D]).to(V.device)
        self.D = D

        self.m = x.shape[1]    # number of points
        self.neighbors = self.cubes.repeat(self.m,1,1,1,1)    # repeat m cubes
        
        # set multi-scale volume
        self.Vq = [V]
        for q in range(1, self.Q):
            # iteratively downsampling
            self.Vq.append(F.avg_pool3d(self.Vq[-1], 2))

    def cube_sampling(self, x):
        # x: coordinates
        with torch.no_grad():
            for q in range(self.Q):
                # make sure the cubes have the same size
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2**q)
                xq = xq.contiguous().view(1,-1,3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale  # rescale the coordinates
                # sample the q-th cube
                vq = F.grid_sample(self.Vq[q], xq, mode='bilinear', padding_mode='border', align_corners=True)
                # update the cubes
                self.neighbors[:,q] = vq[0,0].view(self.m, self.K, self.K, self.K)
        
        return self.neighbors.clone()
    
class DeformBlockGNN(nn.Module):
    def __init__(self, C=128, K=5, n_scale=3, sf=.1, gnn_layers=2, use_gcn=True, gat_heads=8,use_pytorch3d=True):
        super(DeformBlockGNN, self).__init__()
        self.sf=sf
        self.nodeFeatureNet = NodeFeatureNet(C=C, K=K, n_scale=n_scale,use_pytorch3d=use_pytorch3d)
        # Initialize ResidualGNN with parameters adjusted for the task
        self.gnn = DeformationGNN(input_features=C*2,  # Adjust based on NodeFeatureNet output
                                   hidden_features=C,
                                   output_dim=3,  # Assuming 3D deformation vector
                                   num_layers=gnn_layers,
                                   gat_heads=gat_heads,  # Adjust as needed
                                   use_gcn=use_gcn  # Choose between GCN and GAT
                                   )  # Based on deformation requirements
    
    def set_data(self, x, V,f=None,edge_list=None):
        # x: coordinats
        # V: input brain MRI volume
        self.nodeFeatureNet.set_data(x,V,f=f,edge_list=edge_list)
        self.f = f
        self.V = V
        self.edge_list = edge_list
    
    def forward(self, v):
        x = self.nodeFeatureNet(v)
        x = x.squeeze()
        dx = self.gnn(x, self.edge_list)*self.sf #threshold the deformation like before
        return dx

class CSRFnetV3(nn.Module):
    """
    The deformation network of CortexODE model.

    dim_h (C): hidden dimension
    kernel_size (K): size of convolutional kernels
    n_scale (Q): number of scales of the multi-scale input
    """
    
    def __init__(self, dim_h=128,
                       kernel_size=5,
                       n_scale=3,
                       sf=.1,
                       gnn_layers=5,
                       use_gcn=True,
                       gat_heads=8,
                       use_pytorch3d=True
                       ):
        
        super(CSRFnetV3, self).__init__()

        C = dim_h        # hidden dimension
        K = kernel_size  # kernel size
        
        
        self.block1 = DeformBlockGNN(C, 
                                     K, 
                                     n_scale,
                                     sf,
                                     gnn_layers=gnn_layers,
                                     use_gcn=use_gcn,
                                     gat_heads=gat_heads,
                                     use_pytorch3d=use_pytorch3d
                                     )
        
    def set_data(self, x, V,f=None,reduced_DOF=False):
        # x: coordinats
        # V: input brain MRI volume
        assert x.shape[0] == 1
        assert f.shape[0] == 1
        assert x.shape[1] != 1
        assert f.shape[1] != 1
        self.f = f
        self.reduced_DOF=reduced_DOF
        edge_list = torch.cat([self.f[0,:,[0,1]],
                               self.f[0,:,[1,2]],
                               self.f[0,:,[2,0]]], dim=0).transpose(1,0)#moved from after x

        edge_list = add_self_loops(edge_list)[0]
        
        self.edge_list = edge_list
        
        self.block1.set_data(x,V,f=f,edge_list=edge_list)
        
    #this method gets called by odeint, and thus the method signature has t in it even though the t is ignored
    def forward(self, t, x):
        dx = self.block1(x) # current
        dx = dx.unsqueeze(0) # current
        
        return dx #current
