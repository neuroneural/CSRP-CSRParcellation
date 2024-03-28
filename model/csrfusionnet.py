import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import compute_normal
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATConv

from util.mesh import compute_normal

#N layer GCN using pyg
class NLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,num_layers):
        super(NLayerGCN, self).__init__()
        assert num_layers > 1, "Number of layers should be greater than 1"
        
        # Initialize the GAT layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(pyg_nn.GCNConv(in_features, hidden_features))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(pyg_nn.GCNConv(hidden_features, hidden_features))

        # Output layer
        self.layers.append(pyg_nn.GCNConv(hidden_features, out_features))
    
    def forward(self, x, edge_list):
        # Create a subset of vertices
        x = x.squeeze()
        
        # Pass the subset of vertices and edges to the GCN layers
        # Assuming self.layers is a list of GCN layers in your model
        for layer in self.layers:
            x = layer(x, edge_list)

        return x


class NLayerGAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(NLayerGAT, self).__init__()
        assert num_layers > 1, "Number of layers should be greater than 1"

        # Initialize the GAT layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GATConv(in_features, hidden_features))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_features, hidden_features))

        # Output layer
        self.layers.append(GATConv(hidden_features, out_features))

    def forward(self, x, edge_list):
        x = x.squeeze()
        assert x.dim() == 2, f"Input should be 2D, but got shape {x.shape}"
        # Apply all but the last GAT layer with ReLU activation
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_list))

        # Apply the last GAT layer with tanh activation
        x = torch.tanh(self.layers[-1](x, edge_list))

        return x

class NodeFeatureNet(nn.Module):
    def __init__(self, C=128, K=5, n_scale=1):
        super(NodeFeatureNet, self).__init__()
        # mlp layers
        self.fc1 = nn.Linear(6, C)
        self.fc2 = nn.Linear(C*2, C*4)
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
        z_local = z_local.view(-1, self.m, self.C)
        z_local = self.localfc(z_local)
        
        # point feature
        normal = compute_normal(v,self.f)
        x = torch.cat([v, normal], 2)
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
        self.initialized == True
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
    def __init__(self, C=128, K=5, n_scale=3,sf=1.0,gnn_layers=2,gnnVersion=2):
        super(DeformBlockGNN, self).__init__()
        self.nodeFeatureNet = NodeFeatureNet(C=C,K=K,n_scale=n_scale)
        #chatgpt,declare a custom 2 layer gcn here that takes features from nodeFeatureNet to predict node features, use a tanh nonlinearity at the end instead of softmax. 
        self.n_scale = n_scale
        self.C = C
        self.K = K
        self.sf = sf
        if gnnVersion == 1:
            self.gcn = NLayerGAT(C*2, C, 3,gnn_layers)  # Adjust dimensions as needed
            print('NLayerGAT',gnn_layers)
        else:
            self.gcn = NLayerGCN(C*2, C, 3,gnn_layers)  # Adjust dimensions as needed
            print('NLayerGCN',gnn_layers)
    
    def set_data(self, x, V,f=None,edge_list=None):
        # x: coordinats
        # V: input brain MRI volume
        self.nodeFeatureNet.set_data(x,V,f=f,edge_list=edge_list)
        self.f = f
        self.V = V
        self.edge_list = edge_list
    
    def forward(self, v):
        x = self.nodeFeatureNet(v)
        dx = self.gcn(x, self.edge_list)*self.sf #threshold the deformation like before
        return dx

class CSRFnet(nn.Module):
    """
    The deformation network of CortexODE model.

    dim_in: input dimension
    dim_h (C): hidden dimension
    kernel_size (K): size of convolutional kernels
    n_scale (Q): number of scales of the multi-scale input
    """
    
    def __init__(self, dim_in=3,
                       dim_h=128,
                       kernel_size=5,
                       n_scale=3,
                       sf=1.0,
                       gnn_layers=5,
                       gnnVersion=2):
        
        super(CSRFnet, self).__init__()

        C = dim_h        # hidden dimension
        K = kernel_size  # kernel size
        

        self.block1 = DeformBlockGNN(C, K, n_scale,sf,gnn_layers=gnn_layers,gnnVersion=gnnVersion)#chatgpt, i'm merging this and all of its class dependencies into this file, help me 
        
    def set_data(self, x, V,f=None):
        # x: coordinats
        # V: input brain MRI volume
        assert x.shape[0] == 1
        assert f.shape[0] == 1
        assert x.shape[1] != 1
        assert f.shape[1] != 1
        self.f = f
        edge_list = torch.cat([self.f[0,:,[0,1]],
                               self.f[0,:,[1,2]],
                               self.f[0,:,[2,0]]], dim=0).transpose(1,0)#moved from after x

        edge_list = add_self_loops(edge_list)[0]
        
        self.edge_list = edge_list
        
        self.block1.set_data(x,V,f=f,edge_list=edge_list)
        
    #this method gets called by odeint, and thus the method signature has t in it even though the t is ignored
    def forward(self, t, x):
        
        dx = self.block1(x)
        
        #todo cleanup
        # # local feature
        # z_local = self.cube_sampling(x)
        # z_local = self.localconv(z_local)
        # z_local = z_local.view(-1, self.m, self.C)
        # z_local = self.localfc(z_local)
        
        # # point feature
        # z_point = F.leaky_relu(self.fc1(x), 0.2)
        
        # # feature fusion
        # z = torch.cat([z_point, z_local], 2)
        # z = F.leaky_relu(self.fc2(z), 0.2)
        # z = F.leaky_relu(self.fc3(z), 0.2)
        # dx = self.fc4(z)
        #print ('dx shape',dx.shape)
        dx = dx.unsqueeze(0)
        return dx
    
    