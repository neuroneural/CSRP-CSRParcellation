import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import compute_normal
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATConv

from util.mesh import compute_normal

from model.classificationgnn import ClassificationGNN
from pytorch3d.structures import Meshes

class NodeFeatureNet(nn.Module):
    def __init__(self, C=128, K=5, n_scale=1, use_pytorch3d=True):
        super(NodeFeatureNet, self).__init__()
        # mlp layers
        self.use_pytorch3d = use_pytorch3d
        self.fc1 = nn.Linear(6, C)
        self.fc2 = nn.Linear(2 * C, C * 4)
        self.fc3 = nn.Linear(C * 4, C * 2)
        
        # for local convolution operation
        self.localconv = nn.Conv3d(n_scale, C, (K, K, K))
        self.localfc = nn.Linear(C, C)
        
        Q = n_scale  # number of scales
        
        self.n_scale = n_scale
        self.K = K        
        self.C = C
        self.Q = Q
        # for cube sampling
        self.initialized = False
        grid = np.linspace(-K // 2, K // 2, K)
        grid_3d = np.stack(np.meshgrid(grid, grid, grid), axis=0).transpose(2, 1, 3, 0)
        self.x_shift = torch.Tensor(grid_3d.copy()).view(-1, 3)
        self.cubes = torch.zeros([1, self.Q, self.K, self.K, self.K])

    def forward(self, v):
        z_local = self.cube_sampling(v)
        z_local = self.localconv(z_local)
        z_local = F.leaky_relu(z_local, 0.2)
        z_local = z_local.view(-1, self.m, self.C)
        z_local = self.localfc(z_local)
        z_local = F.leaky_relu(z_local, 0.2)
        
        if not self.use_pytorch3d:
            normal = compute_normal(v, self.f)
        else:
            mesh = Meshes(verts=v, faces=self.f)
            normal = mesh.verts_normals_packed()
            normal = normal.unsqueeze(0)
        
        x = torch.cat([v, normal], 2)
        z_point = F.leaky_relu(self.fc1(x), 0.2)
        
        # feature fusion
        z = torch.cat([z_point, z_local], 2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        
        return z  # node features

    def _initialize(self, V):
        self.x_shift = self.x_shift.to(V.device)
        self.cubes = self.cubes.to(V.device)
        self.initialized = True

    def set_data(self, x, V, f=None, edge_list=None):
        if not self.initialized:
            self._initialize(V)
        self.f = f
        self.edge_list = edge_list
        D1, D2, D3 = V[0, 0].shape
        D = max([D1, D2, D3])
        self.rescale = torch.Tensor([D3 / D, D2 / D, D1 / D]).to(V.device)
        self.D = D
        self.m = x.shape[1]
        self.neighbors = self.cubes.repeat(self.m, 1, 1, 1, 1)
        self.Vq = [V]
        for q in range(1, self.Q):
            self.Vq.append(F.avg_pool3d(self.Vq[-1], 2))

    def cube_sampling(self, x):
        with torch.no_grad():
            for q in range(self.Q):
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2**q)
                xq = xq.contiguous().view(1, -1, 3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale
                vq = F.grid_sample(self.Vq[q], xq, mode='bilinear', padding_mode='border', align_corners=True)
                self.neighbors[:, q] = vq[0, 0].view(self.m, self.K, self.K, self.K)
        return self.neighbors.clone()

class ClassificationBlockGNN(nn.Module):
    def __init__(self, C=128, K=5, n_scale=3, gnn_layers=2, use_gcn=True, gat_heads=8, use_pytorch3d=True, num_classes=10):
        super(ClassificationBlockGNN, self).__init__()
        self.nodeFeatureNet = NodeFeatureNet(C=C, K=K, n_scale=n_scale, use_pytorch3d=use_pytorch3d)
        self.gnn = ClassificationGNN(input_features=C * 2,  # Adjust based on NodeFeatureNet output
                                     hidden_features=C,
                                     num_classes=num_classes,  # Number of classes
                                     num_layers=gnn_layers,
                                     gat_heads=gat_heads,
                                     use_gcn=use_gcn)

    def set_data(self, x, V, f=None, edge_list=None):
        self.nodeFeatureNet.set_data(x, V, f=f, edge_list=edge_list)
        self.f = f
        self.V = V
        self.edge_list = edge_list

    def forward(self, v):
        x = self.nodeFeatureNet(v)
        x = x.squeeze()
        logits = self.gnn(x, self.edge_list)
        return logits

class CSRVCNet(nn.Module):
    """
    The classification network for vertex classification.
    """

    def __init__(self, dim_h=128,
                       kernel_size=5,
                       n_scale=3,
                       gnn_layers=5,
                       use_gcn=True,
                       gat_heads=8,
                       use_pytorch3d=True,
                       num_classes=10):
        
        super(CSRVCNet, self).__init__()

        C = dim_h        # hidden dimension
        K = kernel_size  # kernel size
        
        self.block1 = ClassificationBlockGNN(C, 
                                             K, 
                                             n_scale,
                                             gnn_layers=gnn_layers,
                                             use_gcn=use_gcn,
                                             gat_heads=gat_heads,
                                             use_pytorch3d=use_pytorch3d,
                                             num_classes=num_classes)

    def set_data(self, x, V, f=None, reduced_DOF=False):
        assert x.shape[0] == 1
        assert f.shape[0] == 1
        assert x.shape[1] != 1
        assert f.shape[1] != 1
        self.f = f
        self.reduced_DOF = reduced_DOF
        edge_list = torch.cat([self.f[0, :, [0, 1]],
                               self.f[0, :, [1, 2]],
                               self.f[0, :, [2, 0]]], dim=0).transpose(1, 0)

        edge_list = add_self_loops(edge_list)[0]
        
        self.edge_list = edge_list
        
        self.block1.set_data(x, V, f=f, edge_list=edge_list)
        
    def forward(self, x):
        logits = self.block1(x)
        logits = logits.unsqueeze(0)
        return logits
