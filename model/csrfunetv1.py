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

# Updated U-Net with 64 output channels
class Unet(nn.Module):
    def __init__(self, c_in=1, c_out=64):
        super(Unet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=c_in, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.deconv4 = nn.Conv3d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.Conv3d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.lastconv1 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.lastconv2 = nn.Conv3d(in_channels=16, out_channels=c_out, kernel_size=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        x  = F.leaky_relu(self.conv5(x4), 0.2)
        x  = self.up(x)

        x = torch.cat([x, x4], dim=1)
        x = F.leaky_relu(self.deconv4(x), 0.2)
        x = self.up(x)

        x = torch.cat([x, x3], dim=1)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = self.up(x)

        x = torch.cat([x, x2], dim=1)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = self.up(x)

        x = torch.cat([x, x1], dim=1)
        x = F.leaky_relu(self.deconv1(x), 0.2)

        x = F.leaky_relu(self.lastconv1(x), 0.2)
        x = self.lastconv2(x)

        return x

class NodeFeatureNet(nn.Module):
    def __init__(self, C=128, K=5, n_scale=1, use_pytorch3d_normal=True):
        super(NodeFeatureNet, self).__init__()
        # MLP layers
        self.use_pytorch3d_normal = use_pytorch3d_normal
        self.fc1 = nn.Linear(6, C)
        self.fc2 = nn.Linear(2 * C, C * 4)
        self.fc3 = nn.Linear(C * 4, C * 2)
        
        # For local convolution operation
        self.localconv = nn.Conv3d(n_scale, C, (K, K, K))
        self.localfc = nn.Linear(C, C)
        
        self.K = K        
        self.C = C
        self.Q = n_scale
        # For cube sampling
        self.initialized = False
        grid = np.linspace(-K//2, K//2, K)
        grid_3d = np.stack(np.meshgrid(grid, grid, grid), axis=0).transpose(2, 1, 3, 0)
        self.x_shift = torch.Tensor(grid_3d.copy()).view(-1, 3)
        self.cubes = torch.zeros([1, self.Q, self.K, self.K, self.K])

        self.unet = Unet(c_in=1, c_out=64)
    
    def forward(self, v, V):
        V_features = self.unet(V)
        # print('V_features after U-Net:', V_features.shape)
        
        z_local = self.cube_sampling(v, V_features)
        # print('z_local after cube sampling:', z_local.shape)
        
        z_local = self.localconv(z_local)
        # print('z_local after local conv:', z_local.shape)
        
        z_local = F.leaky_relu(z_local, 0.2)
        # print('z_local after ReLU:', z_local.shape)
        
        z_local = z_local.view(-1, self.m, self.C)
        # print('z_local after view:', z_local.shape)
        
        z_local = self.localfc(z_local)
        # print('z_local after local FC:', z_local.shape)
        
        z_local = F.leaky_relu(z_local, 0.2)
        # print('z_local after ReLU:', z_local.shape)
        
        # Point feature
        if not self.use_pytorch3d_normal:
            normal = compute_normal(v, self.f)
        else:
            mesh = Meshes(verts=v, faces=self.f)
            normal = mesh.verts_normals_packed()
            normal = normal.unsqueeze(0)
        
        x = torch.cat([v, normal], 2)
        # print('x after cat with normals:', x.shape)
        
        z_point = F.leaky_relu(self.fc1(x), 0.2)
        # print('z_point after fc1:', z_point.shape)
        
        # Feature fusion
        z = torch.cat([z_point, z_local], 2)
        # print('z after cat z_point and z_local:', z.shape)
        
        z = F.leaky_relu(self.fc2(z), 0.2)
        # print('z after fc2:', z.shape)
        
        z = F.leaky_relu(self.fc3(z), 0.2)
        # print('z after fc3:', z.shape)
        
        return z
    
    def _initialize(self, V):
        # Initialize coordinates shift and cubes
        self.x_shift = self.x_shift.to(V.device)
        self.cubes = self.cubes.to(V.device)
        self.initialized = True
    
    def set_data(self, x, V, f=None, edge_list=None):
        # x: coordinates
        # V: input brain MRI volume
        if not self.initialized:
            self._initialize(V)
        self.f = f
        self.edge_list = edge_list
        
        # Set the shape of the volume
        D1, D2, D3 = V[0, 0].shape
        D = max([D1, D2, D3])
        # Rescale for grid sampling
        self.rescale = torch.Tensor([D3/D, D2/D, D1/D]).to(V.device)
        self.D = D

        self.m = x.shape[1]  # Number of points
        self.neighbors = self.cubes.repeat(self.m, 1, 1, 1, 1)  # Repeat m cubes
        
        # Set multi-scale volume
        self.Vq = [V]
        for q in range(1, self.Q):
            # Iteratively downsampling
            self.Vq.append(F.avg_pool3d(self.Vq[-1], 2))

    def cube_sampling(self, x, V_features):
        # x: coordinates
        with torch.no_grad():
            for q in range(self.Q):
                # Make sure the cubes have the same size
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2**q)
                xq = xq.contiguous().view(1, -1, 3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale  # Rescale the coordinates
                # print('xq shape:', xq.shape)
                
                # Sample the q-th cube
                vq = F.grid_sample(V_features, xq, mode='bilinear', padding_mode='border', align_corners=True)
                # print('vq shape after grid_sample:', vq.shape)
                
                # Update the cubes
                self.neighbors[:, q] = vq[0, 0].view(self.m, self.K, self.K, self.K)
        
        return self.neighbors.clone()

class DeformBlockGNN(nn.Module):
    def __init__(self, C=128, K=5, n_scale=3, sf=.1, gnn_layers=2, use_gcn=True, gat_heads=8, use_pytorch3d_normal=True):
        super(DeformBlockGNN, self).__init__()
        self.sf = sf
        self.nodeFeatureNet = NodeFeatureNet(C=C, K=K, n_scale=n_scale, use_pytorch3d_normal=use_pytorch3d_normal)
        self.gnn = DeformationGNN(input_features=C * 2, hidden_features=C, output_dim=3, num_layers=gnn_layers, gat_heads=gat_heads, use_gcn=use_gcn)

    def set_data(self, x, V, f=None, edge_list=None):
        self.nodeFeatureNet.set_data(x, V, f=f, edge_list=edge_list)
        self.f = f
        self.V = V
        self.edge_list = edge_list

    def forward(self, v):
        x = self.nodeFeatureNet(v, self.V)
        x = x.squeeze()
        dx = self.gnn(x, self.edge_list) * self.sf
        return dx

class CSRFUnetV1(nn.Module):
    def __init__(self, dim_h=128, kernel_size=5, n_scale=3, sf=.1, gnn_layers=5, use_gcn=True, gat_heads=8, use_pytorch3d_normal=True):
        super(CSRFUnetV1, self).__init__()
        C = dim_h
        K = kernel_size

        self.block1 = DeformBlockGNN(C, K, n_scale, sf, gnn_layers=gnn_layers, use_gcn=use_gcn, gat_heads=gat_heads, use_pytorch3d_normal=use_pytorch3d_normal)

    def set_data(self, x, V, f=None, reduced_DOF=False):
        assert x.shape[0] == 1
        assert f.shape[0] == 1
        assert x.shape[1] != 1
        assert f.shape[1] != 1
        self.f = f
        self.reduced_DOF = reduced_DOF
        edge_list = torch.cat([self.f[0, :, [0, 1]], self.f[0, :, [1, 2]], self.f[0, :, [2, 0]]], dim=0).transpose(1, 0)
        edge_list = add_self_loops(edge_list)[0]
        self.edge_list = edge_list
        self.block1.set_data(x, V, f=f, edge_list=edge_list)

    def forward(self, t, x):
        dx = self.block1(x)
        dx = dx.unsqueeze(0)
        return dx
