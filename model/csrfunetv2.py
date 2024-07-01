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
    def __init__(self, C=128, K=1, use_pytorch3d_normal=True):
        super(NodeFeatureNet, self).__init__()
        self.use_pytorch3d_normal = use_pytorch3d_normal
        self.fc1 = nn.Linear(6, C)  # Adjusted input size
        self.fc2 = nn.Linear(2 * C, C * 4)
        self.fc3 = nn.Linear(C * 4, C * 2)

        self.localfc = nn.Linear(C , C)  # Adjust input size to match concatenated features
        self.unet = Unet(c_in=1, c_out=C)  # Updated to match original specification

        self.C = C
        self.K = K
        self.initialized = False
        self.cubes = torch.zeros([1, 1, K, K, K])  # No additional scales needed

    def forward(self, v):
        # print("forward pass start")
        z_local = self.grid_sample_unet(v)
        # print('z_local shape after grid_sample_unet:', z_local.shape)
        
        z_local = z_local.unsqueeze(0)
        z_local = self.localfc(z_local)
        # print('z_local shape after localfc:', z_local.shape)

        if not self.use_pytorch3d_normal:
            normal = compute_normal(v, self.f)
        else:
            mesh = Meshes(verts=v, faces=self.f)
            normal = mesh.verts_normals_packed()
            normal = normal.unsqueeze(0)

        x = torch.cat([v, normal], 2)
        # print('x shape after concat:', x.shape)
        z_point = F.leaky_relu(self.fc1(x), 0.2)
        # print('z_point shape after fc1:', z_point.shape)

        z = torch.cat([z_point, z_local], 2)
        # print('z shape after concat:', z.shape)
        z = F.leaky_relu(self.fc2(z), 0.2)
        # print('z shape after fc2:', z.shape)
        z = F.leaky_relu(self.fc3(z), 0.2)
        # print('z shape after fc3:', z.shape)

        return z

    def set_data(self, x, V, f=None, edge_list=None):
        if not self.initialized:
            self._initialize(V)
        self.f = f
        self.edge_list = edge_list

        D1, D2, D3 = V[0, 0].shape
        D = max([D1, D2, D3])
        self.rescale = torch.Tensor([D3/D, D2/D, D1/D]).to(V.device)
        self.D = D

        self.m = x.shape[1]
        self.neighbors = self.cubes.repeat(self.m, 1, 1, 1, 1)

        self.V_features = self.unet(V)
        # print('V_features shape after unet:', self.V_features.shape)

    def grid_sample_unet(self, x):
        # x: coordinates
        xq = x.contiguous().view(1, -1, 3).unsqueeze(-2).unsqueeze(-2)
        xq = xq / self.rescale  # Rescale the coordinates
        # print('xq max',torch.max(xq))
        # print('xq shape:', xq.shape)
        vq = F.grid_sample(self.V_features, xq, mode='bilinear', padding_mode='border', align_corners=True)
        # print('vq shape after grid_sample:', vq.shape)
        vq = vq.squeeze()
        vq = vq.permute(1,0)
        # print('vq shape after permute(1,0):', vq.shape)
        return vq

    def _initialize(self, V):
        self.cubes = self.cubes.to(V.device)
        self.initialized = True

class DeformBlockGNN(nn.Module):
    def __init__(self, C=128, K=1, sf=.1, gnn_layers=2, use_gcn=True, gat_heads=8, use_pytorch3d_normal=True):
        super(DeformBlockGNN, self).__init__()
        self.sf = sf
        self.nodeFeatureNet = NodeFeatureNet(C=C, K=K, use_pytorch3d_normal=use_pytorch3d_normal)
        self.gnn = DeformationGNN(input_features=C * 2, hidden_features=C, output_dim=3, num_layers=gnn_layers, gat_heads=gat_heads, use_gcn=use_gcn)

    def set_data(self, x, V, f=None, edge_list=None):
        self.nodeFeatureNet.set_data(x, V, f=f, edge_list=edge_list)
        self.f = f
        self.V = V
        self.edge_list = edge_list

    def forward(self, v):
        x = self.nodeFeatureNet(v)
        x = x.squeeze()
        dx = self.gnn(x, self.edge_list) * self.sf
        return dx

class CSRFUnetV2(nn.Module):
    def __init__(self, dim_h=128, kernel_size=5, n_scale=1, sf=.1, gnn_layers=5, use_gcn=True, gat_heads=8, use_pytorch3d_normal=True):
        super(CSRFUnetV2, self).__init__()
        C = dim_h
        self.block1 = DeformBlockGNN(C, kernel_size, sf, gnn_layers=gnn_layers, use_gcn=use_gcn, gat_heads=gat_heads, use_pytorch3d_normal=use_pytorch3d_normal)

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
