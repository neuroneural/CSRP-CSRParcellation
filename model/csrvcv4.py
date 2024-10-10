import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops
from pytorch3d.structures import Meshes

from torch_geometric.utils import dropout_edge

# Assuming compute_normal is available from util.mesh or elsewhere
# If not, you need to implement compute_normal or adjust accordingly
# from util.mesh import compute_normal

def compute_normal(v, f):
    """
    Placeholder for compute_normal function.
    You should replace this with your actual implementation or import.
    """
    mesh = Meshes(verts=v, faces=f)
    normals = mesh.verts_normals_packed()
    return normals.unsqueeze(0)

class NodeFeatureNet(nn.Module):
    def __init__(self, C=128, K=5, n_scale=1, use_pytorch3d_normal=True):
        super(NodeFeatureNet, self).__init__()
        self.use_pytorch3d_normal = use_pytorch3d_normal
        self.fc1 = nn.Linear(6, C)
        self.fc2 = nn.Linear(2 * C, C * 4)
        self.fc3 = nn.Linear(C * 4, C * 2)
        
        # Local convolution operation
        self.localconv = nn.Conv3d(n_scale, C, (K, K, K))
        self.localfc = nn.Linear(C, C)
        
        self.n_scale = n_scale
        self.K = K        
        self.C = C
        self.Q = n_scale  # Number of scales
        
        # Cube sampling initialization
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
        
        # Point feature
        if not self.use_pytorch3d_normal:
            normal = compute_normal(v, self.f)
        else:
            mesh = Meshes(verts=v, faces=self.f)
            normal = mesh.verts_normals_packed()
            normal = normal.unsqueeze(0)
        x = torch.cat([v, normal], 2)
        z_point = F.leaky_relu(self.fc1(x), 0.2)
        
        # Feature fusion
        z = torch.cat([z_point, z_local], 2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        
        return z  # Node features

    def _initialize(self, V):
        # Initialize coordinates shift and cubes
        self.x_shift = self.x_shift.to(V.device)
        self.cubes = self.cubes.to(V.device)
        self.initialized = True

    def set_data(self, x, V, f=None, edge_list=None):
        if not self.initialized:
            self._initialize(V)
        self.f = f
        self.edge_list = edge_list
        # Set the shape of the volume
        D1, D2, D3 = V[0, 0].shape
        D = max([D1, D2, D3])
        # Rescale for grid sampling
        self.rescale = torch.Tensor([D3 / D, D2 / D, D1 / D]).to(V.device)
        self.D = D

        self.m = x.shape[1]  # Number of points
        self.neighbors = self.cubes.repeat(self.m, 1, 1, 1, 1)  # Repeat m cubes
        
        # Set multi-scale volume
        self.Vq = [V]
        for q in range(1, self.Q):
            # Iteratively downsampling
            self.Vq.append(F.avg_pool3d(self.Vq[-1], 2))

    def cube_sampling(self, x):
        with torch.no_grad():
            for q in range(self.Q):
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2 ** q)
                xq = xq.contiguous().view(1, -1, 3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale  # Rescale the coordinates
                vq = F.grid_sample(self.Vq[q], xq, mode='bilinear', padding_mode='border', align_corners=True)
                self.neighbors[:, q] = vq[0, 0].view(self.m, self.K, self.K, self.K)
        
        return self.neighbors.clone()

class CombinedGNN(nn.Module):
    def __init__(self, input_features=256, hidden_features=128, output_dim=3, num_classes=10, num_layers=2, gat_heads=8, use_gcn=True,dropedge_rate=0.5):
        super(CombinedGNN, self).__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_features
        self.dropedge_rate = dropedge_rate

        # GNN layers
        for i in range(num_layers):
            if use_gcn:
                self.layers.append(GCNConv(current_dim, hidden_features))
                next_dim = hidden_features
            else:
                self.layers.append(GATConv(current_dim, hidden_features, heads=gat_heads))
                next_dim = hidden_features * gat_heads

            current_dim = next_dim

        # Separate linear layers for deformation and classification
        self.deformation_head = nn.Linear(current_dim, output_dim)  # Output dx
        self.classification_head = nn.Linear(current_dim, num_classes)  # Output class logits

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            edge_index, _ = dropout_edge(edge_index, p=self.dropedge_rate, training=self.training)
            x = layer(x, edge_index)
            x = F.leaky_relu(x, 0.2)
        
        # x now has dimension [num_nodes, current_dim]

        # Deformation output
        dx = self.deformation_head(x)  # Output dimension [num_nodes, 3]

        # Classification output
        class_logits = self.classification_head(x)  # Output dimension [num_nodes, num_classes]

        # Apply log_softmax to class_logits
        # class_logits = F.log_softmax(class_logits, dim=1)#this will be handled in loss now. 

        return dx, class_logits

class DeformBlockGNN(nn.Module):
    def __init__(self, C=128, K=5, n_scale=1, sf=0.1, gnn_layers=2, use_gcn=True, gat_heads=8, num_classes=10, use_pytorch3d_normal=True):
        super(DeformBlockGNN, self).__init__()
        self.sf = sf
        self.nodeFeatureNet = NodeFeatureNet(C=C, K=K, n_scale=n_scale, use_pytorch3d_normal=use_pytorch3d_normal)
        self.gnn = CombinedGNN(input_features=C * 2,
                               hidden_features=C,
                               output_dim=3,
                               num_classes=num_classes,
                               num_layers=gnn_layers,
                               gat_heads=gat_heads,
                               use_gcn=use_gcn)
        self.class_logits = None  # To store classification logits

    def set_data(self, x, V, f=None, edge_list=None):
        self.nodeFeatureNet.set_data(x, V, f=f, edge_list=edge_list)
        self.f = f
        self.edge_list = edge_list

    def forward(self, v):
        x = self.nodeFeatureNet(v)
        x = x.squeeze()
        dx, class_logits = self.gnn(x, self.edge_list)
        dx = dx * self.sf
        self.class_logits = class_logits  # Store class logits
        return dx

class CSRVCV4(nn.Module):
    """
    The deformation network of CortexODE model, combined with classification.
    """

    def __init__(self, dim_h=128,
                       kernel_size=5,
                       n_scale=1,
                       sf=0.1,
                       gnn_layers=5,
                       use_gcn=True,
                       gat_heads=8,
                       use_pytorch3d_normal=True,
                       num_classes=10):
        
        super(CSRVCV4, self).__init__()

        C = dim_h
        K = kernel_size

        self.block1 = DeformBlockGNN(C, 
                                     K, 
                                     n_scale,
                                     sf,
                                     gnn_layers=gnn_layers,
                                     use_gcn=use_gcn,
                                     gat_heads=gat_heads,
                                     num_classes=num_classes,
                                     use_pytorch3d_normal=use_pytorch3d_normal)
        
    def set_data(self, x, V, f=None, reduced_DOF=False):
        assert x.shape[0] == 1
        assert f.shape[0] == 1
        assert x.shape[1] != 1
        assert f.shape[1] != 1
        self.f = f
        self.reduced_DOF = reduced_DOF

        edge_list = torch.cat([self.f[0, :, [0, 1]],
                               self.f[0, :, [1, 2]],
                               self.f[0, :, [2, 0]],
                               self.f[0, :, [1, 0]],
                               self.f[0, :, [2, 1]],
                               self.f[0, :, [0, 2]],
                               ], dim=0).t()

        edge_list = add_self_loops(edge_list)[0]
        
        self.edge_list = edge_list
        
        self.block1.set_data(x, V, f=f, edge_list=edge_list)
        
    def forward(self, t, x):
        dx = self.block1(x)
        dx = dx.unsqueeze(0)
        return dx

    def get_class_logits(self):
        return self.block1.class_logits
