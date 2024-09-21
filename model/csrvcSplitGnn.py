import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops
from pytorch3d.structures import Meshes

def compute_normal(v, f):
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
        
        self.localconv = nn.Conv3d(n_scale, C, (K, K, K))
        self.localfc = nn.Linear(C, C)
        
        self.n_scale = n_scale
        self.K = K        
        self.C = C
        self.Q = n_scale  # Number of scales
        
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

class SharedGNN(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers, use_gcn=True, gat_heads=2):
        super(SharedGNN, self).__init__()
        self.layers = nn.ModuleList()
        current_dim = input_features

        for _ in range(num_layers):
            if use_gcn:
                conv = GCNConv(current_dim, hidden_features)
            else:
                conv = GATConv(current_dim, hidden_features, heads=gat_heads)
            self.layers.append(conv)
            self.layers.append(nn.BatchNorm1d(hidden_features))
            self.activation = nn.LeakyReLU(0.2)
            current_dim = hidden_features if use_gcn else hidden_features * gat_heads

    def forward(self, x, edge_index):
        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x, edge_index)
            x = self.activation(x)
            x = self.layers[i+1](x)
        return x

class DeformationGNN(nn.Module):
    def __init__(self, input_features, hidden_features, output_dim=3, num_layers=5, use_gcn=True, gat_heads=2):
        super(DeformationGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.use_gcn = use_gcn
        current_dim = input_features

        for _ in range(num_layers):
            if use_gcn:
                conv = GCNConv(current_dim, hidden_features)
            else:
                conv = GATConv(current_dim, hidden_features, heads=gat_heads)
            self.layers.append(conv)
            self.layers.append(nn.BatchNorm1d(hidden_features))
            self.layers.append(nn.Dropout(p=0.5))
            self.activation = nn.LeakyReLU(0.2)
            current_dim = hidden_features if use_gcn else hidden_features * gat_heads

        # Output layer
        if use_gcn:
            self.output_layer = GCNConv(current_dim, output_dim)
        else:
            self.output_layer = GATConv(current_dim, output_dim, heads=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        for i in range(0, len(self.layers), 3):
            residual = x
            x = self.layers[i](x, edge_index)
            x = self.activation(x)
            x = self.layers[i+1](x)
            x = self.layers[i+2](x)
            x = x + residual  # Residual connection
        
        dx = self.output_layer(x, edge_index)
        
        # If using GAT with multiple heads, average the outputs
        if isinstance(self.output_layer, GATConv) and self.output_layer.heads > 1:
            dx = dx.mean(dim=1)
        
        return dx

class ClassificationGNN(nn.Module):
    def __init__(self, input_features, hidden_features, num_classes=10, num_layers=5, use_gcn=True, gat_heads=2):
        super(ClassificationGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.use_gcn = use_gcn
        current_dim = input_features

        for _ in range(num_layers):
            if use_gcn:
                conv = GCNConv(current_dim, hidden_features)
            else:
                conv = GATConv(current_dim, hidden_features, heads=gat_heads)
            self.layers.append(conv)
            self.layers.append(nn.BatchNorm1d(hidden_features))
            self.layers.append(nn.Dropout(p=0.5))
            self.activation = nn.LeakyReLU(0.2)
            current_dim = hidden_features if use_gcn else hidden_features * gat_heads

        # Output layer
        if use_gcn:
            self.output_layer = GCNConv(current_dim, num_classes)
        else:
            self.output_layer = GATConv(current_dim, num_classes, heads=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        for i in range(0, len(self.layers), 3):
            residual = x
            x = self.layers[i](x, edge_index)
            x = self.activation(x)
            x = self.layers[i+1](x)
            x = self.layers[i+2](x)
            x = x + residual  # Residual connection
        
        class_logits = self.output_layer(x, edge_index)
        
        # If using GAT with multiple heads, average the outputs
        if isinstance(self.output_layer, GATConv) and self.output_layer.heads > 1:
            class_logits = class_logits.mean(dim=1)
        
        # Apply log_softmax for classification
        class_logits = F.log_softmax(class_logits, dim=1)
        
        return class_logits

class DeformBlockGNN(nn.Module):
    def __init__(self, C=128, K=5, n_scale=1, sf=0.1, total_layers=10, use_gcn=True, gat_heads=2, num_classes=10, use_pytorch3d_normal=True):
        super(DeformBlockGNN, self).__init__()
        
        # Assertion to ensure total_layers is even
        assert total_layers % 2 == 0, "Total number of layers must be even."
        
        self.sf = sf
        self.nodeFeatureNet = NodeFeatureNet(C=C, K=K, n_scale=n_scale, use_pytorch3d_normal=use_pytorch3d_normal)
        
        # Calculate number of shared layers
        shared_layers = total_layers // 2
        
        # Shared GNN layers
        self.shared_gnn = SharedGNN(
            input_features=C * 2,
            hidden_features=C,
            num_layers=shared_layers,
            use_gcn=use_gcn,
            gat_heads=gat_heads
        )
        
        # Task-specific GNN layers (half the total layers)
        task_layers = shared_layers
        
        self.deform_gnn = DeformationGNN(
            input_features=C,
            hidden_features=C,
            output_dim=3,
            num_layers=task_layers,
            use_gcn=use_gcn,
            gat_heads=gat_heads
        )
        
        self.class_gnn = ClassificationGNN(
            input_features=C,
            hidden_features=C,
            num_classes=num_classes,
            num_layers=task_layers,
            use_gcn=use_gcn,
            gat_heads=gat_heads
        )
        
    def set_data(self, x, V, f=None, edge_list=None):
        self.nodeFeatureNet.set_data(x, V, f=f, edge_list=edge_list)
        self.f = f
        self.edge_list = edge_list

    def forward(self, v):
        x = self.nodeFeatureNet(v).squeeze()
        shared_features = self.shared_gnn(x, self.edge_list)
        dx = self.deform_gnn(shared_features, self.edge_list) * self.sf
        class_logits = self.class_gnn(shared_features, self.edge_list)
        return dx, class_logits

class CSRVCSPLITGNN(nn.Module):
    """
    The deformation network of CortexODE model, combined with classification.
    """
    def __init__(self, dim_h=128,
                       kernel_size=5,
                       n_scale=1,
                       sf=0.1,
                       gnn_layers=10,  # Must be even
                       use_gcn=True,
                       gat_heads=2,
                       use_pytorch3d_normal=True,
                       num_classes=10):
        
        super(CSRVCSPLITGNN, self).__init__()

        C = dim_h
        K = kernel_size

        self.block1 = DeformBlockGNN(
            C, 
            K, 
            n_scale,
            sf,
            total_layers=gnn_layers,  # Must be even
            use_gcn=use_gcn,
            gat_heads=gat_heads,
            num_classes=num_classes,
            use_pytorch3d_normal=use_pytorch3d_normal
        )
        
        # Initialize log variance parameters for uncertainty weighting
        self.log_var_deform = nn.Parameter(torch.zeros(1))
        self.log_var_class = nn.Parameter(torch.zeros(1))
        
        # Internal variable to store class logits
        self.class_logits = None
        
    def set_data(self, x, V, f=None, reduced_DOF=False):
        assert x.shape[0] == 1, "Batch size for x must be 1."
        assert f.shape[0] == 1, "Batch size for f must be 1."
        assert x.shape[1] != 1, "Number of points in x must not be 1."
        assert f.shape[1] != 1, "Number of faces in f must not be 1."
        self.f = f
        self.reduced_DOF = reduced_DOF

        edge_list = torch.cat([
            self.f[0, :, [0, 1]],
            self.f[0, :, [1, 2]],
            self.f[0, :, [2, 0]]
        ], dim=0).t()

        edge_list, _ = add_self_loops(edge_list)
        
        self.edge_list = edge_list
        
        self.block1.set_data(x, V, f=f, edge_list=edge_list)
        
    def forward(self, t, x):
        dx, class_logits = self.block1(x)
        dx = dx.unsqueeze(0)
        
        # Store the class logits for later retrieval
        self.class_logits = class_logits
        
        return dx#, class_logits use get_class_logits
    
    def get_class_logits(self):
        """
        Retrieve the stored class logits from the last forward pass.
        """
        return self.class_logits
