import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator, TimeEncoder
from torch_geometric.utils import add_self_loops
from pytorch3d.structures import Meshes

# Placeholder for compute_normal function.
def compute_normal(v, f):
    """
    Compute vertex normals for a mesh.
    Replace this placeholder with your actual implementation or import.
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

class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = t - last_update[edge_index[0]]
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        if msg is None:
            msg = torch.zeros(edge_index.size(1), self.time_enc.out_channels, device=x.device)
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)

class DeformBlockGNN(nn.Module):
    def __init__(self, C=128, K=5, n_scale=1, sf=0.1, num_classes=10, use_pytorch3d_normal=True, temporal=False):
        super(DeformBlockGNN, self).__init__()
        self.sf = sf
        self.temporal = temporal
        self.C = C
        self.num_classes = num_classes
        self.use_pytorch3d_normal = use_pytorch3d_normal
        self.edge_attr = None

        if self.temporal:
            # Initialize temporal components
            memory_dim = C
            time_dim = C
            embedding_dim = C
            msg_dim = C

            self.time_enc = TimeEncoder(time_dim)
            self.memory = None  # Will be initialized per sample in set_data

            self.gnn = GraphAttentionEmbedding(
                in_channels=memory_dim,
                out_channels=embedding_dim,
                msg_dim=msg_dim,
                time_enc=self.time_enc,
            )

            self.fc_deform = nn.Linear(embedding_dim, 3)  # Deformation vector
            self.fc_class = nn.Linear(embedding_dim, num_classes)  # Class logits
        else:
            # Original GNN
            self.nodeFeatureNet = NodeFeatureNet(C=C, K=K, n_scale=n_scale,
                                                 use_pytorch3d_normal=use_pytorch3d_normal)
            # Define your non-temporal GNN here
            # self.gnn = ...

    def set_data(self, x, V, f=None, edge_list=None, t=None):
        if self.temporal:
            self.edge_index = edge_list.to(x.device)
            self.src = self.edge_index[0]
            self.dst = self.edge_index[1]
            
            # Initialize edge_attr if it's None
            if self.edge_attr is None:
                num_edges = self.edge_index.size(1)
                msg_dim = self.C  # Assuming the same dimension as your node features
                self.edge_attr = torch.zeros(num_edges, msg_dim, device=x.device)  # Placeholder edge attributes

            self.t = t  # Timestamps for each edge
            self.x = x.squeeze(0).to(x.device)
            self.f = f

            num_nodes = x.shape[1]

            # Re-initialize memory with correct num_nodes per sample
            memory_dim = self.C
            time_dim = self.C
            msg_dim = self.C

            self.memory = TGNMemory(
                num_nodes=num_nodes,
                raw_msg_dim=msg_dim,
                memory_dim=memory_dim,
                time_dim=time_dim,
                message_module=IdentityMessage(
                    raw_msg_dim=msg_dim,
                    memory_dim=memory_dim,
                    time_dim=time_dim,
                ),
                aggregator_module=LastAggregator(),
            ).to(x.device)

            self.memory.reset_state()

            # Ensure modules are on the correct device
            self.time_enc = self.time_enc.to(x.device)
            self.gnn = self.gnn.to(x.device)
            self.fc_deform = self.fc_deform.to(x.device)
            self.fc_class = self.fc_class.to(x.device)
        else:
            # Non-temporal
            self.nodeFeatureNet.set_data(x, V, f=f, edge_list=edge_list)
            self.f = f
            self.edge_list = edge_list

    def forward(self, x, t, hidden_states=None):
        if self.temporal:
            # Ensure t is a tensor
            if not isinstance(t, torch.Tensor):
                t = torch.tensor([t], device=x.device, dtype=torch.float32)

            # Temporal processing
            if self.memory is None:
                raise ValueError("Memory not initialized. Call set_data first.")

            if hidden_states is not None:
                self.memory.memory = hidden_states
            else:
                self.memory.reset_state()

            num_nodes = x.shape[1]
            n_id = torch.arange(num_nodes, device=x.device)
            last_update = self.memory.last_update[n_id]

            # GNN forward pass
            z = self.gnn(self.memory.memory[n_id], last_update, self.edge_index, t, self.edge_attr)

            # Compute outputs
            deformation_vectors = self.fc_deform(z)
            class_logits = self.fc_class(z)
            deformation_vectors = deformation_vectors * self.sf

            # Update memory
            t_edge = t.repeat(self.src.size(0))  # Broadcast t to match the number of edges

            # Update state without keyword argument
            self.memory.update_state(self.src, self.dst, t_edge, self.edge_attr)

            return deformation_vectors, class_logits, self.memory.memory.clone()



class TCSRVC(nn.Module):
    """
    The deformation network of CortexODE model, combined with classification, using a Temporal GNN.
    """

    def __init__(self, dim_h=128,
                       kernel_size=5,
                       n_scale=1,
                       sf=0.1,
                       num_classes=10,
                       use_pytorch3d_normal=True):
        super(TCSRVC, self).__init__()

        C = dim_h
        K = kernel_size

        # Initialize DeformBlockGNN with a temporal GNN
        self.block1 = DeformBlockGNN(C,
                                     K,
                                     n_scale,
                                     sf,
                                     num_classes=num_classes,
                                     use_pytorch3d_normal=use_pytorch3d_normal,
                                     temporal=True)  # Temporal GNN

    def set_data(self, x, V, f=None, reduced_DOF=False, t=None):
        self.f = f
        self.reduced_DOF = reduced_DOF

        edge_list = torch.cat([self.f[0, :, [0, 1]],
                               self.f[0, :, [1, 2]],
                               self.f[0, :, [2, 0]]], dim=0).t()

        edge_list = add_self_loops(edge_list)[0]

        self.edge_list = edge_list

        self.block1.set_data(x, V, f=f, edge_list=edge_list, t=t)

    def forward(self, x, t, hidden_states=None):
        # x: current vertex positions
        # t: current time step
        # hidden_states: any hidden states (if applicable)

        dx, class_logits, hidden_states = self.block1(x, t, hidden_states)
        dx = dx.unsqueeze(0)
        self.class_logits = class_logits

        # Return the deformation vector and class logits
        return dx, class_logits, hidden_states

    def get_class_logits(self):
        return self.class_logits

class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.lin = nn.Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t):
        return torch.cos(self.lin(t.unsqueeze(-1)))
