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

    Args:
        v (torch.Tensor): Vertex positions, shape [batch_size, num_vertices, 3].
        f (torch.Tensor): Face indices, shape [batch_size, num_faces, 3].

    Returns:
        torch.Tensor: Vertex normals, shape [batch_size, num_vertices, 3].
    """
    assert v.dim() == 3 and v.size(2) == 3, f"Expected v to be [batch_size, num_vertices, 3], got {v.shape}"
    assert f.dim() == 3 and f.size(2) == 3, f"Expected f to be [batch_size, num_faces, 3], got {f.shape}"
    
    mesh = Meshes(verts=v, faces=f)
    normals = mesh.verts_normals_packed()
    return normals.unsqueeze(0)  # Shape: [1, total_vertices, 3]

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
        self.x_shift = torch.Tensor(grid_3d.copy()).view(-1, 3)  # Shape: [K^3, 3]
        self.cubes = torch.zeros([1, self.Q, self.K, self.K, self.K])
    
    def forward(self, v):
        """
        Forward pass for NodeFeatureNet.

        Args:
            v (torch.Tensor): Vertex positions, shape [batch_size, num_vertices, 3].

        Returns:
            torch.Tensor: Node features, shape [batch_size, num_vertices, C*2].
        """
        assert v.dim() == 3 and v.size(2) == 3, f"Expected v to be [batch_size, num_vertices, 3], got {v.shape}"
        assert hasattr(self, 'f'), "Face indices 'f' not set. Call set_data first."
        
        z_local = self.cube_sampling(v)
        assert z_local.dim() == 5 and z_local.size(1) == self.Q, f"Expected z_local to be [batch_size, Q, K, K, K], got {z_local.shape}"
        
        z_local = self.localconv(z_local)
        assert z_local.dim() == 5 and z_local.size(1) == self.C, f"Expected z_local after conv to be [batch_size, C, K, K, K], got {z_local.shape}"
        
        z_local = F.leaky_relu(z_local, 0.2)
        z_local = z_local.view(-1, self.m, self.C)  # Shape: [batch_size, num_points, C]
        assert z_local.dim() == 3 and z_local.size(2) == self.C, f"Expected z_local to be [batch_size, num_points, C], got {z_local.shape}"
        
        z_local = self.localfc(z_local)
        z_local = F.leaky_relu(z_local, 0.2)
        assert z_local.dim() == 3 and z_local.size(2) == self.C, f"Expected z_local after localfc to be [batch_size, num_points, C], got {z_local.shape}"
        
        # Point feature
        if not self.use_pytorch3d_normal:
            normal = compute_normal(v, self.f)
            assert normal.dim() == 3 and normal.size(0) == 1, f"Expected normal to be [1, total_vertices, 3], got {normal.shape}"
            normal = normal.squeeze(0)  # Shape: [total_vertices, 3]
        else:
            mesh = Meshes(verts=v, faces=self.f)
            normal = mesh.verts_normals_packed()
            assert normal.dim() == 2 and normal.size(1) == 3, f"Expected normal to be [total_vertices, 3], got {normal.shape}"
        
        x = torch.cat([v, normal], 2)  # Shape: [batch_size, num_vertices, 6]
        assert x.dim() == 3 and x.size(2) == 6, f"Expected x to be [batch_size, num_vertices, 6], got {x.shape}"
        
        z_point = F.leaky_relu(self.fc1(x), 0.2)
        assert z_point.dim() == 3 and z_point.size(2) == self.C, f"Expected z_point to be [batch_size, num_vertices, C], got {z_point.shape}"
        
        # Feature fusion
        z = torch.cat([z_point, z_local], 2)  # Shape: [batch_size, num_vertices, 2*C]
        assert z.dim() == 3 and z.size(2) == 2 * self.C, f"Expected z to be [batch_size, num_vertices, 2*C], got {z.shape}"
        
        z = F.leaky_relu(self.fc2(z), 0.2)
        assert z.dim() == 3 and z.size(2) == self.C * 4, f"Expected z after fc2 to be [batch_size, num_vertices, C*4], got {z.shape}"
        
        z = F.leaky_relu(self.fc3(z), 0.2)
        assert z.dim() == 3 and z.size(2) == self.C * 2, f"Expected z after fc3 to be [batch_size, num_vertices, C*2], got {z.shape}"
        
        return z  # Node features: [batch_size, num_vertices, C*2]

    def _initialize(self, V):
        """
        Initialize coordinates shift and cubes.

        Args:
            V (torch.Tensor): Volumetric data, shape [batch_size, channels, D1, D2, D3].
        """
        assert V.dim() == 5, f"Expected V to be [batch_size, channels, D1, D2, D3], got {V.shape}"
        
        self.x_shift = self.x_shift.to(V.device)
        self.cubes = self.cubes.to(V.device)
        self.initialized = True

    def set_data(self, x, V, f=None, edge_list=None):
        """
        Set data for NodeFeatureNet.

        Args:
            x (torch.Tensor): Vertex positions, shape [batch_size, num_vertices, 3].
            V (torch.Tensor): Volumetric data, shape [batch_size, channels, D1, D2, D3].
            f (torch.Tensor): Face indices, shape [batch_size, num_faces, 3].
            edge_list (torch.Tensor): Edge list for graph, shape [2, num_edges].
        """
        if not self.initialized:
            self._initialize(V)
        
        self.f = f
        self.edge_list = edge_list
        
        # Set the shape of the volume
        D1, D2, D3 = V.size(2), V.size(3), V.size(4)
        D = max([D1, D2, D3])
        assert D > 0, "Volume dimensions must be positive."
        
        # Rescale for grid sampling
        self.rescale = torch.Tensor([D3 / D, D2 / D, D1 / D]).to(V.device)
        self.D = D

        self.m = x.size(1)  # Number of points
        self.neighbors = self.cubes.repeat(self.m, 1, 1, 1, 1)  # Shape: [m, Q, K, K, K]
        assert self.neighbors.shape == (self.m, self.Q, self.K, self.K, self.K), f"Expected neighbors to be [{self.m}, {self.Q}, {self.K}, {self.K}, {self.K}], got {self.neighbors.shape}"

        # Set multi-scale volume
        self.Vq = [V]
        for q in range(1, self.Q):
            # Iteratively downsampling
            self.Vq.append(F.avg_pool3d(self.Vq[-1], 2))
            assert self.Vq[q].dim() == 5, f"Expected Vq[{q}] to be 5D, got {self.Vq[q].dim()}D"

    def cube_sampling(self, x):
        """
        Sample cubes around each point in x.

        Args:
            x (torch.Tensor): Vertex positions, shape [batch_size, num_vertices, 3].

        Returns:
            torch.Tensor: Neighboring cubes, shape [m, Q, K, K, K].
        """
        assert x.dim() == 3 and x.size(2) == 3, f"Expected x to be [batch_size, num_vertices, 3], got {x.shape}"
        batch_size, num_vertices, _ = x.shape
        self.m = num_vertices  # Number of points
        
        with torch.no_grad():
            for q in range(self.Q):
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2 ** q)  # Shape: [batch_size, num_vertices, K^3, 3]
                xq = xq.contiguous().view(1, -1, 3).unsqueeze(-2).unsqueeze(-2)  # Shape: [1, num_vertices*K^3, 1, 1, 3]
                xq = xq / self.rescale  # Rescale the coordinates
                assert xq.dim() == 5 and xq.size(-1) == 3, f"Expected xq to be [1, num_vertices*K^3, 1, 1, 3], got {xq.shape}"
                
                # Use trilinear mode for 3D grid sampling
                vq = F.grid_sample(self.Vq[q], xq, mode='trilinear', padding_mode='border', align_corners=True)
                assert vq.dim() == 5, f"Expected vq to be 5D, got {vq.dim()}D"
                
                # Reshape and assign to neighbors
                self.neighbors[:, q] = vq[0, 0].view(self.m, self.K, self.K, self.K)
                assert self.neighbors[:, q].shape == (self.m, self.K, self.K, self.K), f"Expected neighbors[:, {q}] to be [{self.m}, {self.K}, {self.K}, {self.K}], got {self.neighbors[:, q].shape}"
        
        return self.neighbors.clone()  # Shape: [m, Q, K, K, K]

class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, 
            out_channels // 2, 
            heads=2,
            dropout=0.1, 
            edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        """
        Forward pass for GraphAttentionEmbedding.

        Args:
            x (torch.Tensor): Node features, shape [num_nodes, in_channels].
            last_update (torch.Tensor): Last update times, shape [num_nodes].
            edge_index (torch.Tensor): Edge indices, shape [2, num_edges].
            t (torch.Tensor): Encoded relative times, shape [num_edges, time_enc.out_channels].
            msg (torch.Tensor or None): Messages, shape [num_edges, msg_dim].

        Returns:
            torch.Tensor: Updated node features, shape [num_nodes, out_channels].
        """
        # Debug assertions
        assert x.dim() == 2, f"Expected x to be 2D, got {x.dim()}D"
        assert last_update.dim() == 1, f"Expected last_update to be 1D, got {last_update.dim()}D"
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, f"Expected edge_index to be [2, num_edges], got {edge_index.shape}"
        assert t.dim() == 2, f"Expected t to be [num_edges, C], got {t.shape}"
        if msg is not None:
            assert msg.dim() == 2, f"Expected msg to be [num_edges, C], got {msg.shape}"
        else:
            # Initialize msg if None
            msg = torch.zeros(edge_index.size(1), self.time_enc.out_channels, device=x.device)
            assert msg.dim() == 2, f"Initialized msg should be 2D, got {msg.dim()}D"
        
        # Ensure both tensors are 2D
        assert t.dim() == 2, f"rel_t_enc should be 2D, got {t.dim()}D"
        assert msg.dim() == 2, f"msg should be 2D, got {msg.dim()}D"
        
        edge_attr = torch.cat([t, msg], dim=-1)  # [num_edges, 2C]
        assert edge_attr.dim() == 2 and edge_attr.size(1) == (self.time_enc.out_channels + msg.size(1)), f"Expected edge_attr to be [num_edges, {self.time_enc.out_channels + msg.size(1)}], got {edge_attr.shape}"
        
        out = self.conv(x, edge_index, edge_attr)
        assert out.dim() == 2, f"Expected output from TransformerConv to be 2D, got {out.dim()}D"
        
        return out  # Shape: [num_nodes, out_channels]

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
            time_dim = C  # Assuming you want to encode time into C dimensions
            embedding_dim = C
            msg_dim = C

            # Initialize PyG's TimeEncoder
            self.time_enc = TimeEncoder(out_channels=time_dim)

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
            self.nodeFeatureNet = NodeFeatureNet(
                C=C, 
                K=K, 
                n_scale=n_scale,
                use_pytorch3d_normal=use_pytorch3d_normal
            )
            # Define your non-temporal GNN here
            # self.gnn = ...

    def set_data(self, x, V, f=None, edge_list=None, t=None):
        """
        Set data for DeformBlockGNN.

        Args:
            x (torch.Tensor): Vertex positions, shape [batch_size, num_vertices, 3].
            V (torch.Tensor): Volumetric data, shape [batch_size, channels, D1, D2, D3].
            f (torch.Tensor): Face indices, shape [batch_size, num_faces, 3].
            edge_list (torch.Tensor): Edge list for graph, shape [2, num_edges].
            t (torch.Tensor or None): Timestamps for each edge, shape [num_edges].
        """
        if self.temporal:
            assert edge_list is not None, "edge_list must be provided for temporal mode."
            self.edge_index = edge_list.to(x.device)
            assert self.edge_index.dim() == 2 and self.edge_index.size(0) == 2, f"Expected edge_index to be [2, num_edges], got {self.edge_index.shape}"
            self.src = self.edge_index[0]
            self.dst = self.edge_index[1]
            
            # Initialize edge_attr if it's None
            if self.edge_attr is None:
                num_edges = self.edge_index.size(1)
                msg_dim = self.C  # Assuming the same dimension as your node features
                self.edge_attr = torch.zeros(num_edges, msg_dim, device=x.device)  # Placeholder edge attributes
                assert self.edge_attr.shape == (num_edges, msg_dim), f"Expected edge_attr to be [{num_edges}, {msg_dim}], got {self.edge_attr.shape}"

            self.t = t  # Timestamps for each edge (handled per step)
            self.x = x.squeeze(0).to(x.device)  # Assuming batch_size=1 for temporal memory
            assert self.x.dim() == 2 and self.x.size(1) == 3, f"Expected x to be [num_vertices, 3], got {self.x.shape}"
            self.f = f

            num_nodes = x.size(1)
            assert num_nodes > 0, "Number of nodes must be positive."

            # Re-initialize memory with correct num_nodes per sample
            memory_dim = self.C
            time_dim = self.time_enc.out_channels
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
            assert self.memory.memory.shape == (num_nodes, memory_dim), f"Expected memory to be [{num_nodes}, {memory_dim}], got {self.memory.memory.shape}"

            # Ensure modules are on the correct device
            self.time_enc = self.time_enc.to(x.device)
            self.gnn = self.gnn.to(x.device)
            self.fc_deform = self.fc_deform.to(x.device)
            self.fc_class = self.fc_class.to(x.device)
        else:
            # Non-temporal
            assert edge_list is not None, "edge_list must be provided for non-temporal mode."
            self.nodeFeatureNet.set_data(x, V, f=f, edge_list=edge_list)
            self.f = f
            self.edge_list = edge_list

    def forward(self, x, t, hidden_states=None):
        """
        Forward pass for DeformBlockGNN.

        Args:
            x (torch.Tensor): Vertex positions, shape [num_vertices, 3].
            t (torch.Tensor or float): Current time step.
            hidden_states (torch.Tensor or None): Hidden states for memory.

        Returns:
            tuple: (deformation_vectors, class_logits, updated_memory)
        """
        if self.temporal:
            # Ensure t is a tensor of type Float
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=x.device, dtype=torch.float)
            else:
                t = t.to(x.device).to(torch.float)
            # Allow t to be a 1D tensor with length equal to the number of edges
            assert t.dim() == 1, f"Expected t to be a 1D tensor of timestamps, got {t.shape}"
    
            # Temporal processing
            if self.memory is None:
                raise ValueError("Memory not initialized. Call set_data first.")
    
            if hidden_states is not None:
                assert hidden_states.shape == self.memory.memory.shape, f"Expected hidden_states to match memory shape {self.memory.memory.shape}, got {hidden_states.shape}"
                self.memory.memory = hidden_states
            else:
                self.memory.reset_state()
    
            num_nodes = x.shape[0]
            n_id = torch.arange(num_nodes, device=x.device)
            assert n_id.shape == (num_nodes,), f"Expected n_id to be [{num_nodes}], got {n_id.shape}"
            last_update = self.memory.last_update[n_id]  # Shape: [num_nodes]
            assert last_update.dim() == 1 and last_update.size(0) == num_nodes, f"Expected last_update to be [num_nodes], got {last_update.shape}"
    
            # Compute relative time as Float for encoding, per edge
            rel_t = (t - last_update.float())[self.src]  # Shape: [num_edges]
            assert rel_t.dim() == 1 and rel_t.size(0) == self.edge_index.size(1), f"Expected rel_t to be [num_edges], got {rel_t.shape}"
    
            # Time encoding
            rel_t_enc = self.time_enc(rel_t.unsqueeze(-1))  # Shape: [num_edges, time_dim]
            assert rel_t_enc.dim() == 2 and rel_t_enc.size(1) == self.time_enc.out_channels, f"Expected rel_t_enc to be [num_edges, {self.time_enc.out_channels}], got {rel_t_enc.shape}"
    
            # GNN forward pass
            z = self.gnn(
                self.memory.memory[n_id],  # Shape: [num_nodes, memory_dim]
                last_update,              # Shape: [num_nodes]
                self.edge_index,          # Shape: [2, num_edges]
                rel_t_enc,                # Shape: [num_edges, time_dim]
                self.edge_attr            # Shape: [num_edges, msg_dim]
            )
            assert z.dim() == 2 and z.size(0) == num_nodes, f"Expected z to be [num_nodes, embedding_dim], got {z.shape}"
    
            # Compute outputs
            deformation_vectors = self.fc_deform(z)  # Shape: [num_nodes, 3]
            class_logits = self.fc_class(z)          # Shape: [num_nodes, num_classes]
            class_logits = F.log_softmax(class_logits, dim=1)
            deformation_vectors = deformation_vectors * self.sf
            assert deformation_vectors.dim() == 2 and deformation_vectors.size(1) == 3, f"Expected deformation_vectors to be [num_nodes, 3], got {deformation_vectors.shape}"
            assert class_logits.dim() == 2 and class_logits.size(1) == self.num_classes, f"Expected class_logits to be [num_nodes, {self.num_classes}], got {class_logits.shape}"
    
            # **Do not update memory here**
            return deformation_vectors, class_logits, self.memory.memory.clone()
        else:
            # Non-temporal implementation
            raise NotImplementedError("Non-temporal forward pass is not implemented yet.")

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
        self.block1 = DeformBlockGNN(
            C=C,
            K=K,
            n_scale=n_scale,
            sf=sf,
            num_classes=num_classes,
            use_pytorch3d_normal=use_pytorch3d_normal,
            temporal=True  # Temporal GNN
        )

    def set_data(self, x, V, f=None, reduced_DOF=False):
        """
        Set data for TCSRVC.

        Args:
            x (torch.Tensor): Vertex positions, shape [batch_size, num_vertices, 3].
            V (torch.Tensor): Volumetric data, shape [batch_size, channels, D1, D2, D3].
            f (torch.Tensor): Face indices, shape [batch_size, num_faces, 3].
            reduced_DOF (bool): Whether to use reduced degrees of freedom.
            t (torch.Tensor or None): Timestamps (handled per step).
        """
        self.f = f
        self.reduced_DOF = reduced_DOF

        # Construct edge list from faces
        assert f.dim() == 3 and f.size(2) == 3, f"Expected f to be [batch_size, num_faces, 3], got {f.shape}"
        edge_list = torch.cat([
            f[0, :, [0, 1]],  # Forward edge: 0 -> 1
            f[0, :, [1, 2]],  # Forward edge: 1 -> 2
            f[0, :, [2, 0]],  # Forward edge: 2 -> 0
            f[0, :, [1, 0]],  # Reverse edge: 1 -> 0
            f[0, :, [2, 1]],  # Reverse edge: 2 -> 1
            f[0, :, [0, 2]]   # Reverse edge: 0 -> 2
        ], dim=0).t()  # Shape: [2, num_edges * 2]
        assert edge_list.dim() == 2 and edge_list.size(0) == 2, f"Expected edge_list to be [2, num_edges], got {edge_list.shape}"

        # Add self-loops to the edge list
        edge_list = add_self_loops(edge_list)[0]
        assert edge_list.dim() == 2 and edge_list.size(0) == 2, f"Expected edge_list after add_self_loops to be [2, num_edges + 1], got {edge_list.shape}"

        self.edge_list = edge_list

        # Set data in block1; 't' is handled per step
        self.block1.set_data(x, V, f=f, edge_list=edge_list)  # 't' is handled per step

    def forward(self, x, t, hidden_states=None):
        """
        Forward pass for TCSRVC.

        Args:
            x (torch.Tensor): Current vertex positions, shape [batch_size, num_vertices, 3].
            t (torch.Tensor or float): Current time step.
            hidden_states (torch.Tensor or None): Hidden states from previous steps.

        Returns:
            tuple: (deformation_vector, class_logits, updated_hidden_states)
        """
        assert x.dim() == 3 and x.size(2) == 3, f"Expected x to be [batch_size, num_vertices, 3], got {x.shape}"
        batch_size, num_vertices, _ = x.shape
        # Adjust this if you decide to handle batch_size > 1
        assert batch_size == 1, f"Expected batch_size=1 for temporal processing, got {batch_size}"
        
        # Assuming batch_size=1, extract vertex positions
        x = x.squeeze(0)  # Shape: [num_vertices, 3]
        assert x.dim() == 2 and x.size(1) == 3, f"Expected x after squeeze to be [num_vertices, 3], got {x.shape}"
        
        dx, class_logits, hidden_states = self.block1(x, t, hidden_states)
        dx = dx.unsqueeze(0)  # Shape: [1, num_vertices, 3]
        assert dx.dim() == 3 and dx.size(0) == 1 and dx.size(2) == 3, f"Expected dx to be [1, num_vertices, 3], got {dx.shape}"
        
        self.class_logits = class_logits
        assert class_logits.dim() == 2 and class_logits.size(1) == self.block1.num_classes, f"Expected class_logits to be [num_vertices, {self.block1.num_classes}], got {class_logits.shape}"

        # Return the deformation vector and class logits
        return dx, class_logits, hidden_states

    def get_class_logits(self):
        """
        Get the latest class logits.

        Returns:
            torch.Tensor: Class logits, shape [num_vertices, num_classes].
        """
        return self.class_logits
