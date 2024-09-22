import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATConv

from pytorch3d.structures import Meshes

# Placeholder for compute_normal function. Replace with your actual implementation.
def compute_normal(v, f):
    """
    Compute vertex normals for a mesh.

    Args:
        v (torch.Tensor): Vertex positions, shape [batch_size, num_vertices, 3].
        f (torch.Tensor): Face indices, shape [batch_size, num_faces, 3].

    Returns:
        torch.Tensor: Vertex normals, shape [batch_size, num_vertices, 3].
    """
    mesh = Meshes(verts=v, faces=f)
    normals = mesh.verts_normals_packed()
    return normals.view(v.size(0), -1, 3)  # Shape: [batch_size, num_vertices, 3]

# Placeholder for DeformationGNN class. Replace with your actual implementation.
class DeformationGNN(nn.Module):
    def __init__(self, input_features, hidden_features, output_dim, num_layers, gat_heads, use_gcn):
        super(DeformationGNN, self).__init__()
        # Implement your GNN layers here
        # This is a dummy implementation; replace with actual GNN
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, output_dim)
        )
    
    def forward(self, x, edge_list):
        return self.layers(x)  # Replace with actual GNN forward pass

# Minimal GraphAttentionEmbedding class with TransformerConv
class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim):
        super().__init__()
        # TransformerConv is used to aggregate information from neighbors.
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, edge_dim=msg_dim)

    def forward(self, x, edge_index, msg):
        if msg is None:
            # If there is no message, initialize it with zeros.
            msg = torch.zeros(edge_index.size(1), x.size(1), device=x.device)
        
        # Edge attributes are formed from message embeddings. Time encoding is skipped here.
        edge_attr = msg
        return self.conv(x, edge_index, edge_attr)

class NodeFeatureNet(nn.Module):
    def __init__(self, C=128, K=5, n_scale=3, use_pytorch3d_normal=True):
        super(NodeFeatureNet, self).__init__()
        # MLP layers
        self.use_pytorch3d_normal = use_pytorch3d_normal
        self.fc1 = nn.Linear(6, C)
        self.fc2 = nn.Linear(2 * C, C * 4)
        self.fc3 = nn.Linear(C * 4, C * 2)
        
        # For local convolution operation
        self.localconv = nn.Conv3d(n_scale, C, (K, K, K))
        self.localfc = nn.Linear(C, C)
        
        Q = n_scale      # Number of scales
        
        self.n_scale = n_scale
        self.K = K        
        self.C = C
        self.Q = Q
        # For cube sampling
        self.initialized = False
        grid = np.linspace(-K // 2, K // 2, K)
        grid_3d = np.stack(np.meshgrid(grid, grid, grid), axis=0).transpose(2,1,3,0)
        self.x_shift = torch.Tensor(grid_3d.copy()).view(-1, 3)
        self.cubes = torch.zeros([1, self.Q, self.K, self.K, self.K])
    
    def forward(self, v):
        """
        Forward pass for NodeFeatureNet.

        Args:
            v (torch.Tensor): Vertex positions, shape [batch_size=1, num_vertices, 3].

        Returns:
            torch.Tensor: Node features, shape [batch_size=1, num_vertices, C*2].
        """
        z_local = self.cube_sampling(v)  # [m, Q, K, K, K]
        z_local = self.localconv(z_local)  # [m, C, 1, 1, 1]
        z_local = F.leaky_relu(z_local, 0.2)  # Activation
        z_local = z_local.view(-1, self.m, self.C)  # [1, m, C]
        z_local = self.localfc(z_local)  # [1, m, C]
        z_local = F.leaky_relu(z_local, 0.2)  # Activation
        
        # Point feature
        if not self.use_pytorch3d_normal:
            normal = compute_normal(v, self.f)  # Shape: [batch_size, num_vertices, 3]
        else:
            normal = compute_normal(v, self.f)  # Utilizing the compute_normal function
        
        x = torch.cat([v, normal], 2)  # Shape: [batch_size, num_vertices, 6]
        z_point = F.leaky_relu(self.fc1(x), 0.2)  # Shape: [batch_size, num_vertices, C]
        
        # Feature fusion
        z = torch.cat([z_point, z_local], 2)  # Shape: [batch_size, num_vertices, 2*C]
        z = F.leaky_relu(self.fc2(z), 0.2)    # Shape: [batch_size, num_vertices, C*4]
        z = F.leaky_relu(self.fc3(z), 0.2)    # Shape: [batch_size, num_vertices, C*2]
        
        return z    # Node features

    def _initialize(self, V):
        # Initialize coordinates shift and cubes
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
            edge_list (torch.Tensor): Edge list for graph, shape [2, num_edges]. (Not used here)
        """
        if not self.initialized:
            self._initialize(V)
        self.f = f
        self.edge_list = edge_list
        # Set the shape of the volume
        D1, D2, D3 = V.shape[2], V.shape[3], V.shape[4]
        D = max([D1, D2, D3])
        # Rescale for grid sampling
        self.rescale = torch.tensor([D3/D, D2/D, D1/D], device=V.device)
        self.D = D

        self.m = x.shape[1]    # Number of points
        self.neighbors = self.cubes.repeat(self.m, 1, 1, 1, 1)    # Repeat m cubes
        
        # Set multi-scale volume
        self.Vq = [V]
        for q in range(1, self.Q):
            # Iteratively downsampling
            self.Vq.append(F.avg_pool3d(self.Vq[-1], 2))

    def cube_sampling(self, x):
        """
        Sample cubes around each point in x.

        Args:
            x (torch.Tensor): Vertex positions, shape [batch_size=1, num_vertices, 3].

        Returns:
            torch.Tensor: Neighboring cubes, shape [num_vertices, Q, K, K, K].
        """
        with torch.no_grad():
            for q in range(self.Q):
                # Make sure the cubes have the same size
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2 ** q)
                xq = xq.contiguous().view(1, -1, 3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale  # Rescale the coordinates
                # Sample the q-th cube
                vq = F.grid_sample(self.Vq[q], xq, mode='bilinear', padding_mode='border', align_corners=True)
                # Update the cubes
                self.neighbors[:, q] = vq[0, 0].view(self.m, self.K, self.K, self.K)
        
        return self.neighbors.clone()

class DeformBlockGNN(nn.Module):
    def __init__(self, C=128, K=5, n_scale=3, sf=.1, gnn_layers=2, use_gcn=True, gat_heads=8, use_pytorch3d_normal=True):
        super(DeformBlockGNN, self).__init__()
        self.C = C
        self.sf = sf
        self.nodeFeatureNet = NodeFeatureNet(C=C, K=K, n_scale=n_scale, use_pytorch3d_normal=use_pytorch3d_normal)
        
        # GNN block to aggregate information from neighbors
        self.gnn = GraphAttentionEmbedding(
            in_channels=C * 2,  # Assuming NodeFeatureNet outputs C*2 features
            out_channels=C,
            msg_dim=C  # Messages are C-dimensional
        )
        # Fully connected layer to compute the deformation vectors (3D)
        self.fc_deform = nn.Linear(C, 3)
        
        # Initialize TGNMemory in set_data
        self.memory = None  # Initialize memory as None

    def set_data(self, x, V, f=None, edge_list=None):
        """
        Set data for DeformBlockGNN.

        Args:
            x (torch.Tensor): Vertex positions, shape [num_vertices, 3].
            V (torch.Tensor): Volumetric data, shape [batch_size=1, channels, D1, D2, D3].
            f (torch.Tensor): Face indices, shape [batch_size=1, num_faces, 3].
            edge_list (torch.Tensor): Edge list for graph, shape [2, num_edges].
        """
        self.nodeFeatureNet.set_data(x, V, f=f, edge_list=edge_list)
        self.f = f
        self.V = V
        self.edge_list = edge_list

        num_nodes = x.size(0)  # Number of nodes
        if self.memory is None or self.memory.num_nodes != num_nodes:
            # Initialize TGNMemory
            
            self.memory = TGNMemory(
                num_nodes=num_nodes,     # Number of nodes in the current mesh
                raw_msg_dim=self.C,      # Dimension of raw message passed between nodes
                memory_dim=self.C,       # Memory dimension, matches node feature dimension
                time_dim=1,         # Unused here since time encoding is skipped
                message_module=IdentityMessage(self.C, self.C, 1),  # Identity message module__init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
                aggregator_module=LastAggregator()  # Aggregator that updates based on last interaction
            )#tested this to some extent externally. 
        
        self.memory.reset_state()  # Reset memory at the start of the batch
        self.t = torch.zeros(1, device=x.device).long()  # Initialize current time step

    def forward(self, memory):
        """
        Forward pass for DeformBlockGNN.

        Args:
            memory (torch.Tensor): Current memory state, shape [num_nodes, memory_dim].

        Returns:
            tuple: (deformation vectors, updated memory)
        """
        n_id = torch.arange(self.nodeFeatureNet.m, device=memory.device)  # Assuming batch_size=1

        # Retrieve the memory and last update information for the nodes
        current_memory, last_update = self.memory(n_id)  # [num_nodes, memory_dim], [num_nodes]

        print('edge_list.shape before max',self.edge_list.shape)
        print('edge_list before max',self.edge_list)
        # Ensure edge indices are valid and check bounds
        max_node_idx = self.nodeFeatureNet.m - 1  # Maximum valid node index
        assert torch.max(self.edge_list) <= max_node_idx, f"Edge index out of bounds! Max allowed index: {max_node_idx}"

        # Last update time for each node involved in the edges
        src_last_update = last_update[self.edge_list[0]]  # [num_edges]

        # Compute the relative time between now and last update for each edge
        rel_t = self.t - src_last_update.long()  # Time differences

        # Update the current time step
        self.t += 1

        # Use GNN to aggregate node features and messages (ignoring time encoding here)
        z = self.gnn(current_memory, self.edge_list, None)  # [num_nodes, out_channels]

        # Identify valid edges within the node index range
        valid_indices = (self.edge_list[0] < self.nodeFeatureNet.m) & (self.edge_list[1] < self.nodeFeatureNet.m)
        valid_edge_index = self.edge_list[:, valid_indices]
        valid_t = rel_t[valid_indices]

        # Extract the source node features for valid edges
        src_valid = valid_edge_index[0]  # Get source nodes from valid edges
        raw_msg = z[src_valid]           # [num_valid_edges, C]

        # Update memory only if there are valid edges
        print('valid_edge_index[0].shape',valid_edge_index[0].shape)
        print('valid_t.shape',valid_t.shape)
        print('raw_msg.shape',raw_msg.shape)
        if valid_edge_index.size(1) > 0:
            # Update the memory using the source and destination nodes (src = dst)
            self.memory.update_state(
                valid_edge_index[0],  # src
                valid_edge_index[0],  # dst (same as src)
                valid_t,              # t
                raw_msg               # msg
            )

        # Predict deformations for all nodes in the graph
        deformation = self.fc_deform(z)  # [num_nodes, 3]
        deformation = deformation * self.sf  # Apply scaling factor

        return deformation, self.memory.memory  # Return deformation and updated memory

class TCSRVC(nn.Module):
    """
    The deformation network of CortexODE model.
    """
    
    def __init__(self, dim_h=128,
                       kernel_size=5,
                       n_scale=3,
                       sf=.1,
                       gnn_layers=5,
                       use_gcn=True,
                       gat_heads=8,
                       use_pytorch3d_normal=True
                       ):
        
        super(TCSRVC, self).__init__()

        C = dim_h        # hidden dimension
        K = kernel_size  # kernel size
        
        self.block1 = DeformBlockGNN(C=C, 
                                     K=K, 
                                     n_scale=n_scale,
                                     sf=sf,
                                     gnn_layers=gnn_layers,
                                     use_gcn=use_gcn,
                                     gat_heads=gat_heads,
                                     use_pytorch3d_normal=use_pytorch3d_normal
                                     )
        
    def set_data(self, x, V, f=None, reduced_DOF=False):
        """
        Set data for TCSRVC.

        Args:
            x (torch.Tensor): Vertex positions, shape [batch_size=1, num_vertices, 3].
            V (torch.Tensor): Volumetric data, shape [batch_size=1, channels=3, D1, D2, D3].
            f (torch.Tensor): Face indices, shape [batch_size=1, num_faces, 3].
            reduced_DOF (bool): Whether to use reduced degrees of freedom.
        """
        # Validations
        assert x.shape[0] == 1, "Batch size should be 1"
        assert f.shape[0] == 1, "Batch size for faces should be 1"
        assert x.shape[1] != 1, "Number of points should be greater than 1"
        assert f.shape[1] != 1, "Number of faces should be greater than 1"
        
        self.f = f
        self.reduced_DOF = reduced_DOF
        
        # Generate edge_list by extracting edges from faces
        edge_list = torch.cat([
            f[0, :, [0, 1]],
            f[0, :, [1, 2]],
            f[0, :, [2, 0]]
        ], dim=0).transpose(0, 1)  # Shape: [2, num_edges]
        
        # Add self-loops
        edge_list = add_self_loops(edge_list)[0]  # Shape: [2, num_edges + num_nodes]
        
        self.edge_list = edge_list
        
        # Set data in DeformBlockGNN
        self.block1.set_data(x.squeeze(0), V, f=f, edge_list=edge_list)
        
    def forward(self, x, memory):
        """
        Forward pass for TCSRVC.

        Args:
            t (torch.Tensor or int): Current discrete time step.
            x (torch.Tensor): Current vertex positions, shape [batch_size=1, num_vertices, 3].
            memory (torch.Tensor): Current memory state, shape [num_nodes, memory_dim].

        Returns:
            tuple: (deformation_vector, updated_memory)
        """
        dx, updated_memory = self.block1(memory)  # [num_nodes, 3], [num_nodes, memory_dim]
        print('dx.shape','updated_memory.shape',dx.shape,updated_memory.shape)
        dx = dx.unsqueeze(0)  # [1, num_nodes, 3]
        return dx, updated_memory  # [1, num_nodes, 3], [num_nodes, memory_dim]

# Helper function to generate triangular mesh edge_index from faces and add self-loops
def generate_mesh_edge_index(faces, num_nodes):
    """
    Generate the edge_index tensor from triangular mesh faces and add self-loops.
    This captures the connectivity between vertices based on the faces.

    Args:
        faces (torch.Tensor): Face indices, shape [num_faces, 3].
        num_nodes (int): Number of nodes in the mesh.

    Returns:
        torch.Tensor: edge_index tensor, shape [2, num_edges].
    """
    # Extract edges from faces (each face has three edges)
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)  # Shape: [num_faces * 3, 2]

    # Create the `edge_index` by flattening the edges in both directions for undirected graph
    edge_index = torch.cat([edges, edges.flip(1)], dim=0).t()  # [2, num_faces * 6]

    # Add self-loops to the edge index
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    return edge_index

# Function to randomly choose between a 4-node or 5-node mesh
def get_random_mesh():
    if torch.rand(1).item() > 0.5:
        # Case 1: 4 nodes (original)
        faces = torch.tensor([
            [0, 1, 2],  # Face 1 (triangle)
            [1, 2, 3]   # Face 2 (triangle)
        ])
        x = torch.rand((1, 4, 3))  # [batch_size=1, num_vertices=4, 3]
        num_nodes = 4
    else:
        # Case 2: 5 nodes (adding one more face for 5 nodes)
        faces = torch.tensor([
            [0, 1, 2],  # Face 1 (triangle)
            [1, 2, 3],  # Face 2 (triangle)
            [2, 3, 4]   # Face 3 (triangle with node 4)
        ])
        x = torch.rand((1, 5, 3))  # [batch_size=1, num_vertices=5, 3]
        num_nodes = 5

    edge_index = generate_mesh_edge_index(faces, num_nodes)  # Generate mesh edges with self-loops
    return x, edge_index, faces

# Training loop for the model with TGN
def train(num_batches=5, num_timesteps=3):
    device = torch.device('cpu')

    # Initialize the TCSRVC model
    model = TCSRVC(
        dim_h=128,
        kernel_size=5,
        n_scale=3,
        sf=0.1,
        gnn_layers=5,
        use_gcn=True,
        gat_heads=8,
        use_pytorch3d_normal=True
    ).to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection for debugging

    # Initialize memory as None
    memory = None

    # Training loop
    for batch_idx in range(num_batches):
        # Get random mesh (either 4-node or 5-node)
        x, edge_index, faces = get_random_mesh()
        x, edge_index = x.to(device), edge_index.to(device)
        
        print('x.shape','edge_index.shape')
        print(x.shape,edge_index.shape)
        # Clone `x` to create `y_hat`, which will store updated coordinates after each timestep
        y_hat = x.clone()

        # Compute the initial loss (MSE between `y_hat` and ones)
        target = torch.ones_like(y_hat)
        initial_loss = F.mse_loss(y_hat, target)
        print(f'Initial Loss: {initial_loss.item()}')

        # Set data in the model
        V = torch.rand((1, 3, 10, 10, 10)).to(device)  # Example volumetric data
        model.set_data(x, V, f=faces.unsqueeze(0), reduced_DOF=False)

        # Reset optimizer gradients
        optimizer.zero_grad()
        # Initialize memory for the batch
        if memory is None or memory.size(0) != x.size(1):
            # Initialize memory by querying model.block1.memory
            num_nodes = x.size(1)
            print('num_nodes', num_nodes)
            print(torch.arange(num_nodes, device=device).shape)
            memory,_ = model.block1.memory(torch.arange(num_nodes, device=device))  # [num_nodes, memory_dim]
            print('after num_nodes')
        # Loop over the timesteps within each batch (memory carries over across timesteps)
        for timestep in range(num_timesteps):
            print('timestep',timestep)
            # Forward pass
            dx, updated_memory = model(x, memory)  # [1, num_vertices, 3], [num_nodes, memory_dim]
            # Update vertex positions
            x = x + dx
            y_hat = y_hat + dx  # Update y_hat accordingly

            # Update memory
            memory = updated_memory

        # Compute the final loss (MSE loss on `y_hat` compared to a tensor of ones)
        final_loss = F.mse_loss(y_hat, target)
        final_loss.backward()
        optimizer.step()

        # Detach memory after the batch is completed to avoid autograd tracking across batches
        memory = memory.detach()

        print(f'Batch {batch_idx}, Final Loss: {final_loss.item()}')

# Run the training loop with multiple timesteps per batch and variable mesh sizes
if __name__ == "__main__":
    train(num_batches=100, num_timesteps=3)
