import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator
from torch_geometric.utils import add_self_loops  # Importing add_self_loops

# Minimal GraphAttentionEmbedding class with TransformerConv
class GraphAttentionEmbedding(torch.nn.Module):
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

# Minimal DeformBlockGNN that includes TGNMemory
class DeformBlockGNN(torch.nn.Module):
    def __init__(self, C=128):
        super().__init__()
        self.C = C

        # These variables are initialized here, but the memory will be dynamically resized.
        self.memory = None  # Placeholder for dynamic memory

        # GNN block to aggregate information from neighbors
        self.gnn = GraphAttentionEmbedding(
            in_channels=C,
            out_channels=C,
            msg_dim=C  # Messages are C-dimensional
        )
        # Fully connected layer to compute the deformation vectors (3D)
        self.fc_deform = torch.nn.Linear(C, 3)

    def set_data(self, x, edge_index, t):
        """
        Set the mesh structure and node features for the current batch.
        This also resets the memory at the start of a new batch.
        """
        num_nodes = x.size(0)  # Dynamically determine the number of nodes in the current batch
        if self.memory is None or self.memory.num_nodes != num_nodes:
            # Dynamically initialize the memory with the correct number of nodes
            self.memory = TGNMemory(
                num_nodes=num_nodes,     # Number of nodes in the current mesh
                raw_msg_dim=self.C,      # Dimension of raw message passed between nodes
                memory_dim=self.C,       # Memory dimension, matches node feature dimension
                time_dim=self.C,         # Unused here since time encoding is skipped
                message_module=IdentityMessage(self.C, self.C, self.C),  # Identity message module
                aggregator_module=LastAggregator()  # Aggregator that updates based on last interaction
            )

        self.edge_index = edge_index  # Mesh edges
        self.x = x                    # Node features (now 3D coordinates)
        self.t = t                    # Time information (although not heavily used here)
        self.memory.reset_state()     # Reset memory at the start of each batch

    def forward(self, memory):
        n_id = torch.arange(self.x.size(0), device=self.x.device)

        # Retrieve the memory and last update information for the nodes
        memory, last_update = self.memory(n_id)

        # Ensure edge indices are valid and check bounds
        max_node_idx = self.x.size(0) - 1  # Maximum valid node index
        assert torch.max(self.edge_index) <= max_node_idx, f"Edge index out of bounds! Max allowed index: {max_node_idx}"

        # Last update time for each node involved in the edges
        src_last_update = last_update[self.edge_index[0]]

        # Compute the relative time between now and last update for each edge
        rel_t = self.t - src_last_update.float()  # Time differences

        # Use GNN to aggregate node features and messages (ignoring time encoding here)
        z = self.gnn(memory, self.edge_index, None)

        # Identify valid edges within the node index range
        valid_indices = (self.edge_index[0] < self.x.size(0)) & (self.edge_index[1] < self.x.size(0))
        valid_edge_index = self.edge_index[:, valid_indices]
        valid_t = self.t[valid_indices]

        # Extract the source node features for valid edges
        src_valid = valid_edge_index[0]  # Get source nodes from valid edges
        raw_msg = z[src_valid]           # Node features serve as raw messages

        # Update memory only if there are valid edges
        if valid_edge_index.size(1) > 0:
            # Update the memory using the source and destination nodes (src = dst)
            self.memory.update_state(
                valid_edge_index[0],  # src
                valid_edge_index[0],  # dst (same as src)
                valid_t,              # t
                raw_msg               # msg
            )

        # Predict deformations for all nodes in the graph
        deformation = self.fc_deform(z)  # Output deformation vectors for each node
        return deformation, memory          # Return updated memory

# Helper function to generate triangular mesh edge_index from faces and add self-loops
def generate_mesh_edge_index(faces, num_nodes):
    """
    Generate the edge_index tensor from triangular mesh faces and add self-loops.
    This captures the connectivity between vertices based on the faces.
    """
    # Extract edges from faces (each face has three edges)
    edges = torch.cat([
        faces[:, [0, 1]],  # Edge between vertices 0 and 1
        faces[:, [1, 2]],  # Edge between vertices 1 and 2
        faces[:, [2, 0]]   # Edge between vertices 2 and 0
    ], dim=0)

    # Create the `edge_index` by flattening the edges in both directions for undirected graph
    edge_index = torch.cat([edges, edges.flip(1)], dim=0).t()  # Transpose to match expected shape

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
        x = torch.rand((4, 3))  # 4 nodes with 3D coordinates
        num_nodes = 4
    else:
        # Case 2: 5 nodes (adding one more face for 5 nodes)
        faces = torch.tensor([
            [0, 1, 2],  # Face 1 (triangle)
            [1, 2, 3],  # Face 2 (triangle)
            [2, 3, 4]   # Face 3 (triangle with node 4)
        ])
        x = torch.rand((5, 3))  # 5 nodes with 3D coordinates
        num_nodes = 5

    edge_index = generate_mesh_edge_index(faces, num_nodes)  # Generate mesh edges with self-loops
    t = torch.randint(0, 100, (edge_index.size(1),))  # Random integer timestamps
    return x, edge_index, t

# Training loop for the minimalist model with dynamic mesh sizes
def train(num_batches=5, num_timesteps=3):
    device = torch.device('cpu')

    # Model initialization
    model = DeformBlockGNN(C=128).to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection for debugging

    # Training loop
    for batch_idx in range(num_batches):
        # Get random mesh (either 4-node or 5-node)
        x, edge_index, t = get_random_mesh()
        x, edge_index, t = x.to(device), edge_index.to(device), t.to(device)
        
        # Clone `x` to create `y_hat`, which will store updated coordinates after each timestep
        y_hat = x.clone()

        # Compute the initial loss (MSE between `y_hat` and ones)
        target = torch.ones_like(y_hat)
        initial_loss = F.mse_loss(y_hat, target)
        print(f'Initial Loss: {initial_loss.item()}')

        # Reset memory at the start of each batch
        model.set_data(x, edge_index, t)
        optimizer.zero_grad()

        # Initialize the memory for the batch
        batch_memory, _last_update = model.memory(torch.arange(x.size(0), device=device))  # Initialize memory for nodes

        # Loop over the timesteps within each batch (memory carries over across timesteps)
        for timestep in range(num_timesteps):
            # Get the deformation and memory
            deformation, batch_memory = model(batch_memory)  # Memory is passed and updated at each timestep
            
            # Update `y_hat` by adding the deformation to the vertex coordinates
            y_hat = y_hat + deformation
            

        # Compute the final loss (MSE loss on `y_hat` compared to a tensor of ones)
        final_loss = F.mse_loss(y_hat, target)
        final_loss.backward()
        optimizer.step()

        # Detach memory after the batch is completed to avoid autograd tracking across batches
        model.memory.detach()

        print(f'Batch {batch_idx}, Final Loss: {final_loss.item()}')

# Run the training loop with multiple timesteps per batch and variable mesh sizes
train(num_batches=100, num_timesteps=3)
