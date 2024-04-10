import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TopKPooling, global_mean_pool
from torch_geometric.nn import LayerNorm as PyGLayerNorm


class ResidualGNN(nn.Module):
    def __init__(self, input_features=128, hidden_features=128, num_classes=3, num_layers=2, gat_heads=8, dropout=0.1, pool_ratio=.9, use_pooling=False, use_residual=False, use_layernorm=False, use_gcn=True, final_activation='tanh'):
        super(ResidualGNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layernorm else None
        self.pools = nn.ModuleList() if use_pooling else None
        self.dropouts = nn.ModuleList() if dropout is not None else None
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.final_activation = final_activation
        current_dim = input_features

        for i in range(num_layers):
            if use_gcn:
                # Intermediate or first GCN layer
                self.layers.append(GCNConv(current_dim, hidden_features))
                next_dim = hidden_features
            else:
                # Intermediate or first GAT layer
                self.layers.append(GATConv(current_dim, hidden_features, heads=gat_heads))
                next_dim = hidden_features * gat_heads

            current_dim = next_dim
            
            if use_layernorm:
                # When adding to your model
                self.norms.append(PyGLayerNorm(next_dim, mode='graph'))
            if use_pooling and i < num_layers - 1:  # No pooling after the last convolutional layer
                self.pools.append(TopKPooling(next_dim, ratio=pool_ratio))
            if dropout is not None:
                self.dropouts.append(nn.Dropout(dropout))

        # Fully connected layers to map the final convolutional output to class predictions
        self.fc_layers = nn.ModuleList([#removed a dropout from this list, but still apply to fcn
            nn.Linear(current_dim, num_classes)
        ])

    def forward(self, x, edge_index):
        #x_res = x

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.norms and i < len(self.norms):
                x = self.norms[i](x)
            x = F.relu(x)
            if self.dropouts and i < len(self.dropouts):
                x = self.dropouts[i](x)
            if self.use_residual and i > 0:
                # x += x_res
                assert False, "unsupported for now"
            #x_res = x
            if self.pools and i < len(self.pools):
                x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None)

        #removed global pooling because of node classification. 
        
        # Process through fully connected layers
        x = self.fc_layers[-1](x)

        if self.final_activation == 'log_softmax':
            x = F.log_softmax(x, dim=1)
        elif self.final_activation == 'tanh':
            x = torch.tanh(x)
        
        return x
