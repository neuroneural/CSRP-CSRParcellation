import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TopKPooling, global_mean_pool
from torch_geometric.nn import LayerNorm as PyGLayerNorm


class DeformationGNN(nn.Module):
    def __init__(self, input_features=128, hidden_features=128, output_dim=3, num_layers=2, gat_heads=8, use_gcn=True, final_activation='None'):
        super(DeformationGNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.final_activation = final_activation
        current_dim = input_features

        for i in range(num_layers-1):
            if use_gcn:
                # Intermediate or first GCN layer
                self.layers.append(GCNConv(current_dim, hidden_features))
                next_dim = hidden_features
            else:
                # Intermediate or first GAT layer
                self.layers.append(GATConv(current_dim, hidden_features, heads=gat_heads))
                next_dim = hidden_features * gat_heads

            current_dim = next_dim
        
        if use_gcn:
            self.layers.append(GCNConv(hidden_features, output_dim))
        else:
            self.layers.append(GATConv(current_dim, output_dim, heads=1))
        
        print('final activation', self.final_activation)
    def forward(self, x, edge_index):
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) -1:#it is a bug to apply relu to last layer only allows positive delta. 
                x = F.leaky_relu(x, 0.2)
        if self.final_activation == 'log_softmax':
            x = F.log_softmax(x, dim=1)
        elif self.final_activation == 'tanh':
            x = torch.tanh(x)
        
        return x
