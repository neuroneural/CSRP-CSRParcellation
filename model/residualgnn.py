import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TopKPooling, LayerNorm, global_mean_pool

class ResidualGNN(torch.nn.Module):
    def __init__(self, num_node_features=128, num_classes=3, num_layers=2, gat_heads=8, dropout=0.1, pool_ratio=1.0, use_pooling=True, use_residual=True, use_layernorm=True, use_gcn=True, final_activation='tanh'):
        super(ResidualGNN, self).__init__()
        
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList() if use_layernorm else None
        self.pools = torch.nn.ModuleList() if use_pooling else None
        self.dropouts = torch.nn.ModuleList() if dropout is not None else None
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.final_activation = final_activation
        hidden_channels = num_node_features

        for i in range(num_layers):
            out_channels = hidden_channels if i < num_layers - 1 else num_classes
            if use_gcn:
                self.layers.append(GCNConv(hidden_channels, out_channels))
            else:
                self.layers.append(GATConv(hidden_channels, out_channels, heads=gat_heads))
            if use_layernorm:
                self.norms.append(LayerNorm(normalized_shape=out_channels))
            if use_pooling:
                self.pools.append(TopKPooling(out_channels, ratio=pool_ratio))
            if dropout is not None:
                self.dropouts.append(torch.nn.Dropout(dropout))
            hidden_channels = out_channels

        fc_hidden = 2 * hidden_channels

        self.fc_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_channels, fc_hidden),
            torch.nn.Linear(fc_hidden, fc_hidden),
            torch.nn.Linear(fc_hidden, num_classes)
        ])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_res = x

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.norms is not None:
                x = self.norms[i](x)
            x = F.relu(x)
            if self.dropouts is not None:
                x = self.dropouts[i](x)
            if self.use_residual and i > 0:
                x += x_res
            x_res = x
            if self.pools is not None and i < len(self.pools):
                x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)

        x = global_mean_pool(x, batch)

        for fc in self.fc_layers[:-1]:
            x = F.relu(fc(x))
            if self.dropouts is not None:
                x = self.dropouts[-1](x)
        x = self.fc_layers[-1](x)

        if self.final_activation == 'log_softmax':
            return F.log_softmax(x, dim=1)
        elif self.final_activation == 'tanh':
            return torch.tanh(x)
        else:
            raise ValueError("Unsupported final activation function. Choose 'log_softmax' or 'tanh'.")