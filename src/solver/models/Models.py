import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv


class BaseWithNFFNN(nn.Module):
    def __init__(self, n_ffnn, ffnn_hidden_size):
        super(BaseWithNFFNN, self).__init__()

        # FFNN layers after graph aggregation
        self.ffnn_layers = nn.ModuleList([nn.Linear(ffnn_hidden_size, ffnn_hidden_size) for _ in range(n_ffnn)])
        self.fc_final = nn.Linear(ffnn_hidden_size, 1)  # Output 1 value for binary classification

    def forward(self, g, in_feat):
        h = self.gnn_forward(g, in_feat)
        g.ndata["h"] = h

        # Graph aggregation
        agg_h = dgl.mean_nodes(g, "h")

        # FFNN layers
        ffnn_out = agg_h
        for layer in self.ffnn_layers:
            ffnn_out = F.relu(layer(ffnn_out))
        h_single = self.fc_final(ffnn_out)

        # Convert h_single to a probability between 0 and 1 using sigmoid activation
        prob = torch.sigmoid(h_single)

        return prob

    def gnn_forward(self, g, in_feat):
        raise NotImplementedError("This method should be implemented by derived classes.")


class GCNWithNFFNN(BaseWithNFFNN):
    def __init__(self, in_feats, h_feats, n_layers, n_ffnn, ffnn_hidden_size):
        super(GCNWithNFFNN, self).__init__(n_ffnn, ffnn_hidden_size)

        # GCN layers
        self.conv1 = GraphConv(in_feats, h_feats)
        self.layers = nn.ModuleList([GraphConv(h_feats, h_feats) for _ in range(n_layers - 1)])
        self.conv_final = GraphConv(h_feats, h_feats)

    def gnn_forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        for layer in self.layers:
            h = layer(g, h)
            h = F.relu(h)
        h = self.conv_final(g, h)
        return h


class GATWithNFFNN(BaseWithNFFNN):
    def __init__(self, in_feats, h_feats, n_layers, num_heads, n_ffnn, ffnn_hidden_size):
        super(GATWithNFFNN, self).__init__(n_ffnn, ffnn_hidden_size)

        # GAT layers
        self.gat1 = GATConv(in_feats, h_feats, num_heads)
        self.layers = nn.ModuleList([GATConv(h_feats * num_heads, h_feats, num_heads) for _ in range(n_layers - 1)])
        self.gat_final = GATConv(h_feats * num_heads, h_feats, 1)  # Single head for the final GAT layer

    def gnn_forward(self, g, in_feat):
        h = self.gat1(g, in_feat)
        h = F.elu(h.flatten(1))
        for layer in self.layers:
            h = layer(g, h)
            h = F.elu(h.flatten(1))
        h = self.gat_final(g, h).squeeze(1)
        return h
