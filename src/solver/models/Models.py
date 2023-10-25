import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, GINConv

class BaseWithNFFNN(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, n_ffnn, ffnn_hidden_size):
        super(BaseWithNFFNN, self).__init__()

        self.embedding = nn.Embedding(input_feature_dim, gnn_hidden_dim)
        self.ffnn_layers = nn.ModuleList([nn.Linear(ffnn_hidden_size, ffnn_hidden_size) for _ in range(n_ffnn)])
        self.fc_final = nn.Linear(ffnn_hidden_size, 1)

    def forward(self, g, in_feat):
        h = self.embedding(in_feat.long())
        h = self.gnn_forward(g, h)
        g.ndata["h"] = h
        agg_h = dgl.mean_nodes(g, "h")
        prob = self.process_ffnn(agg_h)
        return prob

    def process_ffnn(self, agg_h):
        ffnn_out = agg_h
        for layer in self.ffnn_layers:
            ffnn_out = F.relu(layer(ffnn_out))
        h_single = self.fc_final(ffnn_out)
        return torch.sigmoid(h_single)

    def gnn_forward(self, g, in_feat):
        raise NotImplementedError("This method should be implemented by derived classes.")

class GCNWithNFFNN(BaseWithNFFNN):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, ffnn_layer_num, ffnn_hidden_dim):
        super(GCNWithNFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim)
        self.conv_layers = nn.ModuleList([GraphConv(gnn_hidden_dim, gnn_hidden_dim) for _ in range(gnn_layer_num)])

    def gnn_forward(self, g, h):
        for layer in self.conv_layers:
            h = F.relu(layer(g, h))
        return h

class GATWithNFFNN(BaseWithNFFNN):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, num_heads, ffnn_layer_num, ffnn_hidden_dim):
        super(GATWithNFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim)
        self.gat_layers = nn.ModuleList([GATConv(gnn_hidden_dim, gnn_hidden_dim, num_heads) for _ in range(gnn_layer_num - 1)])
        self.gat_final = GATConv(gnn_hidden_dim * num_heads, gnn_hidden_dim, 1)

    def gnn_forward(self, g, h):
        for layer in self.gat_layers:
            h = F.elu(layer(g, h).flatten(1))
        return self.gat_final(g, h).squeeze(1)

class GINWithNFFNN(BaseWithNFFNN):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, ffnn_layer_num, ffnn_hidden_dim):
        super(GINWithNFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim)
        mlp = nn.Sequential(nn.Linear(gnn_hidden_dim, gnn_hidden_dim), nn.ReLU(), nn.Linear(gnn_hidden_dim, gnn_hidden_dim))
        self.gin_layers = nn.ModuleList([GINConv(mlp, learn_eps=True) for _ in range(gnn_layer_num)])

    def gnn_forward(self, g, h):
        for layer in self.gin_layers:
            h = layer(g, h)
        return h
