import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv


class BaseWithNFFNN(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, n_ffnn, ffnn_hidden_size):
        super(BaseWithNFFNN, self).__init__()

        # Embedding layer to transform integer features into embeddings
        self.embedding = nn.Embedding(input_feature_dim, gnn_hidden_dim)

        # FFNN layers after graph aggregation
        self.ffnn_layers = nn.ModuleList([nn.Linear(ffnn_hidden_size, ffnn_hidden_size) for _ in range(n_ffnn)])
        self.fc_final = nn.Linear(ffnn_hidden_size, 1)  # Output 1 value for binary classification

    def forward(self, g, in_feat):
        h = self.embed_and_process_gnn(g, in_feat)
        g.ndata["h"] = h

        agg_h = self.graph_aggregation(g)
        prob = self.process_ffnn(agg_h)

        return prob

    def embed_and_process_gnn(self, g, in_feat):
        h = self.embedding(in_feat.long())
        return self.gnn_forward(g, h)

    def graph_aggregation(self, g):
        return dgl.mean_nodes(g, "h")

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

        # GCN layers
        self.conv1 = GraphConv(gnn_hidden_dim, gnn_hidden_dim)
        self.layers = nn.ModuleList([GraphConv(gnn_hidden_dim, gnn_hidden_dim) for _ in range(gnn_layer_num - 1)])
        self.conv_final = GraphConv(gnn_hidden_dim, gnn_hidden_dim)

    def gnn_forward(self, g, h):
        h = self.conv1(g, h)
        h = F.relu(h)
        for layer in self.layers:
            h = layer(g, h)
            h = F.relu(h)
        return self.conv_final(g, h)


class GATWithNFFNN(BaseWithNFFNN):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, num_heads, ffnn_layer_num, ffnn_hidden_dim):
        super(GATWithNFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim)

        # GAT layers
        self.gat1 = GATConv(gnn_hidden_dim, gnn_hidden_dim, num_heads)
        self.layers = nn.ModuleList([GATConv(gnn_hidden_dim * num_heads, gnn_hidden_dim, num_heads)
                                     for _ in range(gnn_layer_num - 1)])
        self.gat_final = GATConv(gnn_hidden_dim * num_heads, gnn_hidden_dim, 1)  # Single head for the final GAT layer

    def gnn_forward(self, g, h):
        h = self.gat1(g, h)
        h = F.elu(h.flatten(1))
        for layer in self.layers:
            h = layer(g, h)
            h = F.elu(h.flatten(1))
        return self.gat_final(g, h).squeeze(1)
