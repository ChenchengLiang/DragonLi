import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, GINConv

class BaseWithNFFNN(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, n_ffnn, ffnn_hidden_size,gnn_dropout_rate=0.5,ffnn_dropout_rate=0.5):
        super(BaseWithNFFNN, self).__init__()

        self.embedding = nn.Embedding(input_feature_dim, gnn_hidden_dim)
        self.ffnn_layers = nn.ModuleList([nn.Linear(ffnn_hidden_size, ffnn_hidden_size) for _ in range(n_ffnn)])
        self.fc_final = nn.Linear(ffnn_hidden_size, 1)
        self.gnn_dropout_rate = gnn_dropout_rate
        self.ffnn_dropout_rate = ffnn_dropout_rate


    def forward(self, g, in_feat):
        h = self.embedding(in_feat.long())
        h = self.gnn_forward(g, h)
        g.ndata["h"] = h
        agg_h = dgl.mean_nodes(g, "h")
        prob = self.process_ffnn(agg_h)
        return prob

    def process_ffnn(self, agg_h):
        ffnn_out = agg_h
        for i, layer in enumerate(self.ffnn_layers):
            ffnn_out = F.relu(layer(ffnn_out))
            if i < len(self.ffnn_layers) - 1:  # apply dropout to all layers except the last one
                ffnn_out = F.dropout(ffnn_out, self.ffnn_dropout_rate, training=self.training)
        h_single = self.fc_final(ffnn_out)
        return torch.sigmoid(h_single)

    def gnn_forward(self, g, in_feat):
        raise NotImplementedError("This method should be implemented by derived classes.")

class GCNWithNFFNN(BaseWithNFFNN):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, ffnn_layer_num, ffnn_hidden_dim, gnn_dropout_rate=0.5,ffnn_dropout_rate=0.5):
        super(GCNWithNFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim, gnn_dropout_rate=gnn_dropout_rate,ffnn_dropout_rate=ffnn_dropout_rate)
        self.conv_layers = nn.ModuleList([GraphConv(gnn_hidden_dim, gnn_hidden_dim) for _ in range(gnn_layer_num)])

    def gnn_forward(self, g, h):
        for i, layer in enumerate(self.conv_layers):
            h = F.relu(layer(g, h))
            if i < len(self.conv_layers) - 1:  # Apply dropout to all layers except the last one
                h = F.dropout(h, self.gnn_dropout_rate, training=self.training)
        return h

class GATWithNFFNN(BaseWithNFFNN):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, num_heads, ffnn_layer_num, ffnn_hidden_dim,gnn_dropout_rate=0.5,ffnn_dropout_rate=0.5):
        super(GATWithNFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim,gnn_dropout_rate=gnn_dropout_rate,ffnn_dropout_rate=ffnn_dropout_rate)
        self.gat_layers = nn.ModuleList([GATConv(gnn_hidden_dim, gnn_hidden_dim, num_heads) for _ in range(gnn_layer_num - 1)])
        self.gat_final = GATConv(gnn_hidden_dim * num_heads, gnn_hidden_dim, 1)


    def gnn_forward(self, g, h):
        for i, layer in enumerate(self.gat_layers):
            h = F.elu(layer(g, h).view(h.size(0), -1))
            if i < len(self.gat_layers) - 1:  # apply dropout to all layers except the last one
                h = F.dropout(h, self.gnn_dropout_rate, training=self.training)
        return self.gat_final(g, h).squeeze(1)

class GINWithNFFNN(BaseWithNFFNN):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, ffnn_layer_num, ffnn_hidden_dim, gnn_dropout_rate=0.5,ffnn_dropout_rate=0.5):
        super(GINWithNFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim, gnn_dropout_rate=gnn_dropout_rate,ffnn_dropout_rate=ffnn_dropout_rate)
        mlp = nn.Sequential(nn.Linear(gnn_hidden_dim, gnn_hidden_dim), nn.ReLU(), nn.Linear(gnn_hidden_dim, gnn_hidden_dim))
        self.gin_layers = nn.ModuleList([GINConv(mlp, learn_eps=True) for _ in range(gnn_layer_num)])

    def gnn_forward(self, g, h):
        for i, layer in enumerate(self.gin_layers):
            h = layer(g, h)
            if i < len(self.gin_layers) - 1:  # Apply dropout to all layers except the last one
                h = F.dropout(h, self.gnn_dropout_rate, training=self.training)
        return h
