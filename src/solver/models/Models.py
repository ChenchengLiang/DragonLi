import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, GINConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from dgl.nn.pytorch import SumPooling



class GraphClassifier(nn.Module):
    def __init__(self, shared_gnn, classifier):
        super(GraphClassifier, self).__init__()
        self.shared_gnn = shared_gnn
        self.classifier = classifier

    def forward(self, graphs):
        gnn_output = self.shared_gnn(graphs)
        return self.classifier(gnn_output)


class Classifier(nn.Module):
    def __init__(self, ffnn_hidden_dim, ffnn_layer_num, output_dim, ffnn_dropout_rate=0.5):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(ffnn_hidden_dim*output_dim, ffnn_hidden_dim))
        for _ in range(ffnn_layer_num):
            self.layers.append(nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim))

        self.final_fc = nn.Linear(ffnn_hidden_dim, output_dim)
        self.dropout = nn.Dropout(ffnn_dropout_rate)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(self.dropout(x))
            else:
                x = F.relu(x)

        return F.softmax(self.final_fc(x), dim=1)



class SharedGNN(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, gnn_dropout_rate=0.5):
        super(SharedGNN, self).__init__()
        self.embedding = GraphEmbedding(num_node_types=input_feature_dim, hidden_feats=gnn_hidden_dim,
                                        num_gnn_layers=gnn_layer_num, dropout_rate=gnn_dropout_rate)

    def forward(self, graphs):
        embeddings = [self.embedding(g) for g in graphs]
        concatenated_embedding = torch.cat(embeddings, dim=0)
        return concatenated_embedding


class GraphEmbedding(nn.Module):
    def __init__(self, num_node_types, hidden_feats, num_gnn_layers, dropout_rate):
        super(GraphEmbedding, self).__init__()
        self.node_embedding = nn.Embedding(num_node_types, hidden_feats)
        self.gnn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        # First GNN layer
        self.gnn_layers.append(GraphConv(hidden_feats, hidden_feats))
        # Additional GNN layers
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GraphConv(hidden_feats, hidden_feats))

    def forward(self, g):
        # Embedding each node
        h = self.node_embedding(g.ndata['feat'])

        # Passing through GNN layers
        for i, layer in enumerate(self.gnn_layers):
            h = layer(g, h)
            if i < len(self.gnn_layers) - 1:  # Apply dropout after each layer except the last
                h = F.relu(self.dropout(h))
            else:
                h = F.relu(h)  # No dropout after the last layer

        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # Aggregating node features
        return hg
#
# class GraphClassifier(nn.Module):
#     def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num,ffnn_layer_num ,ffnn_hidden_dim,gnn_dropout_rate=0.5,ffnn_dropout_rate=0.5):
#         super(GraphClassifier, self).__init__()
#         self.embedding = GraphEmbedding(num_node_types=input_feature_dim, hidden_feats=gnn_hidden_dim,num_gnn_layers=gnn_layer_num,dropout_rate=gnn_dropout_rate)
#
#         # Linear layers for processing 2 graphs
#         self.linear_layers_2 = nn.ModuleList()
#         for _ in range(ffnn_layer_num):
#             self.linear_layers_2.append(nn.Linear(ffnn_hidden_dim * 2, ffnn_hidden_dim))
#
#         # Linear layers for processing 3 graphs
#         self.linear_layers_3 = nn.ModuleList()
#         for _ in range(ffnn_layer_num):
#             self.linear_layers_3.append(nn.Linear(ffnn_hidden_dim * 3, ffnn_hidden_dim))
#
#         # Final layers for 2 or 3 graphs
#         self.final_fc_2 = nn.Linear(ffnn_hidden_dim, 2)  # for 2 graphs
#         self.final_fc_3 = nn.Linear(ffnn_hidden_dim, 3)  # for 3 graphs
#         self.dropout = nn.Dropout(ffnn_dropout_rate)
#
#     def forward(self, graphs):
#         embeddings = [self.embedding(g) for g in graphs]
#         concatenated_embedding = torch.cat(embeddings, dim=0)
#
#         if len(graphs) == 2:
#             for i, linear in enumerate(self.linear_layers_2):
#                 concatenated_embedding = linear(concatenated_embedding)
#                 if i < len(self.linear_layers_2) - 1:
#                     concatenated_embedding = F.relu(self.dropout(concatenated_embedding))
#                 else:
#                     concatenated_embedding = F.relu(concatenated_embedding)
#             out = self.final_fc_2(concatenated_embedding)
#         elif len(graphs) == 3:
#             for i, linear in enumerate(self.linear_layers_3):
#                 concatenated_embedding = linear(concatenated_embedding)
#                 if i < len(self.linear_layers_3) - 1:
#                     concatenated_embedding = F.relu(self.dropout(concatenated_embedding))
#                 else:
#                     concatenated_embedding = F.relu(concatenated_embedding)
#             out = self.final_fc_3(concatenated_embedding)
#         else:
#             raise ValueError("Unsupported number of graphs")
#
#         return F.softmax(out,dim=1)
#





class BaseWithNFFNN(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, n_ffnn, ffnn_hidden_size,gnn_dropout_rate=0.5,ffnn_dropout_rate=0.5):
        super(BaseWithNFFNN, self).__init__()

        self.embedding = nn.Embedding(input_feature_dim, gnn_hidden_dim)
        self.ffnn_layers = nn.ModuleList([nn.Linear(ffnn_hidden_size, ffnn_hidden_size) for _ in range(n_ffnn)])
        self.fc_final = nn.Linear(ffnn_hidden_size, 1)
        self.gnn_dropout_rate = gnn_dropout_rate
        self.ffnn_dropout_rate = ffnn_dropout_rate


    def forward(self, g):
        h = self.embedding(g.ndata['feat'])
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

    def gnn_forward(self, g):
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


class GatingNetwork(nn.Module):
    def __init__(self, gnn_hidden_dim):
        super(GatingNetwork, self).__init__()
        self.gate_nn = nn.Linear(gnn_hidden_dim, 1)

    def forward(self, h):
        return self.gate_nn(h)

class GCNWithGAPFFNN(BaseWithNFFNN):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, ffnn_layer_num, ffnn_hidden_dim, gnn_dropout_rate=0.5, ffnn_dropout_rate=0.5):
        super(GCNWithGAPFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim, gnn_dropout_rate, ffnn_dropout_rate)
        self.conv_layers = nn.ModuleList([GraphConv(gnn_hidden_dim, gnn_hidden_dim) for _ in range(gnn_layer_num)])
        self.gating_network = GatingNetwork(gnn_hidden_dim)
        self.global_attention_pooling = GlobalAttentionPooling(self.gating_network)

    def gnn_forward(self, g, h):
        # Apply GNN layers
        for i, layer in enumerate(self.conv_layers):
            h = F.relu(layer(g, h))
            if i < len(self.conv_layers) - 1:
                h = F.dropout(h, self.gnn_dropout_rate, training=self.training)
        return h

    def forward(self, g):
        h = self.embedding(g.ndata['feat'])
        h = self.gnn_forward(g, h)
        g.ndata['h'] = h
        # Apply Global Attention Pooling
        hg = self.global_attention_pooling(g, h)
        prob = self.process_ffnn(hg)
        return prob



class MultiGNNs(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, ffnn_layer_num, ffnn_hidden_dim,
                 gnn_dropout_rate=0.5, ffnn_dropout_rate=0.5):
        super(MultiGNNs, self).__init__()

        # Separate embeddings for each GNN
        self.gcn_embedding = nn.Embedding(input_feature_dim, gnn_hidden_dim)
        self.gin_embedding = nn.Embedding(input_feature_dim, gnn_hidden_dim)

        # GCN and GIN layers
        self.gcn_layers = nn.ModuleList([GraphConv(gnn_hidden_dim, gnn_hidden_dim) for _ in range(gnn_layer_num)])
        mlp = nn.Sequential(nn.Linear(gnn_hidden_dim, gnn_hidden_dim), nn.ReLU(),
                            nn.Linear(gnn_hidden_dim, gnn_hidden_dim))
        self.gin_layers = nn.ModuleList([GINConv(mlp, learn_eps=True) for _ in range(gnn_layer_num)])

        # FFNN layers
        self.ffnn_layers = nn.ModuleList(
            [nn.Linear(gnn_hidden_dim * 2, ffnn_hidden_dim) if i == 0 else nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim)
             for i in range(ffnn_layer_num)])

        self.fc_final = nn.Linear(ffnn_hidden_dim, 1)

        self.gnn_dropout_rate = gnn_dropout_rate
        self.ffnn_dropout_rate = ffnn_dropout_rate

    def forward(self, g):
        # GCN forward pass
        gcn_h = self.gcn_embedding(g.ndata['feat'])
        for i, layer in enumerate(self.gcn_layers):
            gcn_h = F.relu(layer(g, gcn_h))
            if i < len(self.gcn_layers) - 1:
                gcn_h = F.dropout(gcn_h, self.gnn_dropout_rate, training=self.training)

        # GIN forward pass
        gin_h = self.gin_embedding(g.ndata['feat'])
        for i, layer in enumerate(self.gin_layers):
            gin_h = layer(g, gin_h)
            if i < len(self.gin_layers) - 1:
                gin_h = F.dropout(gin_h, self.gnn_dropout_rate, training=self.training)

        # Aggregate node features to form graph representations
        g.ndata["gcn_h"] = gcn_h
        g.ndata["gin_h"] = gin_h
        gcn_agg = dgl.mean_nodes(g, "gcn_h")
        gin_agg = dgl.mean_nodes(g, "gin_h")

        # Concatenate graph representations
        concatenated = torch.cat((gcn_agg, gin_agg), dim=2)
        #print("Shape of concatenated tensor:", concatenated.shape)

        # FFNN processing
        ffnn_out = concatenated
        for i, layer in enumerate(self.ffnn_layers):
            ffnn_out = F.relu(layer(ffnn_out))
            if i < len(self.ffnn_layers) - 1:
                ffnn_out = F.dropout(ffnn_out, self.ffnn_dropout_rate, training=self.training)
        return torch.sigmoid(self.fc_final(ffnn_out))



