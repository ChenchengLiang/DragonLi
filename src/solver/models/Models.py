import time

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, GINConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from dgl.nn.pytorch import SumPooling
from src.solver.independent_utils import color_print, hash_one_dgl_graph

from torch import nn, optim
import pytorch_lightning as pl

from src.solver.models.utils import squeeze_labels, save_model_local_and_mlflow
import torchmetrics
from torchmetrics import Metric


############################################# Rank task 1 #############################################


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        target = torch.argmax(target, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()


class GraphClassifierLightning(pl.LightningModule):
    def __init__(self, shared_gnn, classifier, model_parameters):
        super().__init__()

        self.shared_gnn = shared_gnn
        self.classifier = classifier
        self.model_parameters = model_parameters

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = MyAccuracy()  # torchmetrics.Accuracy(task="multiclass", num_classes=classifier.output_dim)

        self.best_val_accuracy = 0
        self.best_epoch = 0
        self.last_train_loss = float("inf")
        self.last_train_accuracy = 0

        self.is_test = False

    def forward(self, graphs):
        gnn_output = self.shared_gnn(graphs, is_test=self.is_test)
        final_output = self.classifier(gnn_output)
        return final_output

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)

        accuracy = self.accuracy(scores, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy},
                      on_step=False, on_epoch=True, prog_bar=False)
        self.last_train_loss=loss
        self.last_train_accuracy=accuracy
        return {'loss': loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)

        result_dict = {'current_epoch': int(self.current_epoch), 'val_loss': loss,
                       'val_accuracy': accuracy, "best_val_accuracy": self.best_val_accuracy,
                       "best_epoch": self.best_epoch}
        self.log_dict(result_dict,
                      on_step=False, on_epoch=True, prog_bar=False)

        print(
            f"Epoch {self.current_epoch}: train_loss: {self.last_train_loss:.4f}, "
            f"train_accuracy: {self.last_train_accuracy:.4f}, val_loss: {loss:.4f}, val_accuracy: {accuracy:.4f}")
        # mlflow.log_metrics(result_dict)

        # store best model
        if accuracy > self.best_val_accuracy:
            self.best_val_accuracy = accuracy
            self.best_epoch = self.current_epoch
            save_model_local_and_mlflow(self.model_parameters, self.classifier.output_dim, self)
            color_print(f"\nbest_val_accuracy: {self.best_val_accuracy}, best_epoch: {self.best_epoch}, Save model\n",
                        "green")

        return {'loss': loss, "scores": scores, "y": y, "best_val_accuracy": self.best_val_accuracy}

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        scores, y = squeeze_labels(scores, y)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.model_parameters["learning_rate"])


class GNNRankTask1(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, gnn_dropout_rate=0.5, embedding_type='GCN'):
        super(GNNRankTask1, self).__init__()
        embedding_class = GCNEmbedding if embedding_type == 'GCN' else GINEmbedding
        self.embedding = embedding_class(num_node_types=input_feature_dim, hidden_feats=gnn_hidden_dim,
                                         num_gnn_layers=gnn_layer_num, dropout_rate=gnn_dropout_rate)
        self.gnn_hidden_dim = gnn_hidden_dim

        self.single_dgl_hash_table = {}
        self.single_dgl_hash_table_hit = 0

    # dealing batch graphs
    def forward(self, batch_graphs, is_test=False):
        # todo these operations may be optimized
        # print("inside model: single_dgl_hash_table_hit", self.single_dgl_hash_table_hit)
        # print("inside model: single_dgl_hash_table size", len(self.single_dgl_hash_table))

        batch_result_list = []
        for one_data in batch_graphs:
            one_data_unbatched = dgl.unbatch(one_data)
            embeddings = []

            for g in one_data_unbatched:
                if is_test:  # infer
                    hashed_data, _ = hash_one_dgl_graph(g)

                    if hashed_data in self.single_dgl_hash_table:
                        dgl_embedding = self.single_dgl_hash_table[hashed_data]
                        self.single_dgl_hash_table_hit += 1

                    else:
                        dgl_embedding = self.embedding(g)
                        self.single_dgl_hash_table[hashed_data] = dgl_embedding

                    embeddings.append(dgl_embedding)
                else:  # train and validation
                    embeddings.append(self.embedding(g))

            first_element = embeddings[0]
            summed_tensor = torch.zeros(1, 1, self.gnn_hidden_dim, device=first_element[0].device)
            for e in embeddings[1:]:
                summed_tensor += e
            summed_tensor = summed_tensor / len(embeddings[1:])

            concatenated_embedding = torch.cat([first_element, summed_tensor], dim=2)
            batch_result_list.append(concatenated_embedding)

        batch_result_list_stacked = torch.stack(batch_result_list, dim=0)
        return batch_result_list_stacked


############################################# Branch Task 3 #############################################

class GraphClassifier(nn.Module):
    def __init__(self, shared_gnn, classifier):
        super(GraphClassifier, self).__init__()
        self.shared_gnn = shared_gnn
        self.classifier = classifier
        self.is_test = False

    def forward(self, graphs):
        gnn_output = self.shared_gnn(graphs, is_test=self.is_test)
        final_output = self.classifier(gnn_output)
        return final_output


class Classifier(nn.Module):
    def __init__(self, ffnn_hidden_dim, ffnn_layer_num, output_dim, ffnn_dropout_rate=0.5, parent_node=True):
        super(Classifier, self).__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        if output_dim == 1:  # adapt to BCELoss
            self.layers.append(nn.Linear(ffnn_hidden_dim * 2, ffnn_hidden_dim))
        else:
            if parent_node == True:
                self.layers.append(nn.Linear(ffnn_hidden_dim * (output_dim + 1), ffnn_hidden_dim))  # with parent node
            else:
                self.layers.append(nn.Linear(ffnn_hidden_dim * (output_dim), ffnn_hidden_dim))  # without parent node
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
        if self.output_dim == 1:  # adapt to BCELoss
            return torch.sigmoid(self.final_fc(x))
        else:
            return self.final_fc(x)


class BaseEmbedding(nn.Module):
    def __init__(self, num_node_types, hidden_feats, num_gnn_layers, dropout_rate):
        super(BaseEmbedding, self).__init__()
        self.node_embedding = nn.Embedding(num_node_types, hidden_feats)
        self.dropout = nn.Dropout(dropout_rate)
        self.gnn_layers = nn.ModuleList()
        self._build_layers(hidden_feats, num_gnn_layers)

        # self.gating_network = GatingNetwork(hidden_feats)
        # self.global_attention_pooling = GlobalAttentionPooling(self.gating_network)

    def _build_layers(self, hidden_feats, num_gnn_layers):
        # To be implemented in the subclasses
        raise NotImplementedError

    def forward(self, g):
        h = self.node_embedding(g.ndata['feat'])

        for i, layer in enumerate(self.gnn_layers):
            h = self.apply_layer(g, h, layer, i)

        g.ndata['h'] = h
        # hg=self.global_attention_pooling(g, h)
        hg = dgl.mean_nodes(g, 'h')
        return hg

    def apply_layer(self, g, h, layer, layer_idx):
        h = layer(g, h)
        if layer_idx < len(self.gnn_layers) - 1:
            h = F.relu(self.dropout(h))
        else:
            h = F.relu(h)
        return h


class GCNEmbedding(BaseEmbedding):
    def __init__(self, num_node_types, hidden_feats, num_gnn_layers, dropout_rate):
        super(GCNEmbedding, self).__init__(num_node_types, hidden_feats, num_gnn_layers, dropout_rate)
        self._build_layers(hidden_feats, num_gnn_layers)

    def _build_layers(self, hidden_feats, num_gnn_layers):
        self.gnn_layers.append(GraphConv(hidden_feats, hidden_feats))
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GraphConv(hidden_feats, hidden_feats))


class GINEmbedding(BaseEmbedding):
    def __init__(self, num_node_types, hidden_feats, num_gnn_layers, dropout_rate):
        super(GINEmbedding, self).__init__(num_node_types, hidden_feats, num_gnn_layers, dropout_rate)
        self._build_layers(hidden_feats, num_gnn_layers)

    def _build_layers(self, hidden_feats, num_gnn_layers):
        # mlp = nn.Sequential(nn.Linear(hidden_feats, hidden_feats), nn.ReLU(),
        #                     nn.Linear(hidden_feats, hidden_feats))
        mlp = nn.Sequential(nn.Linear(hidden_feats, hidden_feats), nn.ReLU())
        self.gnn_layers.append(GINConv(mlp, learn_eps=True))
        for _ in range(num_gnn_layers - 1):
            mlp = nn.Sequential(nn.Linear(hidden_feats, hidden_feats), nn.ReLU())
            self.gnn_layers.append(GINConv(mlp, learn_eps=True))


class SharedGNN(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, gnn_dropout_rate=0.5, embedding_type='GCN'):
        super(SharedGNN, self).__init__()
        embedding_class = GCNEmbedding if embedding_type == 'GCN' else GINEmbedding
        self.embedding = embedding_class(num_node_types=input_feature_dim, hidden_feats=gnn_hidden_dim,
                                         num_gnn_layers=gnn_layer_num, dropout_rate=gnn_dropout_rate)

    def forward(self, graphs):
        embeddings = [self.embedding(g) for g in graphs]
        concatenated_embedding = torch.cat(embeddings, dim=2)

        return concatenated_embedding


############################################# Task 1, 2 #############################################


class BaseWithNFFNN(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, n_ffnn, ffnn_hidden_size, gnn_dropout_rate=0.5,
                 ffnn_dropout_rate=0.5):
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
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, ffnn_layer_num, ffnn_hidden_dim,
                 gnn_dropout_rate=0.5, ffnn_dropout_rate=0.5):
        super(GCNWithNFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim,
                                           gnn_dropout_rate=gnn_dropout_rate, ffnn_dropout_rate=ffnn_dropout_rate)
        self.conv_layers = nn.ModuleList([GraphConv(gnn_hidden_dim, gnn_hidden_dim) for _ in range(gnn_layer_num)])

    def gnn_forward(self, g, h):
        for i, layer in enumerate(self.conv_layers):
            h = F.relu(layer(g, h))
            if i < len(self.conv_layers) - 1:  # Apply dropout to all layers except the last one
                h = F.dropout(h, self.gnn_dropout_rate, training=self.training)
        return h


class GATWithNFFNN(BaseWithNFFNN):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, num_heads, ffnn_layer_num, ffnn_hidden_dim,
                 gnn_dropout_rate=0.5, ffnn_dropout_rate=0.5):
        super(GATWithNFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim,
                                           gnn_dropout_rate=gnn_dropout_rate, ffnn_dropout_rate=ffnn_dropout_rate)
        self.gat_layers = nn.ModuleList(
            [GATConv(gnn_hidden_dim, gnn_hidden_dim, num_heads) for _ in range(gnn_layer_num - 1)])
        self.gat_final = GATConv(gnn_hidden_dim * num_heads, gnn_hidden_dim, 1)

    def gnn_forward(self, g, h):
        for i, layer in enumerate(self.gat_layers):
            h = F.elu(layer(g, h).view(h.size(0), -1))
            if i < len(self.gat_layers) - 1:  # apply dropout to all layers except the last one
                h = F.dropout(h, self.gnn_dropout_rate, training=self.training)
        return self.gat_final(g, h).squeeze(1)


class GINWithNFFNN(BaseWithNFFNN):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, ffnn_layer_num, ffnn_hidden_dim,
                 gnn_dropout_rate=0.5, ffnn_dropout_rate=0.5):
        super(GINWithNFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim,
                                           gnn_dropout_rate=gnn_dropout_rate, ffnn_dropout_rate=ffnn_dropout_rate)
        # mlp = nn.Sequential(nn.Linear(gnn_hidden_dim, gnn_hidden_dim), nn.ReLU(), nn.Linear(gnn_hidden_dim, gnn_hidden_dim))
        # self.gin_layers = nn.ModuleList([GINConv(mlp, learn_eps=True) for _ in range(gnn_layer_num)])
        self.gin_layers = nn.ModuleList()
        for _ in range(gnn_layer_num):
            # mlp = nn.Sequential(nn.Linear(gnn_hidden_dim, gnn_hidden_dim), nn.ReLU(),
            #                     nn.Linear(gnn_hidden_dim, gnn_hidden_dim))
            mlp = nn.Sequential(nn.Linear(gnn_hidden_dim, gnn_hidden_dim), nn.ReLU())
            self.gin_layers.append(GINConv(mlp, learn_eps=True))

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
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, ffnn_layer_num, ffnn_hidden_dim,
                 gnn_dropout_rate=0.5, ffnn_dropout_rate=0.5):
        super(GCNWithGAPFFNN, self).__init__(input_feature_dim, gnn_hidden_dim, ffnn_layer_num, ffnn_hidden_dim,
                                             gnn_dropout_rate, ffnn_dropout_rate)
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
        # print("Shape of concatenated tensor:", concatenated.shape)

        # FFNN processing
        ffnn_out = concatenated
        for i, layer in enumerate(self.ffnn_layers):
            ffnn_out = F.relu(layer(ffnn_out))
            if i < len(self.ffnn_layers) - 1:
                ffnn_out = F.dropout(ffnn_out, self.ffnn_dropout_rate, training=self.training)
        return torch.sigmoid(self.fc_final(ffnn_out))
