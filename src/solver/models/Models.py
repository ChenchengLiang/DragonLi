import signal
import time
from typing import Dict, Any

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, GINConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from dgl.nn.pytorch import SumPooling
from src.solver.independent_utils import color_print, hash_one_dgl_graph, get_folders, kill_gunicorn_processes, \
    identity_function

from torch import nn, optim
import pytorch_lightning as pl

from src.solver.models.utils import squeeze_labels, save_model_local_and_mlflow, device_info, update_config_file
from torchmetrics import Metric
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import datetime
import subprocess
import os
from src.solver.Constants import project_folder, bench_folder, checkpoint_folder
from pytorch_lightning.loggers import MLFlowLogger
from functools import partial

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
        self.total_epoch = 0
        self.best_global_step = 0
        self.step = 0

        self.last_train_loss = float("inf")
        self.last_train_accuracy = 0
        self.last_val_loss = float("inf")
        self.last_val_accuracy = 0

        self.is_test = False

    def forward(self, graphs):
        gnn_output = self.shared_gnn(graphs, is_test=self.is_test)
        final_output = self.classifier(gnn_output)
        return final_output

    def training_step(self, batch, batch_idx):
        # self.total_epoch += 1
        loss, scores, y = self._common_step(batch, batch_idx)

        accuracy = float(self.accuracy(scores, y))
        self.last_train_loss = float(loss)
        self.last_train_accuracy = accuracy

        return {'loss': loss, "scores": scores, "y": y,"accuracy":accuracy}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = float(self.accuracy(scores, y))
        self.last_val_loss = float(loss)
        self.last_val_accuracy = accuracy

        base_step = int(self.model_parameters["train_data_folder_epoch_map"][
                            os.path.basename(self.model_parameters["current_train_folder"])])
        self.step = base_step + self.global_step

        # store best model
        if accuracy > self.best_val_accuracy:
            self.best_val_accuracy = accuracy
            self.best_epoch = self.total_epoch
            self.best_global_step = self.step
            if "run_id" in self.model_parameters and self.model_parameters["run_id"] is not None:
                save_model_local_and_mlflow(self.model_parameters, self.classifier.output_dim, self)

            color_print(f"\nbest_val_accuracy: {self.best_val_accuracy}, best_epoch: {self.best_epoch}\n",
                        "green")

        return {'loss': loss, "scores": scores, "y": y, "accuracy":accuracy}

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

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["best_val_accuracy"] = self.best_val_accuracy
        checkpoint["best_epoch"] = self.best_epoch
        checkpoint["total_epoch"] = self.total_epoch
        checkpoint["best_global_step"] = self.best_global_step
        checkpoint['last_train_loss'] = self.last_train_loss
        checkpoint['last_train_accuracy'] = self.last_train_accuracy
        checkpoint['last_val_loss'] = self.last_val_loss
        checkpoint['last_val_accuracy'] = self.last_val_accuracy

    @rank_zero_only
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.best_val_accuracy = checkpoint["best_val_accuracy"]
        self.best_epoch = checkpoint["best_epoch"]
        self.total_epoch = checkpoint["total_epoch"]
        self.best_global_step = checkpoint["best_global_step"]
        self.last_train_loss = checkpoint['last_train_loss']
        self.last_train_accuracy = checkpoint['last_train_accuracy']
        self.last_val_loss = checkpoint['last_val_loss']
        self.last_val_accuracy = checkpoint['last_val_accuracy']

    @rank_zero_only
    def on_train_start(self):
        device_info()
        self.model_parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if self.total_epoch < 1:  # indicating initial training
            # if "run_id" in self.model_parameters and self.model_parameters["run_id"] is not None:
            artifact_folder = f"{project_folder}/mlruns/{self.model_parameters['experiment_id']}/{self.model_parameters['run_id']}/artifacts"
            with open(
                    f"{artifact_folder}/data_distribution_{self.model_parameters['label_size']}.txt",
                    'w') as file:
                file.write(self.model_parameters["data_distribution_str"])

            with open(
                    f"{artifact_folder}/{os.path.basename(self.model_parameters['current_train_folder'])}_dataset_statistics.txt",
                    'w') as file:
                file.write(self.model_parameters["dataset_statistics_str"])
            # store configuration file to artifact folder
            update_config_file(
                f"{artifact_folder}/configuration_model_{self.model_parameters['label_size']}_{self.model_parameters['graph_type']}_{self.model_parameters['model_type']}.json",
                self.model_parameters)

            self.logger.log_hyperparams(self.model_parameters)
        # mlflow.log_params(self.model_parameters)

    @rank_zero_only
    def on_train_end(self):

        if self.total_epoch > 1:  # indicating not initial training
            self.model_parameters["train_data_folder_epoch_map"][
                os.path.basename(self.model_parameters["current_train_folder"])] += \
                self.model_parameters["train_step"]

        update_config_file(self.model_parameters["configuration_file"], self.model_parameters)
        color_print("update_config_file done", "green")

    def on_train_epoch_end(self):
        self.total_epoch += 1

    def on_validation_epoch_end(self) -> None:
        print(
            f"Epoch {self.current_epoch}: train_loss: {self.last_train_loss:.4f}, "
            f"train_accuracy: {self.last_train_accuracy:.4f}, val_loss: {self.last_val_loss:.4f},"
            f" val_accuracy: {self.last_val_accuracy:.4f}, best_val_accuracy: {self.best_val_accuracy:.4f}, total_epoch: {self.total_epoch}")

        current_folder_number = self.model_parameters["current_train_folder"].split("_")[-1]
        result_dict = {'current_epoch': int(self.current_epoch), 'val_loss': float(self.last_val_loss),
                       'val_accuracy': self.last_val_accuracy, "best_val_accuracy": self.best_val_accuracy,
                       "best_epoch": self.best_epoch, "total_epoch": int(self.total_epoch),
                       "folder": int(current_folder_number), "global_step": int(self.step),
                       "best_global_step": int(self.best_global_step), "train_loss": float(self.last_train_loss),
                       "train_accuracy": self.last_train_accuracy}
        # self.log_dict(result_dict,
        #               on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.log('best_val_accuracy', self.best_val_accuracy, on_step=False, on_epoch=True)

        self.logger.log_metrics(result_dict,step=self.total_epoch)


class GNNRankTask2(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, gnn_dropout_rate=0.5, embedding_type='GCN',
                 pooling_type='mean'):
        super(GNNRankTask2, self).__init__()
        embedding_class = GCNEmbedding if embedding_type == 'GCN' else GINEmbedding
        self.embedding = embedding_class(num_node_types=input_feature_dim, hidden_feats=gnn_hidden_dim,
                                         num_gnn_layers=gnn_layer_num, dropout_rate=gnn_dropout_rate)
        if pooling_type == 'mean':
            self.pooling = self.mean_gnn_embeddings
        elif pooling_type == 'concat':
            self.pooling = self.concat_gnn_embeddings

        self.single_dgl_hash_table = {}
        self.single_dgl_hash_table_hit = 0

    def concat_gnn_embeddings(self, embedding_list):
        return embedding_list.view(-1)

    def mean_gnn_embeddings(self, embedding_list):
        embedding_mean = embedding_list.mean(dim=0)  # Compute mean along dimension 0
        return embedding_mean.squeeze(0)

    # dealing batch graphs
    def forward(self, batch_graphs, is_test=False):
        if is_test == True:
            # print(f"hash table length {len(self.single_dgl_hash_table)}","single_dgl_hash_table_hit",self.single_dgl_hash_table_hit)
            batch_vector_list = []
            embedding_list = []
            for one_eq_graph in batch_graphs:
                hashed_data, _ = hash_one_dgl_graph(one_eq_graph)
                if hashed_data in self.single_dgl_hash_table:
                    dgl_embedding = self.single_dgl_hash_table[hashed_data]
                    self.single_dgl_hash_table_hit += 1
                else:
                    dgl_embedding = self.embedding(one_eq_graph)
                    self.single_dgl_hash_table[hashed_data] = dgl_embedding
                embedding_list.append(dgl_embedding)
            embedding_list = torch.stack(embedding_list, dim=0)

            pooled_embedding = self.pooling(embedding_list)
            batch_vector_list.append(pooled_embedding)
            batch_vector_stack = torch.stack(batch_vector_list)

        else:
            batch_vector_list = []
            for one_data in batch_graphs:
                embedding_list = self.embedding(one_data)
                pooled_embedding = self.pooling(embedding_list)
                batch_vector_list.append(pooled_embedding)

            batch_vector_stack = torch.stack(batch_vector_list)

        return batch_vector_stack


class GNNRankTask0(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, gnn_dropout_rate=0.5, embedding_type='GCN'):
        super(GNNRankTask0, self).__init__()
        embedding_class = GCNEmbedding if embedding_type == 'GCN' else GINEmbedding
        self.embedding = embedding_class(num_node_types=input_feature_dim, hidden_feats=gnn_hidden_dim,
                                         num_gnn_layers=gnn_layer_num, dropout_rate=gnn_dropout_rate)

    # dealing batch graphs
    def forward(self, batch_graphs, is_test=False):
        embedded_graphs = self.embedding(batch_graphs)
        return embedded_graphs


class GNNRankTask0UnbachLoop(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, gnn_dropout_rate=0.5, embedding_type='GCN'):
        super(GNNRankTask0UnbachLoop, self).__init__()
        embedding_class = GCNEmbedding if embedding_type == 'GCN' else GINEmbedding
        self.embedding = embedding_class(num_node_types=input_feature_dim, hidden_feats=gnn_hidden_dim,
                                         num_gnn_layers=gnn_layer_num, dropout_rate=gnn_dropout_rate)

    # dealing batch graphs
    def forward(self, batch_graphs, is_test=False):
        embedded_graphs = []
        batch_graphs = dgl.unbatch(batch_graphs)
        for g in batch_graphs:
            embedded_graphs.append(self.embedding(g))

        embedded_graphs = torch.stack(embedded_graphs, dim=0)

        return embedded_graphs


class GNNRankTask0HashTable(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, gnn_dropout_rate=0.5, embedding_type='GCN'):
        super(GNNRankTask0HashTable, self).__init__()
        embedding_class = GCNEmbedding if embedding_type == 'GCN' else GINEmbedding
        self.embedding = embedding_class(num_node_types=input_feature_dim, hidden_feats=gnn_hidden_dim,
                                         num_gnn_layers=gnn_layer_num, dropout_rate=gnn_dropout_rate)

        self.single_dgl_hash_table = {}
        self.single_dgl_hash_table_hit = 0

    def forward(self, batch_graphs, is_test=False):
        if is_test:
            embedded_graphs = []
            batch_graphs = dgl.unbatch(batch_graphs)
            for g in batch_graphs:
                hashed_data, _ = hash_one_dgl_graph(g)
                if hashed_data in self.single_dgl_hash_table:
                    dgl_embedding = self.single_dgl_hash_table[hashed_data]
                    self.single_dgl_hash_table_hit += 1
                else:
                    dgl_embedding = self.embedding(g)
                    self.single_dgl_hash_table[hashed_data] = dgl_embedding
                embedded_graphs.append(dgl_embedding)
            embedded_graphs = torch.stack(embedded_graphs, dim=0)
        else:
            embedded_graphs = self.embedding(batch_graphs)

        return embedded_graphs


class GNNRankTask1BatchProcess(nn.Module):
    def __init__(self, input_feature_dim, gnn_hidden_dim, gnn_layer_num, gnn_dropout_rate=0.5, embedding_type='GCN'):
        super(GNNRankTask1BatchProcess, self).__init__()
        embedding_class = GCNEmbedding if embedding_type == 'GCN' else GINEmbedding
        self.embedding = embedding_class(num_node_types=input_feature_dim, hidden_feats=gnn_hidden_dim,
                                         num_gnn_layers=gnn_layer_num, dropout_rate=gnn_dropout_rate)
        self.gnn_hidden_dim = gnn_hidden_dim

    # dealing batch graphs
    def forward(self, batch_graphs, is_test=False):
        batch_result_list = []
        for one_data in batch_graphs:
            embeddings = self.embedding(one_data)

            # g concat GNN embeddings
            first_element = embeddings[0]
            embed_slice = embeddings[1:]
            mean_tensor = torch.mean(embed_slice, dim=0)
            concatenated_embedding = torch.cat([first_element, mean_tensor], dim=1)
            batch_result_list.append(concatenated_embedding)

        batch_result_list_stacked = torch.stack(batch_result_list, dim=0)
        return batch_result_list_stacked


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
    def __init__(self, ffnn_hidden_dim, ffnn_layer_num, output_dim, ffnn_dropout_rate=0.5,
                 first_layer_ffnn_hidden_dim_factor=2):
        super(Classifier, self).__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(ffnn_hidden_dim * first_layer_ffnn_hidden_dim_factor, ffnn_hidden_dim))

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

class ClassifierMultiFilter(nn.Module):
    def __init__(self, ffnn_hidden_dim, ffnn_layer_num, output_dim, ffnn_dropout_rate=0.5,
                 first_layer_ffnn_hidden_dim_factor=2,num_filters=1,pool_type="concat"):
        super(ClassifierMultiFilter, self).__init__()
        self.output_dim = output_dim
        self.num_filters=num_filters
        self.pool_type=pool_type

        self.filters = nn.ModuleList()
        for _ in range(num_filters):
            one_filter = nn.ModuleList()
            #first layer for one_filter
            one_filter.append(nn.Linear(ffnn_hidden_dim * first_layer_ffnn_hidden_dim_factor, ffnn_hidden_dim))
            for _ in range(ffnn_layer_num):
                one_filter.append(nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim))

            self.filters.append(one_filter)

        pool_size_map={"concat":num_filters*ffnn_hidden_dim,"max":ffnn_hidden_dim,"min":ffnn_hidden_dim}
        pool_func_map={"concat":partial(torch.cat,dim=1),"max":partial(torch.max,dim=0),"min":partial(torch.min,dim=0)}
        pool_func_stack_map = {"concat": identity_function, "max": torch.stack,"min":torch.stack}
        self.pool_func_stack=pool_func_stack_map[pool_type]
        pool_size=pool_size_map[pool_type]
        self.pool_func=pool_func_map[pool_type]
        self.final_fc = nn.Linear(pool_size, output_dim)
        self.dropout = nn.Dropout(ffnn_dropout_rate)

    def forward(self, x):

        filter_outputs = []
        for filter_layers in self.filters:
            x_filter = x
            for i, layer in enumerate(filter_layers):
                x_filter = layer(x_filter)
                x_filter = F.relu(self.dropout(x_filter))
            filter_outputs.append(torch.squeeze(x_filter))

        stacked_filter_output=self.pool_func_stack(filter_outputs)
        if self.pool_type == "concat":
            pooled_output = self.pool_func(stacked_filter_output)
        else:
            pooled_output=self.pool_func(stacked_filter_output)[0]
        print("pooled_output",pooled_output.shape)

        # Pass the pooled output through the final layer
        if self.output_dim == 1:  # adapt to BCELoss
            return torch.sigmoid(self.final_fc(pooled_output))
        else:
            return self.final_fc(pooled_output)


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
