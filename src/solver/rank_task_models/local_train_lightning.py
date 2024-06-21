import configparser
import os
import sys

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

os.environ["DGLBACKEND"] = "pytorch"

import torch
import mlflow
import argparse
import json
import datetime
import subprocess
from src.solver.independent_utils import color_print, load_from_pickle_within_zip
from src.solver.Constants import project_folder, bench_folder, RED
import signal
from src.solver.rank_task_models.Dataset import WordEquationDatasetMultiClassificationRankTask, read_dataset_from_zip, \
    DGLDataModule, DGLDataModuleRank0
from src.solver.models.Models import Classifier, GNNRankTask1, GraphClassifier, SharedGNN, GraphClassifierLightning, \
    GNNRankTask0, GNNRankTask0HashTable
import torch.nn as nn
import numpy as np
import random
from src.solver.rank_task_models.train_utils import initialize_model, initialize_model_lightning, MyPrintingCallback, \
    get_gnn_and_classifier
from src.solver.models.train_util import data_loader_2, training_phase, validation_phase, log_and_save_best_model, \
    training_phase_without_loader, validation_phase_without_loader, get_data_distribution
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence  # Helpful for padding sequence data
import dgl
import dgl.data
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.profilers import PyTorchProfiler


def main():
    parameters = {}
    parameters["benchmark_folder"] = "choose_eq_train"
    mlflow_wrapper(parameters)

    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using more than one GPU
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    dgl.seed(random_seed)


def train_wrapper(parameters):
    ############### Dataset initialization ################

    parameters["graph_type"] = "graph_1"

    parameters["train_data_folder_epoch_map"]={"divided_1":0}
    parameters["configuration_file"]=f"{project_folder}/Models/configurations/config_0.json"

    ############### Model initialization ################
    dropout_rate = 0
    hidden_dim = 128

    parameters["ffnn_hidden_dim"] = hidden_dim
    parameters["ffnn_layer_num"] = 2
    parameters["ffnn_dropout_rate"] = dropout_rate

    parameters["model_type"] = "GCNSplit"
    parameters["node_type"] = 5
    parameters["gnn_hidden_dim"] = hidden_dim
    parameters["gnn_layer_num"] = 2
    parameters["gnn_dropout_rate"] = dropout_rate

    parameters["label_size"] = 2

    parameters["save_criterion"] = "valid_accuracy"
    parameters["model_save_path"] = os.path.join(project_folder, "Models",
                                                 f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")
    parameters["current_train_folder"] = "divided_1"
    parameters["batch_size"] = 1000
    parameters["learning_rate"] = 0.001

    parameters["num_epochs"] = 10
    parameters["train_step"] = 10

    # Decide on the GNN type based on parameters
    embedding_type = "GCN" if parameters["model_type"] == "GCNSplit" else "GIN"
    if parameters["model_type"] not in ["GCNSplit", "GINSplit"]:
        raise ValueError("Unsupported model type")

    gnn_model = GNNRankTask1(
        input_feature_dim=parameters["node_type"],
        gnn_hidden_dim=parameters["gnn_hidden_dim"],
        gnn_layer_num=parameters["gnn_layer_num"],
        gnn_dropout_rate=parameters["gnn_dropout_rate"],
        embedding_type=embedding_type
    )
    first_layer_ffnn_hidden_dim_factor= 2 if type(gnn_model) == GNNRankTask1 else 1


    classifier_2 = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                              ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=2,
                              first_layer_ffnn_hidden_dim_factor=first_layer_ffnn_hidden_dim_factor,
                              ffnn_dropout_rate=parameters["ffnn_dropout_rate"])


    model = GraphClassifierLightning(gnn_model, classifier_2, model_parameters=parameters)

    logger = MLFlowLogger(experiment_name=parameters["experiment_name"], run_id=parameters["run_id"])
    profiler = "simple"

    dm = DGLDataModule(parameters, parameters["batch_size"], num_workers=4)

    trainer = pl.Trainer(
        profiler=profiler,
        accelerator="gpu",
        devices=1,
        min_epochs=10,
        max_epochs=10,
        precision=32,
        callbacks=MyPrintingCallback(),
        logger=logger
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    # trainer.test(model, valid_dataloader)


def mlflow_wrapper(parameters):
    today = datetime.date.today().strftime("%Y-%m-%d")
    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui'], preexec_fn=os.setpgrp)
    experiment_name = today + "-" + parameters["benchmark_folder"]
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run() as mlflow_run:
        parameters["run_id"] = mlflow_run.info.run_id
        parameters["experiment_name"] = experiment_name
        parameters["experiment_id"] = mlflow_run.info.experiment_id
        train_wrapper(parameters)

    mlflow_ui_process.terminate()
    os.killpg(os.getpgid(mlflow_ui_process.pid), signal.SIGTERM)


if __name__ == '__main__':
    main()
