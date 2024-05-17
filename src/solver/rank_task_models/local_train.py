import configparser
import os
import sys

from src.solver.models.train_util import data_loader_2, training_phase, validation_phase, log_and_save_best_model

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import mlflow
import argparse
import json
import datetime
import subprocess
from src.solver.independent_utils import color_print, load_from_pickle_within_zip
from src.solver.Constants import project_folder, bench_folder, RED
import signal
from src.solver.rank_task_models.Dataset import WordEquationDatasetMultiClassificationRankTask
from src.solver.models.Models import Classifier, GNNRankTask1, GraphClassifier, SharedGNN
import torch.nn as nn
import numpy as np
import random

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

    parameters["data_folder"] = "divided_1"
    train_dataset = read_dataset_from_zip(parameters)
    parameters["data_folder"] = "valid_data"
    valid_dataset = read_dataset_from_zip(parameters)
    dataset = {"train": train_dataset, "valid": valid_dataset}

    ############### Model initialization ################
    dropout_rate = 0
    hidden_dim = 128

    parameters["ffnn_hidden_dim"] = hidden_dim
    parameters["ffnn_layer_num"] = 2
    parameters["ffnn_dropout_rate"] = dropout_rate

    parameters["model_type"] = "GCNSplit"
    parameters["node_type"] = 4
    parameters["gnn_hidden_dim"] = hidden_dim
    parameters["gnn_layer_num"] = 2
    parameters["gnn_dropout_rate"] = dropout_rate
    model = initialize_model(parameters)

    ############### Initialize training parameters ################
    parameters["batch_size"] = 1000
    parameters["learning_rate"] = 0.001
    train_dataloader, valid_dataloader = data_loader_2(dataset, parameters)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    loss_function = nn.CrossEntropyLoss()
    best_model = None
    best_valid_loss = float('inf')  # Initialize with a high value
    best_valid_accuracy = float('-inf')  # Initialize with a low value
    epoch_info_log = ""

    ############### Training ################
    parameters["num_epochs"] = 300
    parameters["train_step"] = 300
    start_epoch = 0
    classification_type = "multi_classification"
    parameters["label_size"] = 2
    parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", parameters["device"])
    parameters["save_criterion"] = "valid_accuracy"
    parameters["model_save_path"] = os.path.join(project_folder, "Models",
                                                 f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")
    parameters["current_train_folder"] = "divided_1"

    for index, epoch in enumerate(range(start_epoch, parameters["num_epochs"] + 1)):
        # Training Phase
        model, avg_train_loss = training_phase(model, train_dataloader, loss_function, optimizer, parameters)

        # Validation Phase
        model, avg_valid_loss, valid_accuracy = validation_phase(model, valid_dataloader, loss_function,
                                                                 classification_type, parameters)

        # Save based on specified criterion
        best_model, best_valid_loss, best_valid_accuracy, epoch_info_log = log_and_save_best_model(parameters, epoch,
                                                                                                   best_model, model,
                                                                                                   "multi_class",
                                                                                                   parameters[
                                                                                                       "label_size"],
                                                                                                   avg_train_loss,
                                                                                                   avg_valid_loss,
                                                                                                   valid_accuracy,
                                                                                                   best_valid_loss,
                                                                                                   best_valid_accuracy,
                                                                                                   epoch_info_log,
                                                                                                   index)
        if index == parameters["train_step"] - 1 or epoch >= parameters["num_epochs"] or best_valid_accuracy == 1.0:
            break

    # Return the trained model and the best metrics
    best_metrics = {f"best_valid_loss_multi_class_{parameters['label_size']}": best_valid_loss,
                    f"best_valid_accuracy_multi_class_{parameters['label_size']}": best_valid_accuracy}

    mlflow.log_metrics(best_metrics)


def mlflow_wrapper(parameters):
    today = datetime.date.today().strftime("%Y-%m-%d")
    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui'], preexec_fn=os.setpgrp)
    experiment_name = today + "-" + parameters["benchmark_folder"]
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run() as mlflow_run:
        parameters["run_id"] = mlflow_run.info.run_id
        train_wrapper(parameters)

    mlflow_ui_process.terminate()
    os.killpg(os.getpgid(mlflow_ui_process.pid), signal.SIGTERM)


def initialize_model(parameters):
    classifier_2 = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                              ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=2,
                              ffnn_dropout_rate=parameters["ffnn_dropout_rate"], parent_node=False)

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
    # Initialize GraphClassifiers with the respective GNN models
    model = GraphClassifier(gnn_model, classifier_2)
    return model


def read_dataset_from_zip(parameters):
    data_folder = parameters["data_folder"]
    pickle_folder = os.path.join(bench_folder, parameters["benchmark_folder"], data_folder)
    graph_type = parameters["graph_type"]

    # Filenames for the ZIP files
    zip_file = os.path.join(pickle_folder, f"dataset_{graph_type}.pkl.zip")
    if os.path.exists(zip_file):
        print("-" * 10, "load dataset from zipped pickle:", data_folder, "-" * 10)
        # Names of the pickle files inside ZIP archives
        pickle_name = f"dataset_{graph_type}.pkl"
        # Load the datasets directly from ZIP files
        dataset = load_from_pickle_within_zip(zip_file, pickle_name)
        dataset_statistics = dataset.statistics()
    else:
        color_print(f"Error: ZIP file not found: {zip_file}", RED)
        dataset = None
        dataset_statistics = ""

    mlflow.log_text(dataset_statistics, artifact_file=f"{data_folder}_dataset_statistics.txt")
    return dataset


if __name__ == '__main__':
    main()
