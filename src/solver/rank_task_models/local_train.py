import configparser
import os
import sys


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
from src.solver.rank_task_models.Dataset import WordEquationDatasetMultiClassificationRankTask, read_dataset_from_zip
from src.solver.models.Models import Classifier, GNNRankTask1, GraphClassifier, SharedGNN
import torch.nn as nn
import numpy as np
import random
from src.solver.rank_task_models.train_utils import initialize_model
from src.solver.models.train_util import data_loader_2, training_phase, validation_phase, log_and_save_best_model, \
    training_phase_without_loader, validation_phase_without_loader, get_data_distribution
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence  # Helpful for padding sequence data



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


    train_dataset = read_dataset_from_zip(parameters,"divided_1")
    if not os.path.exists(os.path.join(bench_folder, parameters["benchmark_folder"], "valid_data")):
        valid_dataset=train_dataset
    else:
        valid_dataset = read_dataset_from_zip(parameters,"valid_data")
    dataset = {"train": train_dataset, "valid": valid_dataset}

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
    model = initialize_model(parameters)

    ############### Initialize training parameters ################
    parameters["batch_size"] = 1000
    parameters["learning_rate"] = 0.001
    train_dataloader, valid_dataloader = data_loader_2(dataset, parameters)
    get_data_distribution(dataset, parameters)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    loss_function = nn.CrossEntropyLoss()
    best_model = None
    best_valid_loss = float('inf')  # Initialize with a high value
    best_valid_accuracy = float('-inf')  # Initialize with a low value
    epoch_info_log = ""

    ############### Training ################
    parameters["num_epochs"] = 10
    parameters["train_step"] = 10
    start_epoch = 0
    classification_type = "multi_classification"
    parameters["label_size"] = 2
    parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", parameters["device"])
    parameters["save_criterion"] = "valid_accuracy"
    parameters["model_save_path"] = os.path.join(project_folder, "Models",
                                                 f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")
    parameters["current_train_folder"] = "divided_1"

    model.to(parameters["device"])
    print("Have", torch.cuda.device_count(), "GPUs!")


    for index, epoch in enumerate(range(start_epoch, parameters["num_epochs"] + 1)):

        # # Training Phase
        # model, avg_train_loss = training_phase_without_loader(model, dataset["train"], loss_function, optimizer, parameters)
        #
        # # Validation Phase
        # model, avg_valid_loss, valid_accuracy = validation_phase_without_loader(model,  dataset["valid"], loss_function,
        #                                                          classification_type, parameters)
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





if __name__ == '__main__':
    main()
