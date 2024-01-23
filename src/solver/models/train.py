import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.solver.models.Models import GCNWithNFFNN,GATWithNFFNN,GINWithNFFNN
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Dict
from collections import Counter
from src.solver.Constants import project_folder
from Dataset import WordEquationDatasetBinaryClassification
import mlflow
import argparse
import json
import datetime
import subprocess
from src.solver.independent_utils import color_print
import signal
from src.solver.models.train_util import train_one_model,create_data_loaders,train_multiple_models,train_multiple_models_separately,check_run_exists,check_experiment_exists
def main():
    # parse argument
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')

    arg_parser.add_argument('--configuration_file', type=str, default=None,
                            help='path to configuration json file ')

    args = arg_parser.parse_args()

    # Accessing the arguments
    configuration_file = args.configuration_file



    if configuration_file is not None:
        #read json file
        with open(configuration_file) as f:
            train_config = json.load(f)
    else:
        task="task_3"
        model_type="GCNSplit"#GINSplit

        train_config = {
                "benchmark":"01_track_multi_word_equations_generated_train_1_40000_new_small_test","graph_type": "graph_1", "model_type": model_type,"task":task,
            "num_epochs": 50, "learning_rate": 0.001,
            "save_criterion": "valid_accuracy", "batch_size": 10000, "gnn_hidden_dim": 16,
            "gnn_layer_num": 2, "num_heads": 2, "gnn_dropout_rate":0.5,"ffnn_hidden_dim": 16, "ffnn_layer_num": 2,"ffnn_dropout_rate":0.5,
            "node_type":4,"train_step":10,"run_id":"91b35e83ce2343d7afdd46fddebc1640"
        }


    benchmark_folder = config['Path']['woorpje_benchmarks']
    today = datetime.date.today().strftime("%Y-%m-%d")

    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui'], preexec_fn=os.setpgrp)
    if "/divided" in train_config["benchmark"]:
        experiment_name = today + "-" + train_config["benchmark"].split("/")[0]
    else:
        experiment_name = today + "-" + train_config["benchmark"]
    if check_experiment_exists(train_config["experiment_id"]):
        mlflow.set_experiment(experiment_id=train_config["experiment_id"])
    else:
        mlflow.set_experiment(experiment_name)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    torch.autograd.set_detect_anomaly(True)
    if check_run_exists(train_config["run_id"]):
        with mlflow.start_run(run_id=train_config["run_id"]) as mlflow_run:
            color_print(text=f"use the existing run id {mlflow_run.info.run_id}",color="yellow")
            train_a_model(train_config,mlflow_run,benchmark_folder)
    else:
        with mlflow.start_run() as mlflow_run:
            color_print(text=f"create a new run id {mlflow_run.info.run_id}", color="yellow")
            train_a_model(train_config, mlflow_run, benchmark_folder)


    mlflow_ui_process.terminate()
    os.killpg(os.getpgid(mlflow_ui_process.pid), signal.SIGTERM)

    #todo update configuration file

    print("done")

def train_a_model(train_config,mlflow_run,benchmark_folder):
    train_config["run_id"] = mlflow_run.info.run_id
    train_config["experiment_id"] = mlflow_run.info.experiment_id
    mlflow.log_params(train_config)
    if train_config["task"] == "task_3":
        # train_multiple_models(train_config,benchmark_folder)
        train_multiple_models_separately(train_config, benchmark_folder)
    else:
        train_one_model(train_config, benchmark_folder)





if __name__ == '__main__':
    main()


