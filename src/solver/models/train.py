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
        today = datetime.date.today().strftime("%Y-%m-%d")
        task="task_3"
        model_type="GCNSplit"#GINSplit
        benchmark="debug-train"

        train_config = {
                "benchmark":benchmark,"graph_type": "graph_1", "model_type": model_type,"task":task,
            "num_epochs": 10, "learning_rate": 0.001,"share_gnn":False,
            "save_criterion": "valid_accuracy", "batch_size": 1000, "gnn_hidden_dim": 128,
            "gnn_layer_num": 2, "num_heads": 2, "gnn_dropout_rate":0.5,"ffnn_hidden_dim": 128, "ffnn_layer_num": 2,"ffnn_dropout_rate":0.5,
            "node_type":4,"train_step":10,"run_id":None,"experiment_name":today + "-" + benchmark,"experiment_id":None
        }



    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui'], preexec_fn=os.setpgrp)
    mlflow.set_experiment(train_config["experiment_name"])
    # if check_experiment_exists(train_config["experiment_id"]):
    #     mlflow.set_experiment(experiment_name=train_config["experiment_name"],experiment_id=train_config["experiment_id"])
    # else:
    #     mlflow.set_experiment(train_config["experiment_name"])


    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    torch.autograd.set_detect_anomaly(True)
    if check_run_exists(train_config["run_id"]):
        with mlflow.start_run(run_id=train_config["run_id"]) as mlflow_run:
            color_print(text=f"use the existing run id {mlflow_run.info.run_id}",color="yellow")
            # pick one unfinished train
            train_config["current_train_folder"]=None
            for key,value in train_config["train_data_folder_epoch_map"].items():
                if value<train_config["num_epochs"]:
                    train_config["current_train_folder"]=train_config["benchmark"]+"/"+key
                    break

            if train_config["current_train_folder"] is None:
                color_print(text=f"all training folders are done", color="green")
            else:
                color_print(text=f"current training folder:{train_config['current_train_folder']}", color="yellow")
                train_config=train_a_model(train_config,mlflow_run)
                train_config["train_data_folder_epoch_map"][os.path.basename(train_config["current_train_folder"])]+=train_config["train_step"]
                # update configuration file
                with open(configuration_file, 'w') as f:
                    json.dump(train_config, f, indent=4)
    else:
        with mlflow.start_run() as mlflow_run:
            train_config["current_train_folder"] = benchmark
            color_print(text=f"create a new run id {mlflow_run.info.run_id}", color="yellow")
            train_a_model(train_config, mlflow_run,)

    mlflow_ui_process.terminate()
    os.killpg(os.getpgid(mlflow_ui_process.pid), signal.SIGTERM)


    print("done")

def train_a_model(train_config,mlflow_run):
    print("-"*10)
    color_print(f"torch.cuda.is_available: {torch.cuda.is_available()}","green")
    color_print(f"torch vesion: {torch.__version__}","green")
    print("-" * 10)

    benchmark_folder = config['Path']['woorpje_benchmarks']
    train_config["run_id"] = mlflow_run.info.run_id
    train_config["experiment_id"] = mlflow_run.info.experiment_id
    if len(mlflow_run.data.params)==0:
        mlflow.log_params(train_config)
    if train_config["task"] == "task_3":
        # train_multiple_models(train_config,benchmark_folder)
        train_multiple_models_separately(train_config, benchmark_folder)
    else:
        train_one_model(train_config, benchmark_folder)
    return train_config





if __name__ == '__main__':
    main()


