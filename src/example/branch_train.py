import os
import signal
import subprocess
import sys
import datetime
import dgl
import mlflow
import torch
sys.path.append(".")

from src.solver.models.train_util import check_run_exists, train_multiple_models_separately
from src.solver.algorithms import ElimilateVariablesRecursive
from src.solver.independent_utils import get_folders, color_print
from src.train_data_collection.draw_graphs_for_train_data import draw_graph_for_one_folder
from src.train_data_collection.utils import generate_train_data_in_one_folder

import argparse


def main():
    benchmark="01_track_train"
    benchmark_path = f"benchmaks_and_experimental_results/example/{benchmark}"
    # generate train data
    generate_train_data(benchmark_path=benchmark_path)

    # draw graphs
    graph_type = "graph_1"
    draw_graphs(benchmark_path, graph_type)

    # train
    local_train(graph_type,benchmark)


def generate_train_data(benchmark_path):
    algorithm = ElimilateVariablesRecursive
    algorithm_parameters = {"branch_method": "extract_branching_data_task_3", "extract_algorithm": "fixed",
                            "termination_condition": "termination_condition_0"}  # extract_branching_data_task_2

    folder_list = [folder for folder in get_folders(benchmark_path) if
                   "divided" in folder or "valid" in folder]
    print(folder_list)
    if len(folder_list) != 0:
        for folder in folder_list:
            generate_train_data_in_one_folder(benchmark_path + "/" + folder, algorithm, algorithm_parameters)
    else:
        generate_train_data_in_one_folder(benchmark_path, algorithm, algorithm_parameters)


def draw_graphs(benchmark_path, graph_type):
    recursion_limit = 10000
    # draw graphs from train folder
    sys.setrecursionlimit(recursion_limit)

    # read graph type from command line
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('--graph_type', type=str, default=graph_type, help='graph_type')
    args = arg_parser.parse_args()
    print(args.graph_type)

    task = "task_3"

    folder_list = [folder for folder in get_folders(benchmark_path) if
                   "divided" in folder or "valid" in folder]
    print(folder_list)
    if len(folder_list) != 0:
        for folder in folder_list:
            draw_graph_for_one_folder(args, benchmark_path + "/" + folder, task)
    else:
        draw_graph_for_one_folder(args, benchmark_path, task)



def local_train(graph_type,benchmark):
    today = datetime.date.today().strftime("%Y-%m-%d")
    task = "task_3"
    model_type = "GCNSplit"  # GINSplit
    drop_rate = 0.0
    hidden_dimention = 128
    epochs=1000

    train_config = {
        "benchmark": benchmark, "graph_type": graph_type, "model_type": model_type, "task": task,
        "num_epochs": epochs, "learning_rate": 0.001, "share_gnn": False,
        "save_criterion": "valid_accuracy", "batch_size": 1000, "gnn_hidden_dim": hidden_dimention,
        "gnn_layer_num": 2, "num_heads": 2, "gnn_dropout_rate": drop_rate, "ffnn_hidden_dim": hidden_dimention,
        "ffnn_layer_num": 2, "ffnn_dropout_rate": drop_rate,
        "node_type": 3, "train_step": epochs, "run_id": None, "experiment_name": today + "-" + benchmark,
        "experiment_id": None
    }


    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui'], preexec_fn=os.setpgrp)
    mlflow.set_experiment(train_config["experiment_name"])



    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    torch.autograd.set_detect_anomaly(True)

    with mlflow.start_run() as mlflow_run:
        train_config["current_train_folder"] = benchmark + "/divided_1"
        color_print(text=f"experiment id {mlflow_run.info.experiment_id}", color="yellow")
        color_print(text=f"create a new run id {mlflow_run.info.run_id}", color="yellow")
        _train_a_model(train_config, mlflow_run, )

    mlflow_ui_process.terminate()
    os.killpg(os.getpgid(mlflow_ui_process.pid), signal.SIGTERM)

    print("done")


def _train_a_model(train_config,mlflow_run):
    print("-"*10)
    color_print(f"torch.cuda.is_available: {torch.cuda.is_available()}","green")
    color_print(f"torch vesion: {torch.__version__}","green")
    color_print(f"dgl backend: {dgl.backend.backend_name}", "green")
    print("-" * 10)
    train_config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmark_folder = "benchmaks_and_experimental_results/example"
    train_config["run_id"] = mlflow_run.info.run_id
    train_config["experiment_id"] = mlflow_run.info.experiment_id
    if len(mlflow_run.data.params)==0:
        mlflow.log_params(train_config)

    train_multiple_models_separately(train_config, benchmark_folder)

    return train_config

if __name__ == '__main__':
    main()
