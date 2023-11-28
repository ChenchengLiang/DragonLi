import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

os.environ["DGLBACKEND"] = "pytorch"

import mlflow
import datetime
import subprocess
import signal
from src.solver.models.train_util import train_one_model,create_data_loaders
def main():
    hidden_dimention=32
    train_config_list=[
        # {
        #     "benchmark": "01_track_generated_SAT_train", "graph_type": "graph_1", "model_type": "GCN", "num_epochs": 300,
        #     "learning_rate": 0.001,
        #     "save_criterion": "valid_accuracy", "batch_size": 1000, "gnn_hidden_dim": 128,
        #     "gnn_layer_num": 2, "num_heads": 2, "gnn_dropout_rate": 0.5, "ffnn_hidden_dim": 128, "ffnn_layer_num": 2,
        #     "ffnn_dropout_rate": 0.5
        # },
        # {
        #     "benchmark": "example_train", "graph_type": "graph_1", "model_type": "GIN", "num_epochs": 2,
        #     "learning_rate": 0.001,
        #     "save_criterion": "valid_accuracy", "batch_size": 1000, "gnn_hidden_dim": 128,
        #     "gnn_layer_num": 2, "num_heads": 2, "gnn_dropout_rate": 0.5, "ffnn_hidden_dim": 128, "ffnn_layer_num": 2,
        #     "ffnn_dropout_rate": 0.5
        # },
        # {
        #     "benchmark": "01_track_generated_SAT_train", "graph_type": "graph_1", "model_type": "GCNwithGAP",
        #     "num_epochs": 2,
        #     "learning_rate": 0.001,
        #     "save_criterion": "valid_accuracy", "batch_size": 1000, "gnn_hidden_dim": hidden_dimention,
        #     "gnn_layer_num": 2, "num_heads": 2, "gnn_dropout_rate": 0.5, "ffnn_hidden_dim": hidden_dimention, "ffnn_layer_num": 2,
        #     "ffnn_dropout_rate": 0.5
        # },
        {
            "benchmark": "example_train", "graph_type": "graph_1", "model_type": "MultiGNNs",
            "num_epochs": 2,
            "learning_rate": 0.001,
            "save_criterion": "valid_accuracy", "batch_size": 1000, "gnn_hidden_dim": hidden_dimention,
            "gnn_layer_num": 2, "num_heads": 2, "gnn_dropout_rate": 0.5, "ffnn_hidden_dim": hidden_dimention,
            "ffnn_layer_num": 2,
            "ffnn_dropout_rate": 0.5
        }
    ]


    benchmark_folder = config['Path']['woorpje_benchmarks']

    for train_config in train_config_list:
        mlflow_ui_process = subprocess.Popen(['mlflow', 'ui'], preexec_fn=os.setpgrp)
        today = datetime.date.today().strftime("%Y-%m-%d")
        mlflow.set_experiment(today+"-"+train_config["benchmark"])
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run():
            mlflow.log_params(train_config)
            train_one_model(train_config,benchmark_folder)


    mlflow_ui_process.terminate()

    os.killpg(os.getpgid(mlflow_ui_process.pid), signal.SIGTERM)

    print("done")




if __name__ == '__main__':
    main()


