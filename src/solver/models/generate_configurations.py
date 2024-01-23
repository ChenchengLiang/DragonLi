import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.Constants import project_folder
from src.solver.independent_utils import write_configurations_to_json_file
from src.solver.models.train_util import (initialize_model_structure,load_one_dataset,training_phase,validation_phase,
                                          initialize_train_objects,log_and_save_best_model,save_checkpoint)
from typing import Dict
import mlflow
import datetime
import subprocess
import torch
import signal

def main():
    num_epochs=30 #300
    task="task_3"
    node_type=4
    learning_rate=0.001
    train_step=10
    configurations = []
    for benchmark in ["01_track_multi_word_equations_generated_train_1_40000_new_small_test"]:#["01_track_multi_word_equations_generated_train_1_40000_new_SAT_divided_1"]:
        for graph_type in ["graph_1"]:#["graph_1", "graph_2", "graph_3", "graph_4", "graph_5"]:
            for gnn_layer_num in [2]:#[2,8]:
                for ffnn_layer_num in [2]:
                    for hidden_dim in [16]:#[128,256]:
                        for dropout_rate in [0.5]:
                            for batch_size in [10000]:
                                for model_type in ["GCNSplit","GINSplit"]:#["GCN","GIN","GCNwithGAP","MultiGNNs"]:  # ["GCN", "GAT", "GIN","GCNwithGAP","MultiGNNs"]
                                    if model_type == "GAT":
                                        for num_heads in [1]:
                                            configurations.append({
                                                "benchmark": benchmark, "graph_type": graph_type, "model_type": model_type,"task":task,
                                                "num_epochs": num_epochs, "learning_rate": learning_rate,
                                                "save_criterion": "valid_accuracy", "batch_size": batch_size, "gnn_hidden_dim": hidden_dim,
                                                "gnn_layer_num": gnn_layer_num, "num_heads": num_heads, "gnn_dropout_rate": dropout_rate,
                                                "ffnn_hidden_dim": hidden_dim, "ffnn_layer_num": ffnn_layer_num,
                                                "ffnn_dropout_rate": dropout_rate,"node_type":node_type,"train_step":train_step
                                            })
                                    else:
                                        configurations.append({
                                            "benchmark": benchmark, "graph_type": graph_type, "model_type": model_type,"task":task,
                                            "num_epochs": num_epochs,
                                            "learning_rate": learning_rate,
                                            "save_criterion": "valid_accuracy", "batch_size": batch_size, "gnn_hidden_dim": hidden_dim,
                                            "gnn_layer_num": gnn_layer_num, "num_heads": 0, "gnn_dropout_rate": dropout_rate,
                                            "ffnn_hidden_dim": hidden_dim, "ffnn_layer_num": ffnn_layer_num,
                                            "ffnn_dropout_rate": dropout_rate,"node_type":node_type,"train_step":train_step
                                        })

    #todo: train one epoch to get run id and current epochs
    continuous_train_configurations=[]
    for train_config in configurations:
        train_config=initiate_run_id_for_a_configuration(train_config)
        continuous_train_configurations.append(train_config)

    # Writing the dictionary to a JSON file
    configuration_folder = project_folder + "/Models/configurations"
    write_configurations_to_json_file(configuration_folder=configuration_folder, configurations=continuous_train_configurations)

def initiate_run_id_for_a_configuration(train_config):
    benchmark_folder = config['Path']['woorpje_benchmarks']
    today = datetime.date.today().strftime("%Y-%m-%d")

    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui'], preexec_fn=os.setpgrp)
    mlflow.set_experiment(today + "-" + train_config["benchmark"])
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    torch.autograd.set_detect_anomaly(True)
    with mlflow.start_run() as mlflow_run:
        train_config["run_id"] = mlflow_run.info.run_id
        train_config["experiment_id"] = mlflow_run.info.experiment_id
        mlflow.log_params(train_config)
        train_multiple_models_separately_get_run_id(train_config, benchmark_folder)

    mlflow_ui_process.terminate()
    os.killpg(os.getpgid(mlflow_ui_process.pid), signal.SIGTERM)
    print("done")
    return train_config

def train_multiple_models_separately_get_run_id(parameters, benchmark_folder):
    graph_folder = os.path.join(benchmark_folder, parameters["benchmark"], parameters["graph_type"])
    bench_folder = os.path.join(benchmark_folder, parameters["benchmark"])
    node_type = parameters["node_type"]
    graph_type = parameters["graph_type"]

    shared_gnn, classifier_2, classifier_3, model_2, model_3 = initialize_model_structure(parameters)

    parameters["model_save_path"] = os.path.join(project_folder, "Models",
                                                 f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")
    dataset_2 = load_one_dataset(parameters, bench_folder, graph_folder, node_type, graph_type, 2)
    best_model_2, metrics_2 = train_binary_classification_get_run_id(dataset_2, model=model_2, parameters=parameters)
    dataset_3 = load_one_dataset(parameters, bench_folder, graph_folder, node_type, graph_type, 3)
    best_model_3, metrics_3 = train_multi_classification_get_run_id(dataset_3, model=model_3, parameters=parameters)

    metrics = {**metrics_2, **metrics_3}
    mlflow.log_metrics(metrics)
    print("-" * 10, "train finished", "-" * 10)

def train_binary_classification_get_run_id(dataset, model, parameters: Dict):
    epoch=1
    model_type = "binary_classification"
    train_dataloader, valid_dataloader, optimizer, loss_function, best_model, best_valid_loss, best_valid_accuracy, epoch_info_log, check_point_model_path = initialize_train_objects(
        dataset, parameters, model, model_type=model_type)
    # Training Phase
    model, avg_train_loss = training_phase(model, train_dataloader, loss_function, optimizer)
    # Validation Phase
    model, avg_valid_loss, valid_accuracy = validation_phase(model, valid_dataloader, loss_function, model_type)
    # Save based on specified criterion
    best_model, best_valid_loss, best_valid_accuracy, epoch_info_log = log_and_save_best_model(parameters, epoch,
                                                                                               best_model, model,
                                                                                               "binary", 2,
                                                                                               avg_train_loss,
                                                                                               avg_valid_loss,
                                                                                               valid_accuracy,
                                                                                               best_valid_loss,
                                                                                               best_valid_accuracy,
                                                                                               epoch_info_log)
    save_checkpoint(model, optimizer, epoch, best_valid_loss, best_valid_accuracy, parameters,
                    filename=check_point_model_path)
    # Return the trained model and the best metrics
    best_metrics = {"best_valid_loss_binary": best_valid_loss, "best_valid_accuracy_binary": best_valid_accuracy}
    return best_model, best_metrics
def train_multi_classification_get_run_id(dataset, model, parameters: Dict):
    epoch=1
    model_type = "multi_classification"
    train_dataloader, valid_dataloader, optimizer, loss_function, best_model, best_valid_loss, best_valid_accuracy, epoch_info_log, check_point_model_path = initialize_train_objects(
        dataset, parameters, model, model_type=model_type)

    # Training Phase
    model, avg_train_loss = training_phase(model, train_dataloader, loss_function, optimizer)

    # Validation Phase
    model, avg_valid_loss, valid_accuracy = validation_phase(model, valid_dataloader, loss_function, model_type)

    # Save based on specified criterion
    best_model, best_valid_loss, best_valid_accuracy, epoch_info_log = log_and_save_best_model(parameters, epoch,
                                                                                               best_model, model,
                                                                                               "multi_class",
                                                                                               dataset._label_size,
                                                                                               avg_train_loss,
                                                                                               avg_valid_loss,
                                                                                               valid_accuracy,
                                                                                               best_valid_loss,
                                                                                               best_valid_accuracy,
                                                                                               epoch_info_log)
    save_checkpoint(model, optimizer, epoch, best_valid_loss, best_valid_accuracy, parameters,
                    filename=check_point_model_path)
    # Return the trained model and the best metrics
    best_metrics = {"best_valid_loss_multi_class": best_valid_loss,
                    "best_valid_accuracy_multi_class": best_valid_accuracy}
    return best_model, best_metrics







if __name__ == '__main__':
    main()
