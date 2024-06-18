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
from src.solver.models.train_util import (initialize_model_structure,training_phase,validation_phase,
                                          initialize_train_objects,log_and_save_best_model,save_checkpoint)
from typing import Dict
import mlflow
import datetime
import subprocess
import torch
import signal

def main():
    num_epochs=200
    task="rank_task_1"#"task_3"
    node_type=5
    learning_rate=0.001
    train_step=10
    label_size=2
    configurations = []
    for benchmark in ["choose_eq_train"]:
    #for benchmark in  ["rank_01_track_multi_word_equations_generated_train_1_40000_new_divided_300_chunk_size_multiple_path"]:
        for graph_type in ["graph_1"]:#["graph_1","graph_2","graph_3","graph_4","graph_5"]:
            for gnn_layer_num in [2]:#[2,8]:
                for ffnn_layer_num in [2]:
                    for hidden_dim in [128]:#[128,256]:
                        for dropout_rate in [0.2]:
                            for batch_size in [10000]:
                                for model_type in ["GCNSplit"]:#["GCN","GIN","GCNwithGAP","MultiGNNs"]:  # ["GCN", "GAT", "GIN","GCNwithGAP","MultiGNNs"]
                                    for share_gnn in [False]:
                                        if model_type == "GAT":
                                            for num_heads in [1]:
                                                configurations.append({
                                                    "benchmark": benchmark, "graph_type": graph_type, "model_type": model_type,"task":task,
                                                    "num_epochs": num_epochs, "learning_rate": learning_rate,
                                                    "save_criterion": "valid_accuracy", "batch_size": batch_size, "gnn_hidden_dim": hidden_dim,
                                                    "gnn_layer_num": gnn_layer_num, "num_heads": num_heads, "gnn_dropout_rate": dropout_rate,
                                                    "ffnn_hidden_dim": hidden_dim, "ffnn_layer_num": ffnn_layer_num,"label_size":label_size,
                                                    "ffnn_dropout_rate": dropout_rate,"node_type":node_type,"train_step":train_step,"share_gnn":share_gnn
                                                })
                                        else:
                                            configurations.append({
                                                "benchmark": benchmark, "graph_type": graph_type, "model_type": model_type,"task":task,
                                                "num_epochs": num_epochs,
                                                "learning_rate": learning_rate,
                                                "save_criterion": "valid_accuracy", "batch_size": batch_size, "gnn_hidden_dim": hidden_dim,
                                                "gnn_layer_num": gnn_layer_num, "num_heads": 0, "gnn_dropout_rate": dropout_rate,
                                                "ffnn_hidden_dim": hidden_dim, "ffnn_layer_num": ffnn_layer_num,"label_size":label_size,
                                                "ffnn_dropout_rate": dropout_rate,"node_type":node_type,"train_step":train_step,"share_gnn":share_gnn
                                            })

    # Writing the dictionary to a JSON file
    configuration_folder = project_folder + "/Models/configurations"
    write_configurations_to_json_file(configuration_folder=configuration_folder, configurations=configurations)
    print("Done")



if __name__ == '__main__':
    main()
