import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.Constants import project_folder
import json
import shutil
from src.solver.independent_utils import write_configurations_to_json_file


def main():
    configurations = []
    for benchmark in ["01_track_generated_train_data_sat_with_some_leafs"]:
        for graph_type in ["graph_1"]:#["graph_1", "graph_2", "graph_3", "graph_4", "graph_5"]:
            for gnn_layer_num in [2, 4, 8]:
                for hidden_dim in [32, 64]:
                    for model_type in ["GCN", "GIN"]:  # ["GCN", "GAT", "GIN"]
                        if model_type == "GAT":
                            for num_heads in [1]:
                                configurations.append({
                                    "benchmark": benchmark, "graph_type": graph_type, "model_type": model_type,
                                    "num_epochs": 300, "learning_rate": 0.001,
                                    "save_criterion": "valid_accuracy", "batch_size": 100, "gnn_hidden_dim": hidden_dim,
                                    "gnn_layer_num": gnn_layer_num, "num_heads": num_heads, "gnn_dropout_rate": 0.5,
                                    "ffnn_hidden_dim": hidden_dim, "ffnn_layer_num": 2, "ffnn_dropout_rate": 0.5
                                })
                        else:
                            configurations.append({
                                "benchmark": benchmark, "graph_type": graph_type, "model_type": model_type,
                                "num_epochs": 300,
                                "learning_rate": 0.001,
                                "save_criterion": "valid_accuracy", "batch_size": 100, "gnn_hidden_dim": hidden_dim,
                                "gnn_layer_num": gnn_layer_num, "num_heads": 0, "gnn_dropout_rate": 0.5,
                                "ffnn_hidden_dim": hidden_dim, "ffnn_layer_num": 2, "ffnn_dropout_rate": 0.5
                            })

    # Writing the dictionary to a JSON file
    configuration_folder = project_folder + "/models/configurations"
    write_configurations_to_json_file(configuration_folder=configuration_folder, configurations=configurations)


if __name__ == '__main__':
    main()
