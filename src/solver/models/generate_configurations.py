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


def main():
    num_epochs=300
    task="task_3"
    configurations = []
    for benchmark in ["03_track_generated_train_1_20000_task_3"]:
        for graph_type in ["graph_1", "graph_2", "graph_3", "graph_4", "graph_5"]:
            for gnn_layer_num in [2,8]:
                for ffnn_layer_num in [2,8]:
                    for hidden_dim in [128]:
                        for dropout_rate in [0.5]:
                            for model_type in ["GCNSplit","GINSplit"]:#["GCN","GIN","GCNwithGAP","MultiGNNs"]:  # ["GCN", "GAT", "GIN","GCNwithGAP","MultiGNNs"]
                                if model_type == "GAT":
                                    for num_heads in [1]:
                                        configurations.append({
                                            "benchmark": benchmark, "graph_type": graph_type, "model_type": model_type,"task":task,
                                            "num_epochs": num_epochs, "learning_rate": 0.001,
                                            "save_criterion": "valid_accuracy", "batch_size": 1000, "gnn_hidden_dim": hidden_dim,
                                            "gnn_layer_num": gnn_layer_num, "num_heads": num_heads, "gnn_dropout_rate": dropout_rate,
                                            "ffnn_hidden_dim": hidden_dim, "ffnn_layer_num": ffnn_layer_num, "ffnn_dropout_rate": dropout_rate
                                        })
                                else:
                                    configurations.append({
                                        "benchmark": benchmark, "graph_type": graph_type, "model_type": model_type,"task":task,
                                        "num_epochs": num_epochs,
                                        "learning_rate": 0.001,
                                        "save_criterion": "valid_accuracy", "batch_size": 1000, "gnn_hidden_dim": hidden_dim,
                                        "gnn_layer_num": gnn_layer_num, "num_heads": 0, "gnn_dropout_rate": dropout_rate,
                                        "ffnn_hidden_dim": hidden_dim, "ffnn_layer_num": ffnn_layer_num, "ffnn_dropout_rate": dropout_rate
                                    })

    # Writing the dictionary to a JSON file
    configuration_folder = project_folder + "/Models/configurations"
    write_configurations_to_json_file(configuration_folder=configuration_folder, configurations=configurations)


if __name__ == '__main__':
    main()
