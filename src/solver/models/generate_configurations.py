
import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

from src.solver.Constants import project_folder
import json
import shutil
from src.solver.independent_utils import write_configurations_to_json_file
def main():
    configurations=[]
    for benchmark in ["01_track_generated_train_data_sat_from_solver"]:
        for graph_type in ["graph_1", "graph_2"]:
            for model_type in ["GCN", "GAT", "GIN"]:
                configurations.append({
                    "benchmark":benchmark,"graph_type": graph_type, "model_type": model_type, "num_epochs": 300, "learning_rate": 0.001,
                    "save_criterion": "valid_accuracy", "batch_size": 20, "gnn_hidden_dim": 32,
                    "gnn_layer_num": 2, "num_heads": 2, "ffnn_hidden_dim": 32, "ffnn_layer_num": 2
                })

    # Writing the dictionary to a JSON file
    configuration_folder=project_folder+"/models/configurations"
    write_configurations_to_json_file(configuration_folder=configuration_folder,configurations=configurations)


if __name__ == '__main__':
    main()