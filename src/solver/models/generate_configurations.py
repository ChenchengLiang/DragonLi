
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
def main():
    configurations=[]
    for benchmark in ["example_train"]:
        for graph_type in ["graph_1","graph_2"]:
            for model_type in ["GCN", "GAT", "GIN"]:  # ["GCN","GAT","GIN"]
                configurations.append({
                    "benchmark":benchmark,"graph_type": graph_type, "model_type": model_type, "num_epochs": 50, "learning_rate": 0.001,
                    "save_criterion": "valid_accuracy", "batch_size": 20, "gnn_hidden_dim": 32,
                    "gnn_layer_num": 2, "num_heads": 2, "ffnn_hidden_dim": 32, "ffnn_layer_num": 2
                })

    # Writing the dictionary to a JSON file
    configuration_folder=project_folder+"/models/configurations"
    if os.path.exists(configuration_folder) == False:
        os.mkdir(configuration_folder)
    else:
        shutil.rmtree(configuration_folder)
        os.mkdir(configuration_folder)

    for i,config in enumerate(configurations):

        file_name= configuration_folder+"/config_"+str(i)+".json"
        with open(file_name, 'w') as f:
            json.dump(config, f, indent=4)


if __name__ == '__main__':
    main()