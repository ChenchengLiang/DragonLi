import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.Constants import project_folder, rank_task_label_size_map, rank_task_node_type_map
from src.solver.independent_utils import write_configurations_to_json_file


def main():
    num_epochs = 100
    train_step = 10
    task = "rank_task"  # "task_3"
    rank_task = 1
    multi_classification_pooling_type = "concat"  # conat, mean
    learning_rate = 0.001
    train_batch_size=1000
    valid_batch_size_factor = 1
    valid_batch_size = 2000
    node_type = rank_task_node_type_map[rank_task]
    label_size = rank_task_label_size_map[rank_task]
    configurations = []



    #for benchmark in ["01_track_multi_word_equations_eq_2_50_generated_train_core+UNSAT+bootstrapping_core+UNSAT_rank_task_0"]:
    for benchmark in ["01_track_multi_word_equations_eq_2_50_generated_train_core+UNSAT+bootstrapping_core+UNSAT_rank_task_1"]:
    #for benchmark in ["01_track_multi_word_equations_eq_2_50_generated_train_core+UNSAT+bootstrapping_core+UNSAT_rank_task_2"]:
    #for benchmark in ["unsatcores_01_track_multi_word_equations_eq_5_20_generated_train_1_20000_one_core_rank_task_0"]:
    #for benchmark in ["unsatcores_01_track_multi_word_equations_eq_5_20_generated_train_1_20000_one_core_rank_task_1"]:
    #for benchmark in ["unsatcores_01_track_multi_word_equations_eq_5_20_generated_train_1_20000_one_core_rank_task_2"]:
    #for benchmark in ["unsatcores_01_track_multi_word_equations_eq_2_50_generated_train_1_20000_one_core_rank_task_0"]:
    #for benchmark in ["unsatcores_01_track_multi_word_equations_eq_2_50_generated_train_1_20000_one_core_rank_task_1"]:
    #for benchmark in ["unsatcores_01_track_multi_word_equations_eq_2_50_generated_train_1_20000_one_core_rank_task_2"]:
    #for benchmark in ["unsatcore_01_track_multi_word_equations_generated_train_1_40000_one_core_rank_task_0"]:
    #for benchmark in ["unsatcore_01_track_multi_word_equations_generated_train_1_40000_one_core_rank_task_1"]:
    #for benchmark in ["unsatcore_01_track_multi_word_equations_generated_train_1_40000_one_core_rank_task_2"]:
    #for benchmark in ["unsatcore_01_track_multi_word_equations_generated_train_1_40000_one_core+proof_tree_new_graph_rank_task_0"]:
    #for benchmark in ["unsatcore_01_track_multi_word_equations_generated_train_1_40000_one_core+proof_tree_new_graph_rank_task_1"]:
    #for benchmark in ["unsatcore_01_track_multi_word_equations_generated_train_1_40000_one_core+proof_tree_new_graph_rank_task_2"]:
    #for benchmark in ["01_track_multi_word_equations_eq_2_50_generated_train_1_10000_UNSAT_data_extraction-part_1_rank_task_0"]:
    #for benchmark in ["01_track_multi_word_equations_eq_2_50_generated_train_1_10000_UNSAT_data_extraction-part_1_rank_task_1"]:
    #for benchmark in ["01_track_multi_word_equations_eq_2_50_generated_train_1_10000_UNSAT_data_extraction-part_1_rank_task_2"]:
        #for graph_type in ["graph_1"]:
        for graph_type in ["graph_1","graph_2","graph_3","graph_4","graph_5"]:
            for classifier_pool_type in ["concat"]:#["concat","max","min"]:
                for classifier_num_filter in [1]:#[1, 2, 4]:
                    for gnn_num_filters in [1]:#[1,2,4]:
                        for gnn_pool_type in ["concat"]:
                            for gnn_layer_num in [2]:
                                for ffnn_layer_num in [2]:
                                    for hidden_dim in [128]:  # [64, 128]:
                                        for dropout_rate in [0.2,0.5]:
                                            for batch_size in [train_batch_size]:
                                                for model_type in [
                                                    "GCNSplit"]:  # ["GCN","GIN","GCNwithGAP","MultiGNNs"]:  # ["GCN", "GAT", "GIN","GCNwithGAP","MultiGNNs"]
                                                    for share_gnn in [False]:
                                                        if model_type == "GAT":
                                                            for num_heads in [1]:
                                                                configurations.append({
                                                                    "benchmark": benchmark, "graph_type": graph_type,
                                                                    "model_type": model_type, "task": task, "rank_task": rank_task,
                                                                    "num_epochs": num_epochs, "learning_rate": learning_rate,
                                                                    "save_criterion": "valid_accuracy", "batch_size": batch_size,
                                                                    "gnn_hidden_dim": hidden_dim,
                                                                    "gnn_layer_num": gnn_layer_num, "num_heads": num_heads,
                                                                    "gnn_dropout_rate": dropout_rate,
                                                                    "ffnn_hidden_dim": hidden_dim, "ffnn_layer_num": ffnn_layer_num,
                                                                    "label_size": label_size,
                                                                    "ffnn_dropout_rate": dropout_rate, "node_type": node_type,
                                                                    "train_step": train_step,
                                                                    "share_gnn": share_gnn, "pooling_type": multi_classification_pooling_type,
                                                                    "valid_batch_size_factor": valid_batch_size_factor,
                                                                    "valid_batch_size": valid_batch_size,
                                                                    "classifier_pool_type":classifier_pool_type,
                                                                    "classifier_num_filter":classifier_num_filter,"gnn_num_filters":gnn_num_filters,"gnn_pool_type":gnn_pool_type
                                                                })
                                                        else:
                                                            configurations.append({
                                                                "benchmark": benchmark, "graph_type": graph_type,
                                                                "model_type": model_type, "task": task, "rank_task": rank_task,
                                                                "num_epochs": num_epochs,
                                                                "learning_rate": learning_rate,
                                                                "save_criterion": "valid_accuracy", "batch_size": batch_size,
                                                                "gnn_hidden_dim": hidden_dim,
                                                                "gnn_layer_num": gnn_layer_num, "num_heads": 0,
                                                                "gnn_dropout_rate": dropout_rate,
                                                                "ffnn_hidden_dim": hidden_dim, "ffnn_layer_num": ffnn_layer_num,
                                                                "label_size": label_size,
                                                                "ffnn_dropout_rate": dropout_rate, "node_type": node_type,
                                                                "train_step": train_step,
                                                                "share_gnn": share_gnn, "pooling_type": multi_classification_pooling_type,
                                                                "valid_batch_size_factor": valid_batch_size_factor,
                                                                "valid_batch_size": valid_batch_size,
                                                                "classifier_pool_type": classifier_pool_type,
                                                                "classifier_num_filter": classifier_num_filter,"gnn_num_filters":gnn_num_filters,"gnn_pool_type":gnn_pool_type
                                                            })

    # Writing the dictionary to a JSON file
    configuration_folder = project_folder + "/Models/configurations"
    write_configurations_to_json_file(configuration_folder=configuration_folder, configurations=configurations)
    print("Done")


if __name__ == '__main__':
    main()
