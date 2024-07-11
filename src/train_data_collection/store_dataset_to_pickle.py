import configparser
import os
import sys

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

os.environ["DGLBACKEND"] = "pytorch"
import argparse
from src.solver.independent_utils import get_folders
from src.solver.Constants import bench_folder, recursion_limit, rank_task_label_size_map, rank_task_node_type_map
from src.train_data_collection.utils import store_dataset_to_pickle_one_folder, prepare_and_save_datasets_rank


def main():
    # draw graphs from train folder
    sys.setrecursionlimit(recursion_limit)

    # read graph type from command line
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('graph_type', type=str, help='graph_type')
    args = arg_parser.parse_args()

    # draw graphs for all folders
    # benchmark = "01_track_train_task_3_1_2000"
    # parameters = {"node_type": 4}
    # func=prepare_and_save_datasets_task_3
    graph_type = args.graph_type

    rank_task = 1
    # benchmark = "choose_eq_train_rank_2"
    #benchmark = "rank_01_track_multi_word_equations_generated_train_1_40000_new_divided_300_chunk_size_multiple_path_rank_task_2"
    benchmark = "choose_eq_train_rank_1"

    parameters = {"rank_task": rank_task}
    parameters["node_type"] = rank_task_node_type_map[rank_task]
    parameters["label_size"] = rank_task_label_size_map[rank_task]

    func = prepare_and_save_datasets_rank

    folder_list = [folder for folder in get_folders(bench_folder + "/" + benchmark) if
                   "divided" in folder or "valid" in folder]
    print(folder_list)
    if len(folder_list) != 0:
        for folder in folder_list:
            store_dataset_to_pickle_one_folder(graph_type, benchmark + "/" + folder, parameters, func)
    else:
        store_dataset_to_pickle_one_folder(graph_type, benchmark, parameters, func)


if __name__ == '__main__':
    main()
