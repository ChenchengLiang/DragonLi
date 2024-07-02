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
from src.solver.Constants import bench_folder, recursion_limit
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

    benchmark = "choose_eq_train_rank_2"#"choose_eq_train"
    rank_task=2
    if rank_task == 0:
        parameters = {"node_type": 5,"rank_task":0,"label_size": 2}
    elif rank_task ==1:
        parameters = {"node_type": 5,"rank_task":1,"label_size": 2}
    elif rank_task ==2:
        parameters = {"node_type": 6,"rank_task":2,"label_size": 100}

    func= prepare_and_save_datasets_rank

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
