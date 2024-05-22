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
    arg_parser.add_argument('benchmark', type=str, help='benchmark')
    arg_parser.add_argument('folder', type=str, help='folder')
    args = arg_parser.parse_args()

    graph_type = args.graph_type
    benchmark = args.benchmark
    folder = args.folder

    # parameters = {"node_type": 4}
    # func=prepare_and_save_datasets_task_3


    parameters = {"node_type": 4}
    func= prepare_and_save_datasets_rank

    store_dataset_to_pickle_one_folder(graph_type, benchmark + "/" + folder, parameters, func)


if __name__ == '__main__':
    main()
