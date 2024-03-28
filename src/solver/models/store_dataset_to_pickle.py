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
import json
import datetime
from src.solver.independent_utils import save_to_pickle, compress_to_zip,get_folders
from Dataset import WordEquationDatasetBinaryClassification, WordEquationDatasetMultiModels, \
    WordEquationDatasetMultiClassification,WordEquationDatasetMultiClassificationLazy
from src.solver.Constants import project_folder, bench_folder,recursion_limit


def main():
    # draw graphs from train folder
    sys.setrecursionlimit(recursion_limit)

    # read graph type from command line
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('graph_type', type=str, help='graph_type')
    args = arg_parser.parse_args()

    # draw graphs for all folders
    benchmark = "debug-train"
    parameters = {"node_type":4}

    folder_list = [folder for folder in get_folders(bench_folder + "/" + benchmark) if
                   "divided" in folder or "valid" in folder]
    print(folder_list)
    if len(folder_list) != 0:
        for folder in folder_list:
            store_dataset_to_pickle_one_folder(args, benchmark + "/" + folder,parameters)
    else:
        store_dataset_to_pickle_one_folder(args, benchmark,parameters)

def store_dataset_to_pickle_one_folder(args,folder,parameters):
    parameters["folder"] = folder
    parameters["graph_type"] = args.graph_type

    prepare_and_save_datasets_task_3(parameters)

    print("Done")


def prepare_and_save_datasets_task_3(parameters):
    benchmark_folder = os.path.join(bench_folder, parameters["folder"])
    graph_folder = os.path.join(benchmark_folder, parameters["graph_type"])
    graph_type = parameters["graph_type"]
    print("folder:", parameters["folder"])

    # Filenames for the pickle files
    pickle_file_2 = os.path.join(benchmark_folder, f"dataset_2_{graph_type}.pkl")
    pickle_file_3 = os.path.join(benchmark_folder, f"dataset_3_{graph_type}.pkl")

    # Prepare datasets
    dataset_2 = WordEquationDatasetMultiClassification(graph_folder=graph_folder, node_type=parameters["node_type"],
                                                           label_size=2)
    dataset_3 = WordEquationDatasetMultiClassification(graph_folder=graph_folder, node_type=parameters["node_type"],
                                                           label_size=3)
    # dataset_2 = WordEquationDatasetMultiClassificationLazy(graph_folder=graph_folder, node_type=parameters["node_type"],
    #                                                    label_size=2)
    # dataset_3 = WordEquationDatasetMultiClassificationLazy(graph_folder=graph_folder, node_type=parameters["node_type"],
    #                                                    label_size=3)

    # Save the datasets to pickle files
    save_to_pickle(dataset_2, pickle_file_2)
    save_to_pickle(dataset_3, pickle_file_3)

    # Compress pickle files into ZIP files
    compress_to_zip(pickle_file_2)
    compress_to_zip(pickle_file_3)


if __name__ == '__main__':
    main()
