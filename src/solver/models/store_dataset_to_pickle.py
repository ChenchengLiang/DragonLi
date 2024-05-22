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
from src.solver.independent_utils import save_to_pickle, compress_to_zip, get_folders
from Dataset import WordEquationDatasetBinaryClassification, WordEquationDatasetMultiModels, \
    WordEquationDatasetMultiClassification, WordEquationDatasetMultiClassificationLazy
from src.solver.Constants import project_folder, bench_folder, recursion_limit
from src.solver.rank_task_models.Dataset import WordEquationDatasetMultiClassificationRankTask

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

    benchmark = "rank_smtlib_2023-05-05_without_woorpje_train_300_each_folder"#"choose_eq_train"
    parameters = {"node_type": 4}
    func=prepare_and_save_datasets_rank

    folder_list = [folder for folder in get_folders(bench_folder + "/" + benchmark) if
                   "divided" in folder or "valid" in folder]
    print(folder_list)
    if len(folder_list) != 0:
        for folder in folder_list:
            store_dataset_to_pickle_one_folder(args, benchmark + "/" + folder, parameters,func)
    else:
        store_dataset_to_pickle_one_folder(args, benchmark, parameters,func)


def store_dataset_to_pickle_one_folder(args, folder, parameters,func):
    parameters["folder"] = folder
    parameters["graph_type"] = args.graph_type

    func(parameters)

    print("Done")



def _get_benchmark_folder_and_graph_folder(parameters):
    benchmark_folder = os.path.join(bench_folder, parameters["folder"])
    graph_folder = os.path.join(benchmark_folder, parameters["graph_type"])
    graph_type = parameters["graph_type"]
    print("folder:", parameters["folder"])
    return benchmark_folder, graph_folder, graph_type



def prepare_and_save_datasets_rank(parameters):
    benchmark_folder, graph_folder, graph_type = _get_benchmark_folder_and_graph_folder(parameters)
    pickle_file = os.path.join(benchmark_folder, f"dataset_{graph_type}.pkl")
    dataset = WordEquationDatasetMultiClassificationRankTask(graph_folder=graph_folder)
    # Save the datasets to pickle files
    save_to_pickle(dataset, pickle_file)
    # Compress pickle files into ZIP files
    compress_to_zip(pickle_file)

def prepare_and_save_datasets_task_3(parameters):
    benchmark_folder, graph_folder, graph_type = _get_benchmark_folder_and_graph_folder(parameters)

    # Filenames for the pickle files
    pickle_file_2 = os.path.join(benchmark_folder, f"dataset_2_{graph_type}.pkl")
    pickle_file_3 = os.path.join(benchmark_folder, f"dataset_3_{graph_type}.pkl")

    # Prepare datasets
    dataset_2 = WordEquationDatasetMultiClassification(graph_folder=graph_folder, node_type=parameters["node_type"],
                                                       label_size=2)
    dataset_3 = WordEquationDatasetMultiClassification(graph_folder=graph_folder, node_type=parameters["node_type"],
                                                       label_size=3)

    # Save the datasets to pickle files
    save_to_pickle(dataset_2, pickle_file_2)
    save_to_pickle(dataset_3, pickle_file_3)

    # Compress pickle files into ZIP files
    compress_to_zip(pickle_file_2)
    compress_to_zip(pickle_file_3)




if __name__ == '__main__':
    main()
