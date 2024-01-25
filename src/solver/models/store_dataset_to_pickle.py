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
from src.solver.independent_utils import save_to_pickle, compress_to_zip
from Dataset import WordEquationDatasetBinaryClassification, WordEquationDatasetMultiModels, \
    WordEquationDatasetMultiClassification,WordEquationDatasetMultiClassificationLazy
from src.solver.Constants import project_folder, bench_folder


def main():
    # draw graphs from train folder
    sys.setrecursionlimit(1000000)

    benchmark = "01_track_multi_word_equations_generated_train_1_40000_new_small_test/divided_3"#"01_track_multi_word_equations_generated_train_1_40000"

    # read graph type from command line
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('graph_type', type=str, help='graph_type')
    args = arg_parser.parse_args()

    parameters={}
    parameters["benchmark"] = benchmark
    parameters["graph_type"] = args.graph_type



    prepare_and_save_datasets_task_3(parameters)

    print("Done")


def prepare_and_save_datasets_task_3(parameters):
    parameters["node_type"] = 4
    benchmark_folder = os.path.join(bench_folder, parameters["benchmark"])
    graph_folder = os.path.join(benchmark_folder, parameters["graph_type"])
    graph_type = parameters["graph_type"]
    print("benchmark:", parameters["benchmark"])

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
