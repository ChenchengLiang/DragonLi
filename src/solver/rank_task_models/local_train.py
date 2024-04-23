import configparser
import os
import sys

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import mlflow
import argparse
import json
import datetime
import subprocess
from src.solver.independent_utils import color_print, load_from_pickle_within_zip
from src.solver.Constants import project_folder, bench_folder
import signal
from src.solver.rank_task_models.Dataset import WordEquationDatasetMultiClassificationRankTask


def main():

    parameters={}

    
    parameters["benchmark_folder"] = "choose_eq_train"
    parameters["data_folder"] = "divided_1"
    parameters["graph_type"] = "graph_1"
    read_dataset_from_zip(parameters)



def read_dataset_from_zip(parameters):

    pickle_folder = os.path.join(bench_folder,parameters["benchmark_folder"], parameters["data_folder"])
    graph_type = parameters["graph_type"]

    # Filenames for the ZIP files
    zip_file = os.path.join(pickle_folder, f"dataset_{graph_type}.pkl.zip")
    if os.path.exists(zip_file):
        print("-" * 10, "load dataset from zipped pickle:", parameters["data_folder"], "-" * 10)
        # Names of the pickle files inside ZIP archives
        pickle_name = f"dataset_{graph_type}.pkl"
        # Load the datasets directly from ZIP files
        dataset = load_from_pickle_within_zip(zip_file, pickle_name)
        dataset.statistics()
    else:
        dataset=None

    #mlflow.log_text(dataset_statistics, artifact_file=f"{data_folder}_dataset_statistics.txt")
    return dataset




if __name__ == '__main__':
    main()
