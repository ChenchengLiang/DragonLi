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
from src.solver.independent_utils import color_print
from src.solver.Constants import project_folder, bench_folder
import signal
from src.solver.rank_task_models.Dataset import WordEquationDatasetMultiClassificationRankTask


def main():
    benchmark_folder = "choose_eq_train"
    data_folder = "divided_1"
    graph_type = "graph_1"
    graph_folder = os.path.join(bench_folder, benchmark_folder, data_folder, graph_type)

    dataset = WordEquationDatasetMultiClassificationRankTask(graph_folder=graph_folder)
    dataset.statistics()


if __name__ == '__main__':
    main()
