import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

import sys
from src.solver.Constants import project_folder
import argparse

sys.path.append(project_folder)
from src.solver.Constants import bench_folder, recursion_limit

from src.solver.algorithms.split_equations_extract_data import SplitEquationsExtractData
from src.train_data_collection.utils import generate_train_data_in_one_folder


def main():
    # parse argument
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('benchmark', type=str, default=None,
                            help='benchmark name')
    arg_parser.add_argument('folder', type=str, default=None,
                            help='divided_i or valid_data folder')
    args = arg_parser.parse_args()
    # Accessing the arguments
    benchmark = args.benchmark
    folder = args.folder

    # algorithm = ElimilateVariablesRecursive
    # algorithm_parameters = {"branch_method": "extract_branching_data_task_3", "extract_algorithm": "fixed",
    #                         "termination_condition": "termination_condition_0"}  # extract_branching_data_task_2

    algorithm = SplitEquationsExtractData
    algorithm_parameters = {"branch_method": "fixed", "order_equations_method": "category",
                            "termination_condition": "termination_condition_4", "task": "dynamic_embedding"}

    sys.setrecursionlimit(recursion_limit)
    benchmark_path = bench_folder + "/" + benchmark

    generate_train_data_in_one_folder(benchmark_path + "/" + folder, algorithm, algorithm_parameters)


if __name__ == '__main__':
    main()
