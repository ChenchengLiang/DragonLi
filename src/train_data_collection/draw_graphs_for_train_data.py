import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.independent_utils import strip_file_name_suffix, dump_to_json_with_format,zip_folder
from src.solver.Parser import Parser, EqParser
import glob
from src.solver.utils import graph_func_map
from typing import List, Tuple, Dict, Union, Optional, Callable
from src.solver.Constants import project_folder, bench_folder
from src.train_data_collection.utils import output_eq_graphs, output_pair_eq_graphs, output_split_eq_graphs
import shutil
import argparse


def main():
    # draw graphs from train folder
    sys.setrecursionlimit(1000000)

    benchmark = "01_track_multi_word_equations_generated_train_1_40000_new_SAT_divided_1"
    task = "task_3"

    # read graph type from command line
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('graph_type', type=str, help='graph_type')
    args = arg_parser.parse_args()

    sys.setrecursionlimit(10000000)

    if task == "task_1":
        draw_func = output_eq_graphs  # task 1
    elif task == "task_2":
        draw_func = output_pair_eq_graphs  # task 2
    elif task == "task_3":
        draw_func = output_split_eq_graphs  # task 3

    train_eq_folder = bench_folder + "/" + benchmark + "/train"
    train_zip_file=train_eq_folder+".zip"
    for graph_type in [args.graph_type]:
        # prepare folder
        graph_folder = bench_folder + "/" + benchmark + "/" + graph_type

        if os.path.exists(graph_folder):
            shutil.rmtree(graph_folder)
            os.mkdir(graph_folder)
        else:
            os.mkdir(graph_folder)

        # draw one type graphs
        print(f"- draw {graph_type} -")
        draw_func(zip_file=train_zip_file,graph_folder=graph_folder, graph_func=graph_func_map[graph_type], visualize=False)

        # compress
        zip_folder(folder_path=graph_folder, output_zip_file=graph_folder + ".zip")
        shutil.rmtree(graph_folder)

    print("done")


if __name__ == '__main__':
    main()
