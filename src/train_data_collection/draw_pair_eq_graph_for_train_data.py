import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)


from src.solver.independent_utils import strip_file_name_suffix,dump_to_json_with_format
from src.solver.Parser import Parser,EqParser
import glob
from src.solver.utils import graph_func_map
from typing import List, Tuple, Dict, Union, Optional, Callable
from src.solver.Constants import project_folder,bench_folder
from src.train_data_collection.utils import output_one_eq_graph,output_pair_eq_graphs
import shutil
import argparse
def main():
    sys.setrecursionlimit(1000000)


    # draw graphs from train folder

    # arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    # arg_parser.add_argument('graph_type', type=str, help='graph_type')
    # args = arg_parser.parse_args()

    benchmark = "test_track"
    train_eq_folder = bench_folder + "/" + benchmark + "/train"
    for graph_type in ["graph_1"]:
        # prepare folder
        graph_folder = bench_folder + "/" + benchmark + "/" + graph_type

        if os.path.exists(graph_folder):
            shutil.rmtree(graph_folder)
        print(f"- copy train to {graph_type} -")
        shutil.copytree(train_eq_folder, graph_folder)

        # draw one type graphs
        print(f"- draw {graph_type} -")
        output_pair_eq_graphs(graph_folder=graph_folder, graph_func=graph_func_map[graph_type], visualize=False)











if __name__ == '__main__':
    main()