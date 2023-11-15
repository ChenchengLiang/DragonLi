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
from src.train_data_collection.utils import output_one_eq_graph
import shutil
def main():
    sys.setrecursionlimit(1000000)

    #visualize examples
    # file_list = glob.glob(
    #     bench_folder +"/01_track_generated_train_data/graph_1/*.eq")
    # for file_path in file_list:
    #     output_one_eq_graph(file_path=file_path, graph_func=Equation.get_graph_1, visualize=True)

    # file_list = glob.glob(
    #     bench_folder +"/01_track_generated_train_data/examples/*.eq")
    # for file_path in file_list:
    #     output_one_eq_graph(file_path=file_path, graph_func=Equation.get_graph_2, visualize=True)
    #

    #/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track_generated_train_data_sat_from_solver
    benchmark="example_graphs"

    for graph_type in ["graph_1","graph_2","graph_3","graph_4","graph_5"]:

        file_list = glob.glob(bench_folder +"/"+benchmark+"/"+graph_type+"/*.eq")
        for file_path in file_list:
            output_one_eq_graph(file_path=file_path,graph_func=graph_func_map[graph_type],visualize=False)

    # # draw graphs
    # train_eq_folder = bench_folder + "/" + benchmark + "/train"
    # for graph_type in ["graph_1", "graph_2", "graph_3", "graph_4", "graph_5"]:
    #     # prepare folder
    #     graph_folder = bench_folder + "/" + benchmark + "/" + graph_type
    #
    #     if os.path.exists(graph_folder):
    #         shutil.rmtree(graph_folder)
    #     print(f"- copy train to {graph_type} -")
    #     shutil.copytree(train_eq_folder, graph_folder)
    #
    #     # draw one type graphs
    #     print(f"- draw {graph_type} -")
    #     file_list = glob.glob(graph_folder + "/*.eq")
    #     for file_path in file_list:
    #         output_one_eq_graph(file_path=file_path, graph_func=graph_func_map[graph_type], visualize=False)











if __name__ == '__main__':
    main()