import os.path
from src.solver.independent_utils import strip_file_name_suffix,dump_to_json_with_format
from src.solver.Parser import Parser,EqParser
import json
import glob
from src.solver.DataTypes import Equation
from typing import List, Tuple, Dict, Union, Optional, Callable
from src.solver.Constants import project_folder,bench_folder
def main():

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

    file_list = glob.glob(bench_folder +"/01_track_generated_train_data/SAT_from_solver/graph_1/*.eq")
    for file_path in file_list:
        output_one_eq_graph(file_path=file_path,graph_func=Equation.get_graph_1,visualize=False)

    file_list = glob.glob(bench_folder +"/01_track_generated_train_data/SAT_from_solver/graph_2/*.eq")
    for file_path in file_list:
        output_one_eq_graph(file_path=file_path, graph_func=Equation.get_graph_2, visualize=False)





def output_one_eq_graph(file_path,graph_func:Callable,visualize:bool=False):

    parser_type = EqParser()
    parser = Parser(parser_type)
    parsed_content = parser.parse(file_path)
    #print("parsed_content:", parsed_content)

    answer_file = strip_file_name_suffix(file_path) + ".answer"
    with open(answer_file, 'r') as file:
        answer = file.read()

    for eq in parsed_content["equation_list"]:
        if visualize==True:
            # visualize
            eq.visualize_graph(file_path,graph_func)
        # get gnn format
        nodes, edges = graph_func(eq.left_terms, eq.right_terms)
        satisfiability = answer
        graph_dict = eq.graph_to_gnn_format(nodes, edges, satisfiability)
        #print(graph_dict)
        # Dumping the dictionary to a JSON file
        json_file=strip_file_name_suffix(file_path) + ".json"
        dump_to_json_with_format(graph_dict, json_file)





if __name__ == '__main__':
    main()