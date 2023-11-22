from src.solver.independent_utils import strip_file_name_suffix, dump_to_json_with_format
from src.solver.Parser import Parser, EqParser
import glob
from src.solver.utils import graph_func_map
from typing import List, Tuple, Dict, Union, Optional, Callable
import os
import shutil
import json
from src.solver.Constants import satisfiability_to_int_label
from src.solver.DataTypes import Equation, Edge
from src.solver.algorithms.utils import merge_graphs,graph_to_gnn_format
from src.solver.visualize_util import draw_graph


def dvivde_track_for_cluster(benchmark, chunk_size=50):
    folder = benchmark + "/ALL"
    chunk_size = chunk_size

    folder_counter = 0
    all_folder = folder + "/ALL"
    if not os.path.exists(all_folder):
        os.mkdir(all_folder)

    for file in glob.glob(folder + "/*"):
        shutil.move(file, all_folder)

    for i, eq_file in enumerate(glob.glob(all_folder + "/*.eq")):
        if i % chunk_size == 0:
            folder_counter += 1
            divided_folder_name = folder + "/divided_" + str(folder_counter)
            os.mkdir(divided_folder_name)
        file_name = strip_file_name_suffix(eq_file)
        for f in glob.glob(file_name + ".eq") + glob.glob(file_name + ".answer"):
            shutil.copy(f, divided_folder_name)


def _read_label_and_eqs(f,graph_folder,parser,graph_func):
    with open(f, 'r') as json_file:
        json_dict = json.loads(json_file.read())
    file_name = f.replace(".label.json", "")

    eq_file = file_name + ".eq"
    split_eq_file_list = [graph_folder + "/" + x for x in json_dict["middle_branch_eq_file_name_list"]]

    eq: Equation = parser.parse(eq_file)["equation_list"][0]
    eq_nodes, eq_edges = graph_func(eq.left_terms, eq.right_terms)
    split_eq_list: List[Equation] = [parser.parse(split_eq_file)["equation_list"][0] for split_eq_file in
                                     split_eq_file_list]
    return eq_nodes,eq_edges,split_eq_list, split_eq_file_list, json_dict["label_list"],json_dict["satisfiability_list"]
def output_split_eq_graphs(graph_folder: str, graph_func: Callable, visualize: bool = False):
    parser_type = EqParser()
    parser = Parser(parser_type)
    for f in glob.glob(graph_folder + "/*.label.json"):
        eq_nodes, eq_edges, split_eq_list, split_eq_file_list, label_list,satisfiability_list = _read_label_and_eqs(f, graph_folder, parser,
                                                                                            graph_func)
        multi_graph_dict={}
        for i,(split_eq, split_file, split_label,split_satisfiability) in enumerate(zip(split_eq_list, split_eq_file_list, label_list,satisfiability_list)):
            split_eq_odes, split_eq_edges = graph_func(split_eq.left_terms, split_eq.right_terms)
            if visualize == True:
                merged_nodes, merged_edges = merge_graphs(eq_nodes, eq_edges, split_eq_odes, split_eq_edges)
                draw_graph(nodes=merged_nodes, edges=merged_edges, filename=split_file)
            graph_dict = graph_to_gnn_format(split_eq_odes, split_eq_edges, label=split_label,satisfiability=split_satisfiability)
            multi_graph_dict[i]=graph_dict

        # Dumping the dictionary to a JSON file
        json_file = f.replace(".label.json",".graph.json")
        dump_to_json_with_format(multi_graph_dict, json_file)


def output_pair_eq_graphs(graph_folder: str, graph_func: Callable, visualize: bool = False):
    parser_type = EqParser()
    parser = Parser(parser_type)

    for f in glob.glob(graph_folder + "/*.label.json"):

        eq_nodes,eq_edges,split_eq_list, split_eq_file_list, label_list,satisfiability_list=_read_label_and_eqs(f, graph_folder, parser, graph_func)

        #print("eq", eq.eq_str)
        for split_eq, split_file, split_label,split_satisfiability in zip(split_eq_list, split_eq_file_list, label_list,satisfiability_list):
            #print("split_eq", split_eq.eq_str)
            split_eq_odes, split_eq_edges = graph_func(split_eq.left_terms, split_eq.right_terms)
            merged_nodes, merged_edges = merge_graphs(eq_nodes, eq_edges, split_eq_odes, split_eq_edges)
            if visualize == True:
                draw_graph(nodes=merged_nodes, edges=merged_edges, filename=split_file)

            graph_dict = graph_to_gnn_format(merged_nodes, merged_edges, label=split_label,satisfiability=split_satisfiability)
            # Dumping the dictionary to a JSON file
            json_file = strip_file_name_suffix(split_file) + ".graph.json"
            dump_to_json_with_format(graph_dict, json_file)




def output_eq_graphs(graph_folder: str, graph_func: Callable, visualize: bool = False):
    parser_type = EqParser()
    parser = Parser(parser_type)
    eq_file_list = glob.glob(graph_folder + "/*.eq")
    for file_path in eq_file_list:

        parsed_content = parser.parse(file_path)
        # print("parsed_content:", parsed_content)

        answer_file = strip_file_name_suffix(file_path) + ".answer"
        with open(answer_file, 'r') as file:
            answer = file.read()

        for eq in parsed_content["equation_list"]:
            if visualize == True:
                # visualize
                eq.visualize_graph(file_path, graph_func)
            # get gnn format
            nodes, edges = graph_func(eq.left_terms, eq.right_terms)
            satisfiability = answer
            graph_dict = graph_to_gnn_format(nodes, edges, label=satisfiability_to_int_label[satisfiability],satisfiability=satisfiability)
            # print(graph_dict)
            # Dumping the dictionary to a JSON file
            json_file = strip_file_name_suffix(file_path) + ".graph.json"
            dump_to_json_with_format(graph_dict, json_file)
