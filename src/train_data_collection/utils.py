from src.process_benchmarks.eq2smt_utils import one_eq_file_to_smt2
from src.process_benchmarks.utils import run_on_one_problem
from src.solver.algorithms.split_equation_utils import _get_global_info
from src.solver.independent_utils import strip_file_name_suffix, dump_to_json_with_format, zip_folder, save_to_pickle, \
    compress_to_zip, delete_large_file
from src.solver.Parser import Parser, EqParser
import glob

from src.solver.models.Dataset import WordEquationDatasetMultiClassification
from src.solver.rank_task_models.Dataset import WordEquationDatasetMultiClassificationRankTask1, \
    WordEquationDatasetMultiClassificationRankTask0, WordEquationDatasetMultiClassificationRankTask2
from src.solver.utils import graph_func_map
from typing import List, Callable
import os
import shutil
import json
from src.solver.Constants import satisfiability_to_int_label, UNKNOWN, SAT, UNSAT, bench_folder
from src.solver.DataTypes import Equation, Formula
from src.solver.algorithms.utils import merge_graphs, graph_to_gnn_format, concatenate_eqs
from src.solver.visualize_util import draw_graph
import zipfile
import fnmatch
from tqdm import tqdm
import sys
import time
from itertools import combinations


def clean_temp_files_while_extract_unsatcore(current_unsatcore_eq_file, unsatcore_smt2_file):
    if os.path.exists(current_unsatcore_eq_file):
        os.remove(current_unsatcore_eq_file)
    if os.path.exists(unsatcore_smt2_file):
        os.remove(unsatcore_smt2_file)
    if os.path.exists(current_unsatcore_eq_file + ".answer"):
        os.remove(current_unsatcore_eq_file + ".answer")


def get_sorted_unsatcore_list_with_fixed_eq_number(eq_list: List[Equation], eq_number_to_delete):
    combination_list = list(combinations(eq_list, eq_number_to_delete))

    unsatcore_list: List[List[Equation]] = []
    for index, eq_list_to_delete in enumerate(combination_list):
        unsatcore: List[Equation] = [eq for eq in eq_list if
                                     eq not in eq_list_to_delete]
        unsatcore_list.append(unsatcore)

    unsatcore_list_with_size = [(eq_list, Formula(eq_list).formula_size) for eq_list in unsatcore_list]
    unsarcore_list_sorted = sorted(unsatcore_list_with_size, key=lambda x: x[1])
    unsarcore_list_sorted = [x for x, s in unsarcore_list_sorted]

    return unsarcore_list_sorted


def run_one_problem_according_to_solver(solver, eq_file, parameter_list, solver_log, shell_timeout_for_one_run):
    unsatcore_smt2_file=""
    start_time = time.time()
    if "this" in solver:
        result_dict = run_on_one_problem(eq_file, parameter_list, "this",
                                         solver_log=solver_log,
                                         shell_timeout=shell_timeout_for_one_run)
    elif solver == "woorpje":
        result_dict = run_on_one_problem(eq_file, parameter_list, solver,
                                         solver_log=solver_log,
                                         shell_timeout=shell_timeout_for_one_run)
    else:
        unsatcore_smt2_file = one_eq_file_to_smt2(eq_file)
        result_dict = run_on_one_problem(unsatcore_smt2_file, parameter_list, solver,
                                         solver_log=solver_log,
                                         shell_timeout=shell_timeout_for_one_run)
    solving_time = time.time() - start_time
    return result_dict, solving_time, unsatcore_smt2_file


def solve_the_core_by_different_solver(current_unsatcore_eq_file, solver_parameter_list_map, solver_log,
                                       shell_timeout_for_one_run):
    satisfiability = "UNKNOWN"
    first_solved_solver = None
    log = ""
    for solver, parameter_list in solver_parameter_list_map.items():
        result_dict, solving_time, unsatcore_smt2_file = run_one_problem_according_to_solver(solver, current_unsatcore_eq_file,
                                                                        parameter_list, solver_log,
                                                                        shell_timeout_for_one_run)

        satisfiability = result_dict["result"]
        log_text = f"        solver:{solver}, satisfiability:{satisfiability}, solving_time:{solving_time}"
        print(log_text)
        log += log_text + "\n"
        if satisfiability == "UNSAT":
            log_text = f"        SOLVED, solver:{solver}, satisfiability:{satisfiability}"
            print(log_text)
            log += log_text + "\n"
            first_solved_solver = solver
            break

    return satisfiability, first_solved_solver, unsatcore_smt2_file, solving_time, log


def dvivde_track_for_cluster(benchmark, file_folder="ALL", chunk_size=50):
    folder = benchmark + "/" + file_folder
    chunk_size = chunk_size

    folder_counter = 0
    all_folder = folder + "/ALL"
    if not os.path.exists(all_folder):
        os.mkdir(all_folder)

    for file in glob.glob(folder + "/*"):
        shutil.move(file, all_folder)

    total_file_list = glob.glob(all_folder + "/*.eq") if len(glob.glob(all_folder + "/*.eq")) != 0 else glob.glob(
        all_folder + "/*.smt2")

    for i, eq_file in enumerate(total_file_list):
        if i % chunk_size == 0:
            folder_counter += 1
            divided_folder_name = folder + "/divided_" + str(folder_counter)
            os.mkdir(divided_folder_name)
        file_name = strip_file_name_suffix(eq_file)
        # this only copy one file but faster
        # shutil.copy(eq_file, divided_folder_name)
        for f in glob.glob(file_name + ".*"):
            shutil.copy(f, divided_folder_name)


def _read_label_and_eqs(zip, f, graph_folder, parser, graph_func):
    with zip.open(f) as json_file:
        # with open(f, 'r') as json_file:
        json_dict = json.loads(json_file.read())
    file_name = f.replace(".label.json", "")

    eq_file = file_name + ".eq"
    # split_eq_file_list = [graph_folder + "/" + x for x in json_dict["middle_branch_eq_file_name_list"]]
    split_eq_file_list = ["train/" + x for x in json_dict["middle_branch_eq_file_name_list"]]

    eq: Equation = concatenate_eqs(parser.parse(eq_file, zip)["equation_list"])
    # print("eq",len(eq.eq_str),eq.eq_str)
    eq_nodes, eq_edges = graph_func(eq.left_terms, eq.right_terms)
    split_eq_list: List[Equation] = [concatenate_eqs(parser.parse(split_eq_file, zip)["equation_list"]) for
                                     split_eq_file in
                                     split_eq_file_list]
    return eq_nodes, eq_edges, split_eq_list, split_eq_file_list, json_dict["label_list"], json_dict[
        "satisfiability_list"]


def _read_label_and_eqs_for_rank(zip, f, parser):
    with zip.open(f) as json_file:
        json_dict = json.loads(json_file.read())

    rank_eq_file_list = ["train/" + x for x in json_dict["middle_branch_eq_file_name_list"]]

    split_eq_list: List[Equation] = [parser.parse(split_eq_file, zip)["equation_list"][0] for split_eq_file in
                                     rank_eq_file_list]

    # isomorphic_differentiated_eq_list = differentiate_isomorphic_equations(split_eq_list)

    return split_eq_list, rank_eq_file_list, json_dict["label_list"], json_dict[
        "satisfiability_list"]


def output_rank_eq_graphs(zip_file: str, graph_folder: str, graph_func: Callable, visualize: bool = False):
    parser = get_parser()
    with zipfile.ZipFile(zip_file, 'r') as zip_file_content:
        for f in tqdm(zip_file_content.namelist(), desc="output_rank_eq_graphs"):  # scan all files in zip
            if fnmatch.fnmatch(f, '*.label.json'):
                rank_eq_list, rank_eq_file_list, label_list, satisfiability_list = _read_label_and_eqs_for_rank(
                    zip_file_content, f, parser)
                global_info = _get_global_info(rank_eq_list)
                # print("\nglobal_info", global_info)
                # total_variable_occurrence = sum(global_info["variable_global_occurrences"].values())
                # total_terminal_occurrence = sum(global_info["terminal_global_occurrences"].values())
                # print("total_variable_occurrence", total_variable_occurrence)
                # print("total_terminal_occurrence", total_terminal_occurrence)

                multi_graph_dict = {}
                for i, (split_eq, split_file, split_label, split_satisfiability) in enumerate(
                        zip(rank_eq_list, rank_eq_file_list, label_list, satisfiability_list)):
                    # add global information
                    split_eq_nodes, split_eq_edges = graph_func(split_eq.left_terms, split_eq.right_terms, global_info)

                    if visualize == True:
                        draw_graph(nodes=split_eq_nodes, edges=split_eq_edges,
                                   filename=graph_folder + "/" + split_file.replace("train/", ""))

                    graph_dict = graph_to_gnn_format(split_eq_nodes, split_eq_edges, label=split_label,
                                                     satisfiability=split_satisfiability)
                    # print("split_file",split_file)
                    # print("graph_dict",graph_dict["nodes"])
                    multi_graph_dict[i] = graph_dict

                json_file = graph_folder + "/" + f.replace(".label.json", ".graph.json").replace("train/", "")
                dump_to_json_with_format(multi_graph_dict, json_file)
                # Dumping the dictionary to a JSON file
                # size_limit = 1 #MB
                # if sys.getsizeof(multi_graph_dict) < size_limit * 1024 * 1024:
                #     json_file = graph_folder + "/" + f.replace(".label.json", ".graph.json").replace("train/", "")
                #     dump_to_json_with_format(multi_graph_dict, json_file)
                #     #deleted_graph_json=delete_large_file(json_file,size_limit=10)
                # if len(multi_graph_dict)<50:
                #     json_file = graph_folder + "/" + f.replace(".label.json", ".graph.json").replace("train/", "")
                #     dump_to_json_with_format(multi_graph_dict, json_file)


def output_split_eq_graphs(zip_file: str, graph_folder: str, graph_func: Callable, visualize: bool = False):
    parser = get_parser()
    with zipfile.ZipFile(zip_file, 'r') as zip_file_content:
        for f in tqdm(zip_file_content.namelist(), desc="output_split_eq_graphs"):
            # for f in glob.glob(graph_folder + "/*.label.json"):
            if fnmatch.fnmatch(f, '*.label.json'):
                eq_nodes, eq_edges, split_eq_list, split_eq_file_list, label_list, satisfiability_list = _read_label_and_eqs(
                    zip_file_content, f, graph_folder, parser,
                    graph_func)
                multi_graph_dict = {}
                # get parent eq graph
                graph_dict = graph_to_gnn_format(eq_nodes, eq_edges, label=-1,
                                                 satisfiability=UNKNOWN)
                multi_graph_dict[0] = graph_dict

                for i, (split_eq, split_file, split_label, split_satisfiability) in enumerate(
                        zip(split_eq_list, split_eq_file_list, label_list, satisfiability_list)):
                    split_eq_nodes, split_eq_edges = graph_func(split_eq.left_terms, split_eq.right_terms)

                    if visualize == True:
                        merged_nodes, merged_edges = merge_graphs(eq_nodes, eq_edges, split_eq_nodes, split_eq_edges)
                        draw_graph(nodes=merged_nodes, edges=merged_edges,
                                   filename=graph_folder + "/" + split_file.replace("train/", ""))

                    graph_dict = graph_to_gnn_format(split_eq_nodes, split_eq_edges, label=split_label,
                                                     satisfiability=split_satisfiability)
                    multi_graph_dict[i + 1] = graph_dict

                # Dumping the dictionary to a JSON file
                json_file = graph_folder + "/" + f.replace(".label.json", ".graph.json").replace("train/", "")
                dump_to_json_with_format(multi_graph_dict, json_file)


def output_pair_eq_graphs(zip_file: str, graph_folder: str, graph_func: Callable, visualize: bool = False):
    parser = get_parser()

    with zipfile.ZipFile(zip_file, 'r') as zip_file_content:
        for f in zip_file_content.namelist():
            if fnmatch.fnmatch(f, '*.label.json'):
                eq_nodes, eq_edges, split_eq_list, split_eq_file_list, label_list, satisfiability_list = _read_label_and_eqs(
                    zip_file_content, f, graph_folder, parser, graph_func)

                # print("eq", eq.eq_str)
                for split_eq, split_file, split_label, split_satisfiability in zip(split_eq_list, split_eq_file_list,
                                                                                   label_list, satisfiability_list):
                    # print("split_eq", split_eq.eq_str)
                    split_eq_odes, split_eq_edges = graph_func(split_eq.left_terms, split_eq.right_terms)
                    merged_nodes, merged_edges = merge_graphs(eq_nodes, eq_edges, split_eq_odes, split_eq_edges)
                    if visualize == True:
                        draw_graph(nodes=merged_nodes, edges=merged_edges,
                                   filename=graph_folder + "/" + split_file.replace("train/", ""))

                    graph_dict = graph_to_gnn_format(merged_nodes, merged_edges, label=split_label,
                                                     satisfiability=split_satisfiability)
                    # Dumping the dictionary to a JSON file
                    json_file = graph_folder + "/" + (strip_file_name_suffix(split_file) + ".graph.json").replace(
                        "train/", "")
                    dump_to_json_with_format(graph_dict, json_file)


def output_eq_graphs(zip_file: str, graph_folder: str, graph_func: Callable, visualize: bool = False):
    parser = get_parser()
    with zipfile.ZipFile(zip_file, 'r') as zip_file_content:
        for f in zip_file_content.namelist():
            if fnmatch.fnmatch(f, '*.eq'):
                # eq_file_list = glob.glob(graph_folder + "/*.eq")
                # for file_path in eq_file_list:

                parsed_content = parser.parse(f, zip_file_content)
                # print("parsed_content:", parsed_content)

                answer_file = strip_file_name_suffix(f) + ".answer"
                with zip_file_content.open(answer_file) as file:
                    # with open(answer_file, 'r') as file:
                    answer = file.read()
                    answer = answer.decode('utf-8')

                for eq in parsed_content["equation_list"]:
                    if visualize == True:
                        # visualize
                        pass  # todo adapt to zip file
                        # eq.visualize_graph(file_path, graph_func)
                    # get gnn format
                    nodes, edges = graph_func(eq.left_terms, eq.right_terms)
                    satisfiability = answer
                    graph_dict = graph_to_gnn_format(nodes, edges, label=satisfiability_to_int_label[satisfiability],
                                                     satisfiability=satisfiability)
                    # print(graph_dict)
                    # Dumping the dictionary to a JSON file
                    json_file = graph_folder + "/" + (strip_file_name_suffix(f) + ".graph.json").replace("train/", "")
                    dump_to_json_with_format(graph_dict, json_file)


def get_parser():
    parser_type = EqParser()
    return Parser(parser_type)


def generate_train_data_in_one_folder(folder, algorithm, algorithm_parameters, train_data_solvability):
    graph_type = "graph_1"
    if algorithm == "SplitEquationsExtractData":
        parameters_list = ["fixed", "--termination_condition termination_condition_0",
                           f"--graph_type {graph_type}",
                           f"--algorithm SplitEquations",
                           f"--order_equations_method category"
                           ]
    else:
        parameters_list = ["fixed", f"--algorithm {algorithm}", f"--termination_condition termination_condition_0"]

    # prepare train folder
    all_eq_folder = f"{folder}/{train_data_solvability}"
    train_eq_folder = folder + "/train"

    # copy answers from divide folder
    # divided_folder = benchmark + "/ALL"
    # folder_number = sum(
    #     [1 for fo in os.listdir(bench_folder + "/" + divided_folder) if "divided" in os.path.basename(fo)])
    # for i in range(folder_number):
    #     divided_folder_index = i + 17
    #     for answer_file in glob.glob(
    #             bench_folder + "/" + divided_folder + "/divided_" + str(divided_folder_index) + "/*.answer"):
    #         shutil.copy(answer_file, all_eq_folder)

    if not os.path.exists(train_eq_folder):
        os.mkdir(train_eq_folder)
    else:
        shutil.rmtree(train_eq_folder)
        os.mkdir(train_eq_folder)
    for f in glob.glob(all_eq_folder + "/*.eq") + glob.glob(all_eq_folder + "/*.answer") + glob.glob(
            all_eq_folder + "/*.unsatcore"):
        shutil.copy(f, train_eq_folder)

    # extract train data
    eq_file_list = glob.glob(train_eq_folder + "/*.eq")
    eq_file_list_len = len(eq_file_list)
    for i, file_path in enumerate(eq_file_list):
        file_name = strip_file_name_suffix(file_path)
        print(f"-- {i}/{eq_file_list_len} --")
        print(file_path)

        # get satisfiability
        answer_file_path = file_name + ".answer"
        if os.path.exists(answer_file_path):  # read file answer
            print("read answer from file")
            with open(answer_file_path, "r") as f:
                satisfiability = f.read().strip("\n")
        elif train_data_solvability == "UNSAT":
            satisfiability = "UNSAT"
        elif train_data_solvability == "SAT":
            satisfiability = "SAT"
        else:
            result_dict = run_on_one_problem(file_path=file_path,
                                             parameters_list=parameters_list,
                                             solver="this", solver_log=False)
            satisfiability = result_dict["result"]

        print("satisfiability:", satisfiability)
        parameters_list = ["fixed",
                           f"--termination_condition {algorithm_parameters['termination_condition']}",
                           f"--graph_type {graph_type}",
                           f"--algorithm {algorithm}",
                           f"--order_equations_method {algorithm_parameters['order_equations_method']}",
                           f"--output_train_data True",
                           f"--eq_satisfiability {satisfiability}"
                           ]

        result_dict = run_on_one_problem(file_path=file_path,
                                         parameters_list=parameters_list,
                                         solver="this", solver_log=True)

    # compress
    zip_folder(folder_path=train_eq_folder, output_zip_file=train_eq_folder + ".zip")
    shutil.rmtree(train_eq_folder)
    print("done")


def draw_graph_for_one_folder(graph_type, benchmark_path, task, visualize=False):
    if task == "task_1":
        draw_func = output_eq_graphs  # task 1
    elif task == "task_2":
        draw_func = output_pair_eq_graphs  # task 2
    elif task == "task_3":
        draw_func = output_split_eq_graphs  # task 3
    elif task == "rank_task":  # G:List[graph]
        draw_func = output_rank_eq_graphs

    train_eq_folder = benchmark_path + "/train"
    train_zip_file = train_eq_folder + ".zip"
    for graph_type in [graph_type]:
        # prepare folder
        graph_folder = benchmark_path + "/" + graph_type

        if os.path.exists(graph_folder):
            shutil.rmtree(graph_folder)
            os.mkdir(graph_folder)
        else:
            os.mkdir(graph_folder)

        # draw one type graphs
        print(f"- draw {graph_type} -")
        draw_func(zip_file=train_zip_file, graph_folder=graph_folder, graph_func=graph_func_map[graph_type],
                  visualize=visualize)

        # compress
        zip_folder(folder_path=graph_folder, output_zip_file=graph_folder + ".zip")
        shutil.rmtree(graph_folder)

    print("done")


def store_dataset_to_pickle_one_folder(graph_type, folder, parameters, func):
    parameters["folder"] = folder
    parameters["graph_type"] = graph_type

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
    if parameters["rank_task"] == 0:
        dataset = WordEquationDatasetMultiClassificationRankTask0(graph_folder=graph_folder)
    elif parameters["rank_task"] == 1:
        dataset = WordEquationDatasetMultiClassificationRankTask1(graph_folder=graph_folder)
    elif parameters["rank_task"] == 2:
        dataset = WordEquationDatasetMultiClassificationRankTask2(graph_folder=graph_folder,
                                                                  label_size=parameters["label_size"])
    else:
        raise ValueError("rank_task should be 0,1,2")

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
