import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

import argparse
import os
import shutil
import time
from src.train_data_collection.utils import solve_the_core_by_different_solver, clean_temp_files_while_extract_unsatcore
from src.process_benchmarks.eq2smt_utils import one_eq_file_to_smt2
from src.process_benchmarks.utils import run_on_one_problem
from src.solver.Constants import bench_folder, rank_task_label_size_map, benchmark_A_model, benchmark_B_model, \
    mlflow_folder
import glob
from tqdm import tqdm

from src.solver.DataTypes import Formula
from src.solver.Parser import EqParser, Parser
from src.solver.independent_utils import strip_file_name_suffix, create_folder
import json

from src.solver.utils import graph_func_map


def main():
    # parse argument
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('benchmark', type=str,
                            help='benchmark name')
    arg_parser.add_argument('folder', type=str,
                            help='divided_i or valid_data folder')
    args = arg_parser.parse_args()
    # Accessing the arguments
    benchmark = args.benchmark
    folder = args.folder

    working_folder = f"{bench_folder}/{benchmark}"
    parser = Parser(EqParser())
    solver_log = False

    rank_task = 1
    task = "rank_task"
    benchmark_model = benchmark_B_model
    gnn_model_path = f"{mlflow_folder}/{benchmark_model['experiment_id']}/{benchmark_model['run_id']}/artifacts/model_0_{benchmark_model['graph_type']}_{benchmark_model['model_type']}.pth"
    print("gnn_model_path:", gnn_model_path)
    solver_parameter_list_map = {"z3": [], "z3-noodler": ["smt.string_solver=\"noodler\""], "cvc5": [],
                                 "ostrich": [], "woorpje": [],
                                 # "this": ["fixed",
                                 #          f"--termination_condition termination_condition_0",
                                 #          f"--algorithm SplitEquations",
                                 #          f"--graph_type {benchmark_model['graph_type']}",
                                 #          f"--order_equations_method category_random", ],
                                 "this:category_gnn_first_n_iterations": ["fixed",
                                                                          "--termination_condition termination_condition_0",
                                                                          f"--graph_type {benchmark_model['graph_type']}",
                                                                          f"--algorithm SplitEquations",
                                                                          f"--order_equations_method category_gnn_first_n_iterations",
                                                                          f"--gnn_model_path {gnn_model_path}",
                                                                          f"--gnn_task {task}",
                                                                          f"--rank_task {rank_task}"],
                                 "this:gnn_first_n_iterations_category": ["fixed",
                                                                          "--termination_condition termination_condition_0",
                                                                          f"--graph_type {benchmark_model['graph_type']}",
                                                                          f"--algorithm SplitEquations",
                                                                          f"--order_equations_method gnn_first_n_iterations_category",
                                                                          f"--gnn_model_path {gnn_model_path}",
                                                                          f"--gnn_task {task}",
                                                                          f"--rank_task {rank_task}"],
                                 "this:category_gnn": ["fixed",
                                                       "--termination_condition termination_condition_0",
                                                       f"--graph_type {benchmark_model['graph_type']}",
                                                       f"--algorithm SplitEquations",
                                                       f"--order_equations_method category_gnn",
                                                       f"--gnn_model_path {gnn_model_path}",
                                                       f"--gnn_task {task}",
                                                       f"--rank_task {rank_task}"],
                                 "this:category_gnn_formula_size": ["fixed",
                                                                    "--termination_condition termination_condition_0",
                                                                    f"--graph_type {benchmark_model['graph_type']}",
                                                                    f"--algorithm SplitEquations",
                                                                    f"--order_equations_method category_gnn_formula_size",
                                                                    f"--gnn_model_path {gnn_model_path}",
                                                                    f"--gnn_task {task}",
                                                                    f"--rank_task {rank_task}"],
                                 "this:category_gnn_each_n_iterations": ["fixed",
                                                                         "--termination_condition termination_condition_0",
                                                                         f"--graph_type {benchmark_model['graph_type']}",
                                                                         f"--algorithm SplitEquations",
                                                                         f"--order_equations_method category_gnn_each_n_iterations",
                                                                         f"--gnn_model_path {gnn_model_path}",
                                                                         f"--gnn_task {task}",
                                                                         f"--rank_task {rank_task}"],
                                 }
    shell_timeout_for_one_run = 20

    benchmark_folder = f"{working_folder}/{folder}"
    file_list = glob.glob(benchmark_folder + "/*.predicted_unsatcore")
    print("benchmark_folder:", benchmark_folder)
    print(f"file_list:{file_list}")
    for file in tqdm(file_list, desc="file_list processing progress"):
        # parse .predicted_unsatcore file
        print(f"--- current file:{file} ---")
        parsed_content = parser.parse(file)
        ranked_formula = Formula(parsed_content["equation_list"])

        solvability_log = ""
        # increase the eq one by one and check if it is still unsat
        for i in range(ranked_formula.eq_list_length):
            current_core_eq_number = i + 1
            log_text = f"    current_core_eq_number:{current_core_eq_number}"
            print(log_text)
            solvability_log += log_text + "\n"
            current_formula = Formula(ranked_formula.eq_list[:current_core_eq_number])

            # generate current unsat core eq file
            eq_string_to_file = current_formula.eq_string_for_file()
            current_unsatcore_eq_file = f"{strip_file_name_suffix(file)}.current_unsatcore"
            with open(current_unsatcore_eq_file, "w") as f:
                f.write(eq_string_to_file)

            # solve the current unsat core eq file by different solvers
            satisfiability, first_solved_solver, unsatcore_smt2_file, solving_time, log_from_differernt_solvers = solve_the_core_by_different_solver(
                current_unsatcore_eq_file, solver_parameter_list_map, solver_log, shell_timeout_for_one_run)

            solvability_log += log_from_differernt_solvers

            if satisfiability == "UNSAT":
                # give statistics log
                unsatcore_summary_folder = create_folder(f"{strip_file_name_suffix(file)}_predicted_unsatcore_eval")
                summary_dict = {"original_eq_number": ranked_formula.eq_list_length,
                                "current_unsatcore_eq_number": current_formula.eq_list_length,
                                "satisfiability": satisfiability,
                                "first_solved_solver": first_solved_solver,
                                "solving_time": solving_time}
                with open(f"{unsatcore_summary_folder}/summary.json", "w") as f:
                    json.dump(summary_dict, f, indent=4)

                # log solvability
                with open(f"{unsatcore_summary_folder}/solvability_log.txt", "w") as f:
                    f.write(solvability_log)

                # include results to a folder
                shutil.move(current_unsatcore_eq_file, unsatcore_summary_folder)
                shutil.move(unsatcore_smt2_file, unsatcore_summary_folder)
                os.remove(current_unsatcore_eq_file + ".answer")
                shutil.copy(file, unsatcore_summary_folder)

                break

            else:
                # clean temp current eq and smt unsat core files
                clean_temp_files_while_extract_unsatcore(current_unsatcore_eq_file, unsatcore_smt2_file)


if __name__ == '__main__':
    main()
