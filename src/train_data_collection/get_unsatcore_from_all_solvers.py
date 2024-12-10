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
from src.process_benchmarks.eq2smt_utils import one_eq_file_to_smt2
from src.process_benchmarks.utils import run_on_one_problem
from src.solver.Constants import bench_folder
import glob
from tqdm import tqdm
from src.train_data_collection.utils import solve_the_core_by_different_solver, \
    get_sorted_unsatcore_list_with_fixed_eq_number, clean_temp_files_while_extract_unsatcore, \
    run_one_problem_according_to_solver
from src.solver.DataTypes import Formula, Equation
from src.solver.Parser import EqParser, Parser
from src.solver.independent_utils import strip_file_name_suffix, create_folder, color_print
import json
from itertools import combinations


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

    # get_unsatcore_func = get_minimum_unsatcore
    get_unsatcore_func = get_non_minimum_unsatcore

    solver_parameter_list_map = {"z3": [], "z3-noodler": ["smt.string_solver=\"noodler\""], "cvc5": [],
                                 "ostrich": [], "woorpje": [], "this:category_random": ["fixed",
                                                                                        f"--termination_condition termination_condition_0",
                                                                                        f"--algorithm SplitEquations",
                                                                                        f"--graph_type graph_1",
                                                                                        f"--order_equations_method category_random", ]}
    shell_timeout_for_one_run = 10

    benchmark_folder = f"{working_folder}/{folder}"
    file_list = glob.glob(benchmark_folder + "/*.eq")
    for eq_file in tqdm(file_list, desc="file_list processing progress"):
        # parse .predicted_unsatcore file
        print(f"--- current file:{eq_file} ---")
        parsed_content = parser.parse(eq_file)
        ranked_formula = Formula(parsed_content["equation_list"])

        # check the satisfiability first.
        solver_list, satisfiability_list = check_satisfiability(solver_parameter_list_map, eq_file, solver_log,
                                                                shell_timeout_for_one_run)
        for s, sat in zip(solver_list, satisfiability_list):
            print(f"{s}:{sat}")

        if os.path.exists(strip_file_name_suffix(eq_file) + ".answer"):
            os.remove(strip_file_name_suffix(eq_file) + ".answer")

        if "UNSAT" in satisfiability_list and "SAT" in satisfiability_list:
            color_print("inconsistent", "red")
        elif "SAT" in satisfiability_list:
            color_print("SAT, no need to find unsat core", "blue")
        else:  # unsat or unknown, begin to find unsatcore

            print("--- extract unsatcore ---")
            get_unsatcore_func(eq_file, ranked_formula, solver_parameter_list_map, solver_log,
                               shell_timeout_for_one_run)


def get_non_minimum_unsatcore(eq_file, original_formula, solver_parameter_list_map, solver_log,
                              shell_timeout_for_one_run):
    # delete eq one by one to find the unsat core
    solvability_log = ""
    total_eq_number = original_formula.eq_list_length
    current_formula = Formula(original_formula.eq_list)

    current_formula_length = current_formula.eq_list_length
    for i in range(total_eq_number):  # decide which one to delete
        # delete i-th eq
        unsatcore = Formula(current_formula.eq_list[:i] + current_formula.eq_list[i + 1:])

        current_unsatcore_formula, current_unsatcore_eq_file = write_unsatcore_to_eq_file(unsatcore.eq_list, eq_file)

        log_text = f"    current_core_eq_number:{current_unsatcore_formula.eq_list_length}"
        print(log_text)
        solvability_log += log_text + "\n"

        # check whether it is an unsatcore
        # solve the current unsatcore eq file by different solvers
        satisfiability, first_solved_solver, unsatcore_smt2_file, solving_time, log_from_differernt_solvers = solve_the_core_by_different_solver(
            current_unsatcore_eq_file, solver_parameter_list_map, solver_log, shell_timeout_for_one_run)
        solvability_log += log_from_differernt_solvers



        if satisfiability == "UNSAT":  # this eq can be deleted go to next
            #record current best unsatcore
            current_formula = unsatcore
            unsatcore_summary_folder = create_folder(f"{strip_file_name_suffix(eq_file)}_non_minimum_unsatcore")
            store_unsatcore_to_file(unsatcore_summary_folder, original_formula, current_unsatcore_formula,
                                    satisfiability, first_solved_solver, solving_time, solvability_log,
                                    current_unsatcore_eq_file, unsatcore_smt2_file)

        elif satisfiability == "SAT":
            pass
        else:
            pass




def get_minimum_unsatcore(eq_file, ranked_formula, solver_parameter_list_map, solver_log,
                          shell_timeout_for_one_run):
    # delete n to 1 eqs to find the unsat core (increase eq number systematically)
    found_unsatcore = False
    total_eq_number = ranked_formula.eq_list_length
    delete_eq_number_list = list(reversed(range(1, total_eq_number)))

    solvability_log = ""
    for eq_number_to_delete in tqdm(delete_eq_number_list, desc="deleting progress"):
        if found_unsatcore == True:
            break

        unsarcore_list_sorted = get_sorted_unsatcore_list_with_fixed_eq_number(ranked_formula.eq_list,
                                                                               eq_number_to_delete)

        for index, unsatcore in enumerate(unsarcore_list_sorted):
            log_text = f"    Delete {eq_number_to_delete} from {total_eq_number}, {index}/{len(unsarcore_list_sorted)}"
            print(log_text)
            solvability_log += log_text + "\n"

            current_unsatcore_formula, current_unsatcore_eq_file = write_unsatcore_to_eq_file(unsatcore, eq_file)

            log_text = f"    current_core_eq_number:{current_unsatcore_formula.eq_list_length}"
            print(log_text)
            solvability_log += log_text + "\n"

            # check whether it is an unsatcore
            # solve the current unsatcore eq file by different solvers
            satisfiability, first_solved_solver, unsatcore_smt2_file, solving_time, log_from_differernt_solvers = solve_the_core_by_different_solver(
                current_unsatcore_eq_file, solver_parameter_list_map, solver_log, shell_timeout_for_one_run)
            solvability_log += log_from_differernt_solvers

            if satisfiability == "UNSAT":
                unsatcore_summary_folder = create_folder(f"{strip_file_name_suffix(eq_file)}_unsatcore")
                store_unsatcore_to_file(unsatcore_summary_folder, ranked_formula, current_unsatcore_formula,
                                        satisfiability, first_solved_solver, solving_time, solvability_log,
                                        current_unsatcore_eq_file, unsatcore_smt2_file)

                found_unsatcore = True

                break

            else:
                # clean temp current eq and smt unsat core files
                clean_temp_files_while_extract_unsatcore(current_unsatcore_eq_file, unsatcore_smt2_file)


def write_unsatcore_to_eq_file(unsatcore, eq_file):
    # store to eq file
    current_unsatcore_formula = Formula(unsatcore)
    eq_string_to_file = current_unsatcore_formula.eq_string_for_file()
    # create eq file
    current_unsatcore_eq_file = f"{strip_file_name_suffix(eq_file)}.current_unsatcore"
    with open(current_unsatcore_eq_file, "w") as f:
        f.write(eq_string_to_file)
    return current_unsatcore_formula, current_unsatcore_eq_file


def store_unsatcore_to_file(unsatcore_summary_folder, ranked_formula, current_unsatcore_formula,
                            satisfiability, first_solved_solver, solving_time, solvability_log,
                            current_unsatcore_eq_file, unsatcore_smt2_file):
    summary_dict = {"original_eq_number": ranked_formula.eq_list_length,
                    "current_unsatcore_eq_number": current_unsatcore_formula.eq_list_length,
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
    if os.path.exists(unsatcore_smt2_file):
        shutil.move(unsatcore_smt2_file, unsatcore_summary_folder)
    os.remove(current_unsatcore_eq_file + ".answer")


def check_satisfiability(solver_parameter_list_map, eq_file, solver_log, shell_timeout_for_one_run):
    solver_list = []
    satisfiability_list = []

    for solver, parameter_list in solver_parameter_list_map.items():
        result_dict, solving_time, unsatcore_smt2_file = run_one_problem_according_to_solver(solver, eq_file,
                                                                                             parameter_list, solver_log,
                                                                                             shell_timeout_for_one_run)

        solver_list.append(solver)
        satisfiability_list.append(result_dict["result"])

    return solver_list, satisfiability_list


if __name__ == '__main__':
    main()
