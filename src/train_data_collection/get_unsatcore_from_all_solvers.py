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
    get_sorted_unsatcore_list_with_fixed_eq_number, clean_temp_files_while_extract_unsatcore
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

    solver_parameter_list_map = {"z3": [], "z3-noodler": ["smt.string_solver=\"noodler\""], "cvc5": [],
                                 "ostrich": [], "woorpje": [], "this": ["fixed",
                                                                        f"--termination_condition termination_condition_0",
                                                                        f"--algorithm SplitEquations",
                                                                        f"--graph_type graph_1",
                                                                        f"--order_equations_method category_random", ]}
    shell_timeout_for_one_run = 20

    benchmark_folder = f"{working_folder}/{folder}"
    file_list = glob.glob(benchmark_folder + "/*.eq")
    for eq_file in tqdm(file_list, desc="file_list processing progress"):
        # parse .predicted_unsatcore file
        print(f"--- current file:{eq_file} ---")
        parsed_content = parser.parse(eq_file)
        ranked_formula = Formula(parsed_content["equation_list"])

        #check the satisfiability first.
        solver_list,satisfiability_list = check_satisfiability(solver_parameter_list_map, eq_file, solver_log, shell_timeout_for_one_run)
        for s, sat in zip(solver_list, satisfiability_list):
            print(f"{s}:{sat}")

        if "UNSAT" in satisfiability_list and "SAT" in satisfiability_list:
            color_print("inconsistent", "red")
        elif "SAT" in satisfiability_list:
            color_print("SAT, no need to find unsat core", "blue")
        else: # unsat or unknown, begin to find unsatcore

            # delete n to 1 eqs to find the unsat core
            found_unsatcore = False
            total_eq_number = len(parsed_content["equation_list"])
            delete_eq_number_list = list(reversed(range(1, total_eq_number)))


            solvability_log = ""
            for eq_number_to_delete in tqdm(delete_eq_number_list, desc="deleting progress"):
                if found_unsatcore == True:
                    break

                unsarcore_list_sorted = get_sorted_unsatcore_list_with_fixed_eq_number(parsed_content["equation_list"],
                                                                                       eq_number_to_delete)

                for index, unsatcore in enumerate(unsarcore_list_sorted):
                    print(f"    Delete {eq_number_to_delete} from {total_eq_number}, {index}/{len(unsarcore_list_sorted)}")
                    # store to eq file
                    current_unsatcore_formula = Formula(unsatcore)
                    eq_string_to_file = current_unsatcore_formula.eq_string_for_file()
                    # create eq file
                    current_unsatcore_eq_file = f"{strip_file_name_suffix(eq_file)}.current_unsatcore"
                    with open(current_unsatcore_eq_file, "w") as f:
                        f.write(eq_string_to_file)

                    log_text = f"    current_core_eq_number:{current_unsatcore_formula.eq_list_length}"
                    print(log_text)
                    solvability_log += log_text + "\n"

                    # check whether it is an unsatcore
                    # solve the current unsatcore eq file by different solvers
                    satisfiability, first_solved_solver, unsatcore_smt2_file, solving_time, log_from_differernt_solvers = solve_the_core_by_different_solver(
                        current_unsatcore_eq_file, solver_parameter_list_map, solver_log, shell_timeout_for_one_run)
                    solvability_log += log_from_differernt_solvers

                    if satisfiability == "UNSAT":
                        unsatcore_summary_folder = create_folder(f"{strip_file_name_suffix(file)}_unsatcore")
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
                        shutil.move(unsatcore_smt2_file, unsatcore_summary_folder)
                        os.remove(current_unsatcore_eq_file + ".answer")

                        found_unsatcore=True

                        break

                    else:
                        # clean temp current eq and smt unsat core files
                        clean_temp_files_while_extract_unsatcore(current_unsatcore_eq_file,unsatcore_smt2_file)



def check_satisfiability(solver_parameter_list_map,eq_file,solver_log,shell_timeout_for_one_run):
    solver_list = []
    satisfiability_list = []

    for solver, parameter_list in solver_parameter_list_map.items():
        if solver == "this" or solver == "woorpje":
            result_dict = run_on_one_problem(eq_file, parameter_list, solver,
                                             solver_log=solver_log,
                                             shell_timeout=shell_timeout_for_one_run)
        else:
            unsatcore_smt2_file = one_eq_file_to_smt2(eq_file)
            result_dict = run_on_one_problem(unsatcore_smt2_file, parameter_list, solver,
                                             solver_log=solver_log,
                                             shell_timeout=shell_timeout_for_one_run)
        solver_list.append(solver)
        satisfiability_list.append(result_dict["result"])

    return solver_list,satisfiability_list

if __name__ == '__main__':
    main()
