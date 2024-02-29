import configparser
import glob
import os
import sys
from typing import List, Tuple

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.Constants import bench_folder, project_folder, UNKNOWN, SUCCESS, FAIL, RED, GREEN, YELLOW, COLORRESET
from src.solver.Parser import Parser, EqParser, SMT2Parser
from src.solver.Solver import Solver
from src.solver.utils import print_results, graph_func_map
from src.solver.algorithms import ElimilateVariablesRecursive,SplitEquations
from src.solver.independent_utils import strip_file_name_suffix, check_list_consistence
from src.process_benchmarks.utils import run_on_one_problem
from src.process_benchmarks.eq2smt_utils import one_eq_file_to_smt2
import csv


def main():
    algorithm_configuration_list:List[Tuple[str,List[str]]] = [
        (ElimilateVariablesRecursive,["fixed", f"--termination_condition execute_termination_condition_0"]),
        (SplitEquations, ["fixed", f"--algorithm SplitEquations", f"--choose_unknown_eq_method fixed"]),
    ]


    # test
    consistance_list = []
    for file_path in glob.glob(bench_folder + "/regression_test/ALL/*.eq"):
        satisfiability_list=run_solvers(file_path, algorithm_configuration_list)


        consistance=check_consistency(satisfiability_list)
        consistance_list.append((os.path.basename(file_path), consistance,satisfiability_list))

    print("-" * 10, "consistance", "-" * 10)
    for x in consistance_list:
        print(x)

    #write to cvs
    write_to_csv(consistance_list)

def write_to_csv(consistance_list):
    # Data to be written to the CSV file
    row_1 = ['file', 'consistency'] + [x[0] for x in consistance_list[0][2]]

    # Open the CSV file in write mode ('w') and create a writer object
    with open(bench_folder + "/regression_test/results.csv", 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the row to the CSV file
        writer.writerow(row_1)
        for c in consistance_list:
            writer.writerow([c[0], c[1]] + [x[1] for x in c[2]])

def check_consistency(satisfiability_list: List[Tuple[str, str]]) -> bool:

    filtered_statuses = [status for _, status in satisfiability_list if status != UNKNOWN]

    if not filtered_statuses:
        return True  # List is consistent if there are no SAT or UNSAT values.

    # Check if all elements are the same in the filtered list (either all SAT or all UNSAT)
    return all(status == filtered_statuses[0] for status in filtered_statuses)


def run_solvers(file_path, algorithm_configuration_list):
    for sh_file in glob.glob(bench_folder + "/src/process_benchmarks/temp_shell/*"):
        os.remove(sh_file)
    if not os.path.exists(strip_file_name_suffix(file_path) + ".smt2"):
        one_eq_file_to_smt2(file_path)

    satisfiability_list:List[Tuple[str,str]] = []

    #other solvers
    for solver in ["z3", "cvc5", "ostrich", "woorpje"]:
        if solver == "woorpje":
            file = strip_file_name_suffix(file_path) + ".eq"
        else:
            file = strip_file_name_suffix(file_path) + ".smt2"
        other_solver_result_dict = run_on_one_problem(file_path=file, parameters_list=[], solver=solver,
                                                      solver_log=False)
        satisfiability_list.append((solver,other_solver_result_dict["result"]))

    #this solver
    for i,ac in enumerate(algorithm_configuration_list):
        (algorithm, parameter_list)=ac
        file = strip_file_name_suffix(file_path) + ".eq"
        olver_result_dict = run_on_one_problem(file_path=file, parameters_list=parameter_list, solver="this",
                                                      solver_log=False)

        satisfiability_list.append((f"this:{algorithm.__name__}-config_{i}", olver_result_dict["result"]))
    return satisfiability_list



if __name__ == '__main__':
    main()
