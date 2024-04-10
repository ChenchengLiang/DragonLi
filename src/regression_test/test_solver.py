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
from src.solver.algorithms import ElimilateVariablesRecursive, SplitEquations
from src.solver.independent_utils import strip_file_name_suffix, check_list_consistence
from src.process_benchmarks.utils import run_on_one_problem
from src.process_benchmarks.eq2smt_utils import one_eq_file_to_smt2
import csv
from collections import defaultdict
from tqdm import tqdm


def main():
    algorithm_configuration_list: List[Tuple[str, List[str]]] = [
        (ElimilateVariablesRecursive, ["fixed", f"--termination_condition termination_condition_0"]),
        (SplitEquations, ["fixed", f"--algorithm SplitEquations", f"--order_equations_method fixed",
                          f"--termination_condition termination_condition_0"]),
        (SplitEquations, ["fixed", f"--algorithm SplitEquations", f"--order_equations_method random",
                          f"--termination_condition termination_condition_0"]),
        (SplitEquations, ["fixed", f"--algorithm SplitEquations", f"--order_equations_method category",
                          f"--termination_condition termination_condition_0"]),

    ]

    log = True

    # test
    consistance_list = []
    for file_path in tqdm(glob.glob(bench_folder + "/regression_test/ALL/*.eq"), desc="progress"):
        satisfiability_list = run_solvers(file_path, algorithm_configuration_list, log=log)

        consistance = check_consistency(satisfiability_list)
        consistance_list.append((os.path.basename(file_path), consistance, satisfiability_list))

    print("-" * 10, "consistance", "-" * 10)
    for x in consistance_list:
        print(x)

    # write to cvs
    write_to_csv(consistance_list)


def write_to_csv(consistance_list):
    # Data to be written to the CSV file
    row_1 = ['file', 'consistency'] + [x[0] for x in consistance_list[0][2]]

    # Open the CSV file in write mode ('w') and create a writer object
    csv_file = bench_folder + "/regression_test/results.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the row to the CSV file
        writer.writerow(row_1)
        for c in consistance_list:
            writer.writerow([c[0], c[1]] + [x[1] for x in c[2]])

    input_csv_path = csv_file  # Replace with your actual input file path
    output_csv_path = os.path.dirname(csv_file) + "/summary.csv"  # Replace with your desired output file path

    count_values(input_csv_path, output_csv_path)


def count_values(input_file_path, output_file_path):
    """
    Count occurrences of 'SAT', 'UNSAT', 'UNKNOWN' in each column of a CSV file and output the results to a new CSV file.

    Parameters:
    - input_file_path: Path to the input CSV file.
    - output_file_path: Path to the output CSV file where the summary will be saved.
    """
    with open(input_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Extract the first row as headers

        # Initialize a dictionary to store the count of 'SAT', 'UNSAT', 'UNKNOWN' for each column
        counts = {header: defaultdict(int) for header in headers}

        for row in reader:
            for header, value in zip(headers, row):
                if value in ['SAT', 'UNSAT', 'UNKNOWN']:
                    counts[header][value] += 1

    # Prepare data for output
    output_data = [['Column', 'SAT', 'UNSAT', 'UNKNOWN']]
    for header in headers:
        output_data.append([
            header,
            counts[header]['SAT'],
            counts[header]['UNSAT'],
            counts[header]['UNKNOWN']
        ])

    # Write the summary counts to a new CSV file
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        output_data = [output_data[0]] + output_data[3:]
        writer.writerows(output_data)


def check_consistency(satisfiability_list: List[Tuple[str, str]]) -> bool:
    filtered_statuses = [status for _, status in satisfiability_list if status != UNKNOWN]

    if not filtered_statuses:
        return True  # List is consistent if there are no SAT or UNSAT values.

    # Check if all elements are the same in the filtered list (either all SAT or all UNSAT)
    return all(status == filtered_statuses[0] for status in filtered_statuses)


def run_solvers(file_path, algorithm_configuration_list, log=False):
    for sh_file in glob.glob(bench_folder + "/src/process_benchmarks/temp_shell/*"):
        os.remove(sh_file)
    if not os.path.exists(strip_file_name_suffix(file_path) + ".smt2"):
        one_eq_file_to_smt2(file_path)

    satisfiability_list: List[Tuple[str, str]] = []

    # other solvers
    for solver in ["z3", "cvc5", "ostrich", "woorpje","z3_noodler"]:
        if solver == "woorpje":
            file = strip_file_name_suffix(file_path) + ".eq"
        else:
            file = strip_file_name_suffix(file_path) + ".smt2"
        other_solver_result_dict = run_on_one_problem(file_path=file, parameters_list=[], solver=solver,
                                                      solver_log=log)
        satisfiability_list.append((solver, other_solver_result_dict["result"]))

    # this solver
    for i, ac in enumerate(algorithm_configuration_list):
        (algorithm, parameter_list) = ac
        file = strip_file_name_suffix(file_path) + ".eq"
        olver_result_dict = run_on_one_problem(file_path=file, parameters_list=parameter_list, solver="this",
                                               solver_log=log)

        satisfiability_list.append((f"this:{algorithm.__name__}-config_{i}", olver_result_dict["result"]))
    return satisfiability_list


if __name__ == '__main__':
    main()
