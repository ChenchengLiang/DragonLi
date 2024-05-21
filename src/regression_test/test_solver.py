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

from src.solver.Constants import bench_folder, project_folder
from src.solver.algorithms import ElimilateVariablesRecursive, SplitEquations
from src.regression_test.utils import run_solvers, write_to_csv, check_consistency
from tqdm import tqdm


def main():
    graph_type = "graph_1"
    model_type = "GCNSplit"
    gnn_model_path = f"{project_folder}/Models/model_0_{graph_type}_{model_type}.pth"

    algorithm_configuration_list: List[Tuple[str, List[str]]] = [
        (ElimilateVariablesRecursive, ["fixed", f"--termination_condition termination_condition_0"]),
        # (SplitEquations, ["fixed", f"--algorithm SplitEquations", f"--order_equations_method fixed",
        #                   f"--termination_condition termination_condition_0"]),
        # (SplitEquations, ["fixed", f"--algorithm SplitEquations", f"--order_equations_method random",
        #                   f"--termination_condition termination_condition_0"]),
        (SplitEquations, ["fixed", f"--algorithm SplitEquations", f"--order_equations_method category",
                          f"--termination_condition termination_condition_0"]),
        (SplitEquations, ["fixed", f"--algorithm SplitEquations", f"--order_equations_method category_gnn",
                          f"--termination_condition termination_condition_0",
                          f"--gnn_model_path {gnn_model_path}"]),
        (SplitEquations, ["fixed", f"--algorithm SplitEquations", f"--order_equations_method gnn",
                          f"--termination_condition termination_condition_0",
                          f"--gnn_model_path {gnn_model_path}"]),

    ]
    other_solver_list=["z3", "cvc5", "ostrich", "woorpje","z3-noodler"]

    log = True
    folder=f"{bench_folder}/regression_test"

    # test
    consistance_list = []
    for file_path in tqdm(glob.glob(f"{folder}/ALL/*.eq"), desc="progress"):
        satisfiability_list = run_solvers(file_path, algorithm_configuration_list, other_solver_list,log=log)

        consistance = check_consistency(satisfiability_list)
        consistance_list.append((os.path.basename(file_path), consistance, satisfiability_list))

    print("-" * 10, "consistance", "-" * 10)
    for x in consistance_list:
        print(x)

    # write to cvs
    write_to_csv(consistance_list,folder)


if __name__ == '__main__':
    main()
