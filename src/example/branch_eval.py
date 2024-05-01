import glob
import os
from typing import List, Tuple

from tqdm import tqdm

import sys
sys.path.append(".")
from src.regression_test.utils import run_solvers, check_consistency, write_to_csv
from src.solver.algorithms import ElimilateVariablesRecursive

def main():
    folder = f"benchmaks_and_experimental_results/example/01_track_eval"
    eval(folder)

def eval(folder):
    #termination_condition_i , i \in [0,1,2] corresponding to BT_{j} j \in [1,2,3]
    algorithm_configuration_list: List[Tuple[str, List[str]]] = [
        (ElimilateVariablesRecursive, ["fixed", f"--termination_condition termination_condition_0"]),
        (ElimilateVariablesRecursive, ["random", f"--termination_condition termination_condition_0"]),
        (ElimilateVariablesRecursive, ["gnn", f"--termination_condition termination_condition_0"]),
        (ElimilateVariablesRecursive, ["gnn:fixed", f"--termination_condition termination_condition_0"]),
        (ElimilateVariablesRecursive, ["gnn:random", f"--termination_condition termination_condition_0"]),
    ]

    log = True
    other_solver_list =[]

    # test
    consistance_list = []
    for file_path in tqdm(glob.glob(f"{folder}/ALL/*.eq"), desc="progress"):
        satisfiability_list = run_solvers(file_path, algorithm_configuration_list,other_solver_list, log=log)

        consistance = check_consistency(satisfiability_list)
        consistance_list.append((os.path.basename(file_path), consistance, satisfiability_list))

    print("-" * 10, "consistance", "-" * 10)
    for x in consistance_list:
        print(x)

    # write to cvs
    write_to_csv(consistance_list, folder)


if __name__ == '__main__':
    main()
