import sys
import configparser



# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

import sys
from src.solver.Constants import project_folder

sys.path.append(project_folder)
from src.solver.Constants import bench_folder, recursion_limit
from src.solver.independent_utils import get_folders

from src.solver.algorithms.split_equations_extract_data import SplitEquationsExtractData
from src.train_data_collection.utils import generate_train_data_in_one_folder

def main():
    benchmark = "conjunctive_01_track_train_rank_task_1_100"#"choose_eq_train"

    # algorithm = ElimilateVariablesRecursive
    # algorithm_parameters = {"branch_method": "extract_branching_data_task_3", "extract_algorithm": "fixed",
    #                         "termination_condition": "termination_condition_0"}  # extract_branching_data_task_2

    algorithm = "SplitEquationsExtractData"
    algorithm_parameters={"branch_method": "fixed", "order_equations_method": "category_random","task":"dynamic_embedding"}
    train_data="UNSAT"

    sys.setrecursionlimit(recursion_limit)
    benchmark_path = bench_folder + "/" + benchmark
    folder_list = [folder for folder in get_folders(benchmark_path) if
                   "divided" in folder or "valid" in folder]
    # folder_list = [folder for folder in get_folders(benchmark_path) if
    #                "divided" in folder]
    print(folder_list)
    if len(folder_list) != 0:
        for folder in folder_list:
            generate_train_data_in_one_folder(benchmark_path + "/" + folder, algorithm, algorithm_parameters,train_data)
    else:
        generate_train_data_in_one_folder(benchmark_path, algorithm, algorithm_parameters,train_data)


if __name__ == '__main__':
    main()
