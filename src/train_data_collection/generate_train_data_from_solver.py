import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)


from src.solver.Constants import bench_folder, recursion_limit
from src.solver.independent_utils import get_folders
from src.train_data_collection.utils import generate_train_data_in_one_folder

def main():
    benchmark = "01_track_multi_word_equations_eq_2_50_generated_train_1_10000_UNSAT_data_extraction"#"choose_eq_train"

    # algorithm = ElimilateVariablesRecursive
    # algorithm_parameters = {"branch_method": "extract_branching_data_task_3", "extract_algorithm": "fixed",
    #                         "termination_condition": "termination_condition_0"}  # extract_branching_data_task_2

    algorithm = "SplitEquationsExtractData"
    algorithm_parameters = {"branch_method": "fixed",
                            "order_equations_method": "category",
                            "termination_condition": "termination_condition_7",
                            "task": "dynamic_embedding"}
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
