
import glob
import os
import sys
import configparser
# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')

sys.path.append(path)
from typing import List, Tuple, Dict
from src.solver.Constants import bench_folder,suffix_dict
from src.process_benchmarks.utils import run_on_one_track

def main():
    benchmark_name = "01_track_multi_word_equations_generated_eval_1001_2000_new_trainable_data_test"


    # extract unsat cores
    solver="z3"

    config = {
        "benchmark_name": benchmark_name,
        "benchmark_folder": f"{bench_folder}/{benchmark_name}/{solver}",
        "solver":solver,
        "parameters_list":[],
        "summary_folder_name": f"{benchmark_name}_summary"
    }
    solver_log = True
    print("config:", config)
    run_on_one_track(config["benchmark_name"], config["benchmark_folder"], config["parameters_list"], config["solver"],
                     suffix_dict, summary_folder_name=config["summary_folder_name"],
                     solver_log=solver_log)

    #
    #
    # # run Dragon with SplitEquations algorithm
    # termination_condition = "termination_condition_0"
    # algorithm="SplitEquations"
    # config = {
    #     "benchmark_name": benchmark_name,
    #     "benchmark_folder": bench_folder + "/" + benchmark_name + "/ALL/ALL",
    #     # "solver":"ostrich",
    #     # "parameters_list":[],
    #     "solver": "this",
    #     "parameters_list": ["fixed",
    #                         f"--termination_condition {termination_condition}",
    #                         f"--algorithm {algorithm}",
    #                         f"--order_equations_method category"
    #                         ],
    #     "summary_folder_name": f"{benchmark_name}_summary"
    # }
    #
    # solver_log = False
    #
    # print("config:", config)
    #
    # run_on_one_track(config["benchmark_name"], config["benchmark_folder"], config["parameters_list"], config["solver"],
    #                  suffix_dict, summary_folder_name=config["summary_folder_name"],
    #                  solver_log=solver_log)


if __name__ == '__main__':
    main()