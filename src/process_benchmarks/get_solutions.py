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
from src.process_benchmarks.utils import run_on_one_track,result_summary,write_to_cvs_file
import csv
from src.solver.Constants import bench_folder, BRANCH_CLOSED, MAX_PATH_REACHED, INTERNAL_TIMEOUT, RECURSION_DEPTH_EXCEEDED, \
    RECURSION_ERROR, project_folder
from src.process_benchmarks.utils import summary_one_track
import argparse
import json

def main():
    # parse argument
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')

    arg_parser.add_argument('--configuration_file', type=str, default=None,
                            help='path to configuration json file ')

    args = arg_parser.parse_args()

    # Accessing the arguments
    configuration_file = args.configuration_file

    if configuration_file is not None:
        # read json file
        with open(configuration_file) as f:
            config = json.load(f)
    else:
        benchmark_name = "test_track_task_3"
        termination_condition = "termination_condition_0"
        config = {
            "benchmark_name": benchmark_name,
            "benchmark_folder":bench_folder+"/"+benchmark_name+"/ALL/ALL",
            #"solver":"ostrich",
            #"parameters_list":[],
            "solver": "this",
            "parameters_list":["fixed",f"--termination_condition {termination_condition}"],
            # "parameters_list": ["gnn","--graph_type graph_3",
            #                     "--gnn_model_path "+project_folder+"/Models/model_0_graph_3_GINSplit.pth",
            #                     f"--gnn_task task_3",f"--termination_condition {termination_condition}"],
            "summary_folder_name":f"{benchmark_name}_summary"
        }

    solver_log = False
    suffix_dict = {"z3": ".smt2", "woorpje": ".eq", "this": ".eq", "ostrich": ".smt2", "cvc5": ".smt2"}

    print("config:",config)

    run_on_one_track(config["benchmark_name"], config["benchmark_folder"], config["parameters_list"], config["solver"], suffix_dict,summary_folder_name=config["summary_folder_name"],
                     solver_log=solver_log)

    print("done")






if __name__ == '__main__':
    main()
