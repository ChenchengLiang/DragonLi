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
from src.process_benchmarks.utils import run_on_one_benchmark,run_on_one_track,result_summary,write_to_cvs_file
from src.solver.Constants import bench_folder, BRANCH_CLOSED, MAX_PATH_REACHED, INTERNAL_TIMEOUT, RECURSION_DEPTH_EXCEEDED, \
    RECURSION_ERROR, project_folder
from src.solver.independent_utils import write_configurations_to_json_file

import json
import shutil

def main():
    solver_param_list = [
        ["this", ["fixed"]],
        ["this", ["random"]],
        ["this",["gnn","--graph_type graph_1","--gnn_model_path "+project_folder+"/models/model_graph_1_GAT.pth"]],
        ["this",["gnn","--graph_type graph_2","--gnn_model_path "+project_folder+"/models/model_graph_2_GAT.pth"]],
        # ["woorpje",[]],
        # ["z3",[]],
        # ["ostrich",[]],
        # ["cvc5",[]],
    ]

    benchmark_dict = {
        #"test_track": bench_folder + "/test",
        "example_track": bench_folder + "/examples",
        # "track_01": bench_folder + "/01_track",
        # "g_track_01_sat":bench_folder + "/01_track_generated/SAT",
        # "g_track_01_mixed": bench_folder + "/01_track_generated/mixed",
        # "g_track_01_eval":bench_folder + "/01_track_generated_eval_data",
        # "track_02": bench_folder + "/02_track",
        # "track_03": bench_folder + "/03_track",
        # "track_04": bench_folder + "/04_track",
        # "track_05": bench_folder + "/05_track",
    }

    configuration_list = []
    for solver_param in solver_param_list:
        solver = solver_param[0]
        parameters_list = solver_param[1]

        for benchmark_name, benchmark_folder in benchmark_dict.items():
            if len(parameters_list)>1:
                graph_type=parameters_list[1].replace("--graph_type ","")
            else:
                graph_type="graph_1"
            configuration_list.append({"solver":solver,"parameters_list":parameters_list,"benchmark_name":benchmark_name,
                                       "benchmark_folder":benchmark_folder})

    # Writing the dictionary to a JSON file
    configuration_folder = project_folder + "/src/process_benchmarks/configurations"
    write_configurations_to_json_file(configuration_folder=configuration_folder,configurations=configuration_list)






if __name__ == '__main__':
    main()
