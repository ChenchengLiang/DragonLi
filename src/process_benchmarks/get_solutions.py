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
import csv
from src.solver.Constants import bench_folder, BRANCH_CLOSED, MAX_PATH_REACHED, INTERNAL_TIMEOUT, RECURSION_DEPTH_EXCEEDED, \
    RECURSION_ERROR, project_folder
from src.process_benchmarks.utils import summary_one_track


def main():
    solver_log = False
    suffix_dict = {"z3": ".smt", "woorpje": ".eq", "this": ".eq", "ostrich": ".smt2", "cvc5": ".smt2"}

    solver_param_list = [
        ["this", ["fixed"]],
        #["this", ["random"]],
        # ["this",["gnn","--graph_type graph_1"]],
        # ["this",["gnn","--graph_type graph_2"]],
        # ["woorpje",[]],
        # ["z3",[]],
        # ["ostrich",[]],
        # ["cvc5",[]],
    ]

    test_track = bench_folder + "/test"
    example_track = bench_folder + "/examples"
    track_01 = bench_folder + "/01_track"
    g_track_01_sat = bench_folder + "/01_track_generated/SAT"
    g_track_01_mixed = bench_folder + "/01_track_generated/mixed"
    g_track_01_eval = bench_folder + "/01_track_generated_eval_data"
    track_02 = bench_folder + "/02_track"
    track_03 = bench_folder + "/03_track"
    track_04 = bench_folder + "/04_track"
    track_05 = bench_folder + "/05_track"

    benchmark_dict = {
        #"test_track": test_track,
        "example_track": example_track,
        # "track_01": track_01,
        # "g_track_01_sat":g_track_01_sat,
        # "g_track_01_mixed": g_track_01_mixed,
        # "g_track_01_eval":g_track_01_eval,
        # "track_02": track_02,
        # "track_03": track_03,
        # "track_04": track_04,
        # "track_05": track_05
    }

    for solver_param in solver_param_list:
        solver = solver_param[0]
        parameters_list = solver_param[1]
        print("solver:", solver, "parameters_list:", parameters_list)

        for benchmark_name, benchmark_folder in benchmark_dict.items():
            run_on_one_track(benchmark_name, benchmark_folder, parameters_list, solver, suffix_dict,
                             solver_log=solver_log)

    # Symmary
    summary_folder = project_folder + "/src/process_benchmarks/summary"
    # summary one cross tracks
    for track in benchmark_dict.keys():
        summary_file_dict = {}
        for solver_param in solver_param_list:
            k = solver_param[0]
            v = solver_param[1]
            v = [i.replace("--graph_type ", "") for i in v]
            parammeters_str = "_".join(v)
            summary_file_dict[k + ":" + parammeters_str] = k + "_" + parammeters_str + "_" + track + "_summary.csv"

        summary_one_track(summary_folder, summary_file_dict, track)




if __name__ == '__main__':
    main()
