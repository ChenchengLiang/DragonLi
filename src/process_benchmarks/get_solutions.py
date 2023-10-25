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
from utils import run_on_one_benchmark
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


def run_on_one_track(benchmark_name: str, benchmark_folder: str, parameters_list, solver, suffix_dict,
                     solver_log: bool = False):
    track_result_list = []

    file_list = glob.glob(benchmark_folder + "/*" + suffix_dict[solver])
    file_list_num = len(file_list)
    for i, file in enumerate(file_list):
        print("processing progress:", i, "/", file_list_num)
        result_dict = run_on_one_benchmark(file, parameters_list, solver, solver_log=solver_log)
        track_result_list.append(
            (os.path.basename(file), result_dict["result"], result_dict["used_time"], result_dict["split_number"]))

    result_summary_dict = result_summary(track_result_list)
    write_to_cvs_file(track_result_list, result_summary_dict, benchmark_name, solver, parameters_list)


def result_summary(track_result_list: List[Tuple[str, str, float]]):
    SAT_count = [entry[1] for entry in track_result_list].count("SAT")
    UNSAT_count = [entry[1] for entry in track_result_list].count("UNSAT")
    UNKNOWN_count = [entry[1] for entry in track_result_list].count("UNKNOWN")
    MAX_VARIABLE_LENGTH_EXCEEDED_count = [entry[1] for entry in track_result_list].count("MAX VARIABLE LENGTH EXCEEDED")
    INTERNAL_TIMEOUT_count = [entry[1] for entry in track_result_list].count(INTERNAL_TIMEOUT)
    BRANCH_CLOSED_count = [entry[1] for entry in track_result_list].count(BRANCH_CLOSED)
    ERROR_count = [entry[1] for entry in track_result_list].count("ERROR")
    MAX_PATH_REACHED_count = [entry[1] for entry in track_result_list].count(MAX_PATH_REACHED)
    RECURSION_DEPTH_EXCEEDED_count = [entry[1] for entry in track_result_list].count(RECURSION_DEPTH_EXCEEDED)
    RECURSION_ERROR_count = [entry[1] for entry in track_result_list].count(RECURSION_ERROR)

    return {"SAT": SAT_count, "UNSAT": UNSAT_count, "UNKNOWN": UNKNOWN_count, "ERROR": ERROR_count,
            INTERNAL_TIMEOUT: INTERNAL_TIMEOUT_count,
            "MAX_VARIABLE_LENGTH_EXCEEDED": MAX_VARIABLE_LENGTH_EXCEEDED_count,
            BRANCH_CLOSED: BRANCH_CLOSED_count, MAX_PATH_REACHED: MAX_PATH_REACHED_count,
            RECURSION_DEPTH_EXCEEDED: RECURSION_DEPTH_EXCEEDED_count, RECURSION_ERROR: RECURSION_ERROR_count,
            "Total": len(track_result_list)}


def write_to_cvs_file(track_result_list: List[Tuple[str, str, float]], summary_dict: Dict, benchmark_name: str,
                      solver: str, parameters_list: List[str]):
    summary_folder = project_folder + "/src/process_benchmarks/summary"
    # Name of the CSV file to write to
    parameters_list = [x.replace("--graph_type ", "") for x in parameters_list]
    parameters_list_str = "_".join(parameters_list)
    if parameters_list_str == "":
        summary_name = solver + "_" + benchmark_name + + "_summary.csv"
    else:
        summary_name = solver + "_" + parameters_list_str + "_" + benchmark_name + "_summary.csv"
    summary_path = os.path.join(summary_folder, summary_name)

    if os.path.exists(summary_path):
        os.remove(summary_path)

    # Writing to csv file
    with open(summary_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Writing the column headers and first row with summary_dict
        if solver == "this":

            csvwriter.writerow(["File Name", "Result", "Used Time", "split_number"] + list(summary_dict.keys()))
            csvwriter.writerow([track_result_list[0][0], track_result_list[0][1], track_result_list[0][2],
                                track_result_list[0][3]] + list(summary_dict.values()))
        else:
            csvwriter.writerow(["File Name", "Result", "Used Time", "", ] + list(summary_dict.keys()))
            csvwriter.writerow([track_result_list[0][0], track_result_list[0][1], track_result_list[0][2],
                                ""] + list(summary_dict.values()))

        # Writing the following rows
        csvwriter.writerows(track_result_list[1:])


if __name__ == '__main__':
    main()
