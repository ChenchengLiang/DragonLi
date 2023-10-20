import glob
import os.path
from typing import List, Tuple, Dict
from utils import run_on_one_benchmark
import csv
from src.solver.Constants import BRANCH_CLOSED, MAX_PATH_REACHED, INTERNAL_TIMEOUT, RECURSION_DEPTH_EXCEEDED, \
    RECURSION_ERROR
from src.process_benchmarks.utils import summary_one_track


def main():
    # solver = "woorpje"
    # solver = "this"
    # solver = "z3"
    # solver = "ostrich"
    # solver = "cvc5"
    for solver in ["woorpje"]:

        suffix_dict = {"z3": ".smt", "woorpje": ".eq", "this": ".eq", "ostrich": ".smt2", "cvc5": ".smt2"}

        test_track = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/test"
        example_track = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/examples"
        track_01 = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track"
        g_track_01 = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track_generated"
        track_02 = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/02_track"
        track_03 = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/03_track"
        track_04 = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/04_track"
        track_05 = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/05_track"

        parameters_list = []

        benchmark_dict = {
            #"test_track":test_track,
            # "example_track":example_track,
            "track_01": track_01,
            "g_track_01":g_track_01,
            # "track_02": track_02,
            # "track_03": track_03,
            # "track_04": track_04,
            # "track_05": track_05
        }
        for benchmark_name, benchmark_folder in benchmark_dict.items():
            run_on_one_track(benchmark_name, benchmark_folder, parameters_list, solver, suffix_dict)


    # summary
    # summary_folder = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/process_benchmarks/summary"
    #
    # for track in benchmark_dict.keys():
    #     summary_file_dict = {"this": "this_" + track + "_summary.csv",
    #                          "woorpje": "woorpje_" + track + "_summary.csv",
    #                          "z3": "z3_" + track + "_summary.csv",
    #                          "ostrich": "ostrich_" + track + "_summary.csv",
    #                          "cvc5": "cvc5_" + track + "_summary.csv"}
    #
    #     summary_one_track(summary_folder, summary_file_dict, track)


def run_on_one_track(benchmark_name: str, benchmark_folder: str, parameters_list, solver, suffix_dict):
    track_result_list = []

    file_list = glob.glob(benchmark_folder + "/*" + suffix_dict[solver])
    for file in file_list:
        result, used_time = run_on_one_benchmark(file, parameters_list, solver)
        track_result_list.append((os.path.basename(file), result, used_time))

    result_summary_dict = result_summary(track_result_list)
    write_to_cvs_file(track_result_list, result_summary_dict, benchmark_name, solver)


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


def write_to_cvs_file(track_result_list: List[Tuple[str, str, float]], summary_dict: Dict, benchmark_name: str, solver):
    summary_folder = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/process_benchmarks/summary"
    # Name of the CSV file to write to
    summary_name = solver + "_" + benchmark_name + "_summary.csv"
    summary_path = os.path.join(summary_folder, summary_name)

    if os.path.exists(summary_path):
        os.remove(summary_path)

    # Writing to csv file
    with open(summary_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Writing the column headers
        csvwriter.writerow(["File Name", "Result", "Used Time", "", ] + list(summary_dict.keys()))

        # Writing first row with summary_dict
        csvwriter.writerow([track_result_list[0][0], track_result_list[0][1], track_result_list[0][2],
                            ""] + list(summary_dict.values()))

        # Writing the following rows
        csvwriter.writerows(track_result_list[1:])


if __name__ == '__main__':
    main()
