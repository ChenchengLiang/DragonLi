'''
This script is used to call this string solver to get the solutions from the Woorpje benchmark and write them back to the benchmark file.

'''
import glob
import os.path
from typing import List, Tuple
from utils import run_on_one_benchmark
import csv


def main():
    example_track = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/examples"
    track_01="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje"

    run_on_one_track(example_track)


def run_on_one_track(benchmark_folder:str):
    track_result_list = []
    file_list = glob.glob(benchmark_folder + "/*.eq")
    for file in file_list:
        result, used_time = run_on_one_benchmark(file)
        track_result_list.append((os.path.basename(file), result, used_time))

    result_summary_dict = result_summary(track_result_list)
    write_to_cvs_file(track_result_list,result_summary_dict)

def result_summary(track_result_list:List[Tuple[str,str,float]]):
    SAT_count = [entry[1] for entry in track_result_list].count("SAT")
    UNSAT_count = [entry[1] for entry in track_result_list].count("UNSAT")
    UNKNOWN_count = [entry[1] for entry in track_result_list].count("UNKNOWN")
    MAX_VARIABLE_LENGTH_EXCEEDED_count = [entry[1] for entry in track_result_list].count("MAX VARIABLE LENGTH EXCEEDED")
    INTERNAL_TIMEOUT_count = [entry[1] for entry in track_result_list].count("INTERNAL TIMEOUT")

    return {"SAT": SAT_count, "UNSAT": UNSAT_count, "UNKNOWN": UNKNOWN_count, "ERROR": 0,"INTERNAL_TIMEOUT":INTERNAL_TIMEOUT_count, "MAX_VARIABLE_LENGTH_EXCEEDED":MAX_VARIABLE_LENGTH_EXCEEDED_count, "Total": len(track_result_list)}

def write_to_cvs_file(track_result_list:List[Tuple[str,str,float]],summary_dict):
    summary_folder = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/process_benchmarks/summary"
    # Name of the CSV file to write to
    summary_name = "summary.csv"
    summary_path = os.path.join(summary_folder, summary_name)

    if os.path.exists(summary_path):
        os.remove(summary_path)

    # Writing to csv file
    with open(summary_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Writing the column headers
        csvwriter.writerow(["File Name", "Result", "Used Time", "", "SAT", "UNSAT", "UNKNOWN", "TIMEOUT", "ERROR", "MAX_VARIABLE_LENGTH_EXCEEDED","Total"])

        # Writing first row with summary_dict
        csvwriter.writerow([track_result_list[0][0],track_result_list[0][1],track_result_list[0][2],
                            "", summary_dict["SAT"], summary_dict["UNSAT"], summary_dict["UNKNOWN"], summary_dict["ERROR"],summary_dict["INTERNAL_TIMEOUT"],summary_dict["MAX_VARIABLE_LENGTH_EXCEEDED"], summary_dict["Total"]])

        # Writing the following rows
        csvwriter.writerows(track_result_list[1:])


if __name__ == '__main__':
    main()
