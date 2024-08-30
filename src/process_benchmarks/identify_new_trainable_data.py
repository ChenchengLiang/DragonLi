import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

import csv
import glob
import os
from src.solver.independent_utils import strip_file_name_suffix
from src.process_benchmarks.utils import summary_one_track
from src.solver.Constants import project_folder,bench_folder
import argparse



def main():
    benchmark_name=f"01_track_multi_word_equations_generated_eval_eq_number_1_rank_task_1_200"
    merged_summary_folder = project_folder+f"/src/process_benchmarks/summary/merge_summary/{benchmark_name}_summary"

    target_solver="z3"
    summary_file_target_solver=f"{target_solver}_{benchmark_name}_summary.csv"
    summary_file_this_solver="this_fixed_termination_condition_0_category_01_track_multi_word_equations_generated_eval_eq_number_1_rank_task_1_200_summary.csv"

    # read from summary file of target solver
    with open(f"{merged_summary_folder}/{summary_file_target_solver}", 'r') as file:
        target_solver_summary = list(csv.reader(file))
    # read from summary file of this solver
    with open(f"{merged_summary_folder}/{summary_file_this_solver}", 'r') as file:
        this_solver_summary = list(csv.reader(file))


    # remove head
    target_solver_summary=target_solver_summary[1:]
    this_solver_summary=this_solver_summary[1:]
    # sort the rows by the first column
    target_solver_summary=sorted(target_solver_summary, key=lambda x: x[0])
    this_solver_summary=sorted(this_solver_summary, key=lambda x: x[0])



    for this_solver_row, target_solver_row in zip(this_solver_summary, target_solver_summary):
        strip_file_name_suffix(this_solver_row)
        print(this_solver_row, target_solver_row)



if __name__ == '__main__':
    main()
