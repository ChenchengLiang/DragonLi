import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.Constants import project_folder, bench_folder, SAT, UNSAT, UNKNOWN, summary_folder
from src.solver.independent_utils import strip_file_name_suffix, create_folder
import shutil
import glob
import csv


def main():
    # collect_answers_from_divided_folders(benchmark="03_track_train_task_3_merged_1_40000")

    

    collect_answers_from_summary_cvs(benchmark="03_track_train_task_3_5001_1000",
                                     cvs_file="this_fixed_execute_termination_condition_0_03_track_train_task_3_1_5000_summary.csv")
    collect_answers_from_summary_cvs(benchmark="03_track_train_task_3_10001_15000",
                                     cvs_file="this_fixed_execute_termination_condition_0_03_track_train_task_3_1_5000_summary.csv")
    collect_answers_from_summary_cvs(benchmark="03_track_train_task_3_15001_20000",
                                     cvs_file="this_fixed_execute_termination_condition_0_03_track_train_task_3_1_5000_summary.csv")
    collect_answers_from_summary_cvs(benchmark="03_track_train_task_3_20001_250000",
                                     cvs_file="this_fixed_execute_termination_condition_0_03_track_train_task_3_1_5000_summary.csv")
    collect_answers_from_summary_cvs(benchmark="03_track_train_task_3_25001_30000",
                                     cvs_file="this_fixed_execute_termination_condition_0_03_track_train_task_3_1_5000_summary.csv")
    collect_answers_from_summary_cvs(benchmark="03_track_train_task_3_30001_35000",
                                     cvs_file="this_fixed_execute_termination_condition_0_03_track_train_task_3_1_5000_summary.csv")
    collect_answers_from_summary_cvs(benchmark="03_track_train_task_3_35001_40000",
                                     cvs_file="this_fixed_execute_termination_condition_0_03_track_train_task_3_1_5000_summary.csv")



def collect_answers_from_summary_cvs(benchmark, cvs_file):
    data_folder = benchmark.replace("_summary", "")
    collect_answers_from_divided_folders(benchmark=data_folder)
    all_data_folder = bench_folder + "/" + data_folder + "/ALL/ALL"

    # collect answers from summary cvs
    one_solver_cvs = summary_folder + "/" + benchmark + "/" + cvs_file

    result_folder = create_folder(
        bench_folder + "/" + data_folder + "/this_fixed_execute_termination_condition_0_03_track_train_task_3_1_5000_summary")
    sat_folder = create_folder(result_folder + "/SAT")
    unsat_folder = create_folder(result_folder + "/UNSAT")
    unknown_folder = create_folder(result_folder + "/UNKNOWN")

    with open(one_solver_cvs, 'r') as file:
        reader = csv.reader(file)
        reader = list(reader)
        file_name_column_index = 0
        satisifiability_column_index = 1
        for row in reader:
            file_name = row[file_name_column_index]
            file_path_without_suffix = strip_file_name_suffix(all_data_folder + "/" + file_name)
            if row[satisifiability_column_index] == "SAT":
                shutil.copy(file_path_without_suffix + ".eq", sat_folder)
                shutil.copy(file_path_without_suffix + ".answer", sat_folder)
                shutil.copy(file_path_without_suffix + ".smt2", sat_folder)

            elif row[satisifiability_column_index] == UNSAT:
                shutil.copy(file_path_without_suffix + ".eq", unsat_folder)
                shutil.copy(file_path_without_suffix + ".answer", unsat_folder)
                shutil.copy(file_path_without_suffix + ".smt2", unsat_folder)
            elif row[satisifiability_column_index] == UNKNOWN:
                shutil.copy(file_path_without_suffix + ".eq", unknown_folder)
                shutil.copy(file_path_without_suffix + ".answer", unknown_folder)
                shutil.copy(file_path_without_suffix + ".smt2", unknown_folder)

    print("done")


def collect_answers_from_divided_folders(benchmark):
    # collect answers from divided folders
    benchmark_folder = bench_folder + "/" + benchmark + "/ALL"

    folder_number = sum([1 for fo in os.listdir(benchmark_folder) if "divided" in os.path.basename(fo)])
    for i in range(folder_number):
        divided_folder_index = i + 1
        for a in glob.glob(benchmark_folder + "/divided_" + str(divided_folder_index) + "/*.answer"):
            # print(a)
            shutil.copy(a, benchmark_folder + "/ALL")

    # separate to SAT UNSAT UNKNOWN
    benchmark_folder = bench_folder + "/" + benchmark

    # create folders
    sat_folder = benchmark_folder + "/SAT"
    unsat_folder = benchmark_folder + "/UNSAT"
    unknown_folder = benchmark_folder + "/UNKNOWN"
    for folder in [sat_folder, unsat_folder, unknown_folder]:
        if os.path.exists(folder) == False:
            os.mkdir(folder)

    # separate files according to answers
    for file in glob.glob(benchmark_folder + "/ALL/ALL/*.eq"):
        file_name = strip_file_name_suffix(file)
        # read .answer file
        answer_file = file_name + ".answer"
        if os.path.exists(answer_file):
            with open(answer_file) as f:
                answer = f.read()
                if answer == SAT:
                    shutil.copy(file, sat_folder)
                    shutil.copy(answer_file, sat_folder)
                elif answer == UNSAT:
                    shutil.copy(file, unsat_folder)
                    shutil.copy(answer_file, unsat_folder)
                else:
                    shutil.copy(file, unknown_folder)
                    shutil.copy(answer_file, unknown_folder)
        else:
            print("error: answer file does not exist")
            exit(1)

    # remove original files
    # for file in glob.glob(benchmark_folder+"/*.eq"):
    #     if os.path.exists(file):
    #         shutil.copy(file,all_folder)
    #         os.remove(file)
    # for file in glob.glob(benchmark_folder + "/*.answer"):
    #     if os.path.exists(file):
    #         shutil.copy(file,all_folder)
    #         os.remove(file)

    print("done")


if __name__ == '__main__':
    main()
