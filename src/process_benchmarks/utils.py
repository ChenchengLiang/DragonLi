from src.solver.Constants import shell_timeout, solver_command_map,project_folder
import os
import time
import subprocess
from src.solver.Constants import UNKNOWN, SAT, UNSAT
from src.solver.independent_utils import strip_file_name_suffix
import csv
from typing import List, Dict, Tuple

def run_on_one_benchmark(file_path:str, parameters_list:List[str], solver:str,solver_log:bool=False):
    # create a shell file to run the main_parameter.py
    shell_file_path = create_a_shell_file(file_path, parameters_list, solver)

    # run the shell file
    result_dict = run_a_shell_file(shell_file_path, file_path, solver,log=solver_log)

    # delete the shell file
    if os.path.exists(shell_file_path):
        os.remove(shell_file_path)

    return result_dict


def create_a_shell_file(file_path, parameter_list="", solver=""):
    parameter_str = " ".join(parameter_list)
    shell_folder = project_folder+"/src/process_benchmarks/temp_shell"
    shell_file_name = "run-" + os.path.basename(file_path) + ".sh"
    shell_file_path = os.path.join(shell_folder, shell_file_name)
    timeout_command = "timeout " + str(shell_timeout)

    solver_command = solver_command_map[solver]
    if os.path.exists(shell_file_path):
        os.remove(shell_file_path)
    with open(shell_file_path, 'w') as file:
        file.write("#!/bin/sh\n")
        file.write(timeout_command + " " + solver_command + " " + file_path + " " + parameter_str + "\n")
        print("run command:",timeout_command + " " + solver_command + " " + file_path + " " + parameter_str)
    return shell_file_path


def run_a_shell_file(shell_file_path: str, problem_file_path: str, solver:str,log:bool=False):
    print("-" * 10)
    print("run " + shell_file_path)
    run_shell_command = ["sh", shell_file_path]
    start = time.time()

    completed_process = subprocess.run(run_shell_command, capture_output=True, text=True, shell=False)
    # eld = subprocess.Popen(run_shell_command, stdout=subprocess.DEVNULL, shell=False)
    # eld.wait()
    end = time.time()
    used_time = end - start
    # print("Output from script:", completed_process.stdout)
    result_dict = process_solver_output(completed_process.stdout, problem_file_path, solver,log=log)
    result_dict["used_time"] = used_time
    print("Finished", "use time: ", used_time)
    return result_dict


def process_solver_output(solver_output: str, problem_file_path: str, solver:str,log:bool=False):
    result = UNKNOWN
    split_number = 0

    if solver == "this":
        lines = solver_output.split('\n')
        for line in lines:
            if "result:" in line:
                result = line.split("result:")[1].strip(" ")
            if "Total explore_paths call:" in line:
                split_number = line.split("Total explore_paths call:")[1].strip(" ")
            # print(line)

    elif solver == "woorpje":
        if "Found a solution" in solver_output:
            result = SAT
        elif "Equation has no solution due to set bounds" in solver_output:
            result = UNSAT

    elif solver == "z3":
        lines = solver_output.split('\n')
        if lines[0] == "sat":
            result = SAT
        elif lines[0] == "unsat":
            result = UNSAT

    elif solver == "ostrich":
        lines = solver_output.split('\n')
        if lines[0] == "sat":
            result = SAT
        elif lines[0] == "unsat":
            result = UNSAT

    elif solver == "cvc5":
        lines = solver_output.split('\n')
        if lines[0] == "sat":
            result = SAT
        elif lines[0] == "unsat":
            result = UNSAT

    # write to log file
    if log==True and (result == SAT or result == UNSAT):
        log_file = problem_file_path + "." + solver + ".log"
        if os.path.exists(log_file):
            os.remove(log_file)
        with open(log_file, 'w') as file:
            file.write(solver_output)

    # update answer file
    answer_file = strip_file_name_suffix(problem_file_path) + ".answer"
    if os.path.exists(answer_file):
        # read the answer file
        with open(answer_file, 'r') as file:
            answer = file.read()
        # update the answer file if there is no sound answer
        if answer != SAT and answer != UNSAT:
            with open(answer_file, 'w') as file:
                file.write(result)
        else:
            pass

    else:
        # create the answer file
        with open(answer_file, 'w') as file:
            file.write(result)
    result_dict={"result":result,"split_number":split_number,"solver":solver}
    return result_dict





def summary_one_track(summary_folder,summary_file_dict,track_name):
    first_summary_solver_row = ["file_names"]
    first_summary_title_row = [""]
    first_summary_data_rows = []

    second_summary_title_row = ["solver"]
    second_summary_data_rows = []

    for solver, summary_file in summary_file_dict.items():
        #first_summary_solver_row.extend([solver, solver])

        first_summary_solver_row,reconstructed_list_title, reconstructed_list, reconstructed_summary_title, reconstructed_summary_data = extract_one_csv_data(summary_folder,
            summary_file,first_summary_solver_row,solver)
        first_summary_title_row.extend(reconstructed_list_title[1:])
        if len(first_summary_data_rows) == 0:
            first_summary_data_rows = [[] for x in reconstructed_list]


        # print("solver",solver)
        # print(reconstructed_list)
        for f, r in zip(first_summary_data_rows, reconstructed_list):
            if len(f)==0:
                f.extend(r)
            else:
                file_name_1 = strip_file_name_suffix(f[0])
                for rr in reconstructed_list:
                    rr=[x for x in rr if x!=""]
                    file_name_2 = strip_file_name_suffix(rr[0])
                    if file_name_1 == file_name_2:
                        f.extend(rr[1:])



        if len(second_summary_title_row) == 1:
            second_summary_title_row.extend(reconstructed_summary_title)

        second_summary_data_rows.append([solver] + reconstructed_summary_data)


    ############################################

    summary_path = os.path.join(summary_folder, track_name+"_reconstructed_summary_1.csv")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    # Writing to csv file
    with open(summary_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Writing the column solvers
        csvwriter.writerow(first_summary_solver_row)
        # Writing the column headers
        csvwriter.writerow(first_summary_title_row)

        for row in first_summary_data_rows:
            csvwriter.writerow(row)

    summary_path = os.path.join(summary_folder, track_name+"_reconstructed_summary_2.csv")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    # Writing to csv file
    with open(summary_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(second_summary_title_row)
        csvwriter.writerows(second_summary_data_rows)


def extract_one_csv_data(summary_folder,summary_file,first_summary_solver_row,solver):

    if "this" in solver:
        first_summary_solver_row.extend([solver, solver, solver])
        column_index = 4
    else:
        first_summary_solver_row.extend([solver, solver])
        column_index = 3

    summary_path = os.path.join(summary_folder, summary_file)
    with open(summary_path, 'r') as file:
        reader = csv.reader(file)
        reader = list(reader)
        reconstructed_first_row = reader[1][:column_index]
        reconstructed_list_title = reader[0][:column_index]
        reconstructed_list = [reconstructed_first_row] + reader[2:]

        reconstructed_summary_title = reader[0][column_index:] + ["sat_average_split_number"]
        reconstructed_summary_data = reader[1][column_index:]



        if "this" in solver:
            #compute average split number for SAT problems
            split_number_list = []
            for row in reader:
                if row[1]=="SAT":
                    split_number_list.append(row[3])
            sat_average_split_number = sum([int(x) for x in split_number_list])/len(split_number_list)
        else:
            sat_average_split_number = 0

        reconstructed_summary_data=reconstructed_summary_data+[sat_average_split_number]

        # print(reconstructed_summary_title)
        # print(reconstructed_summary_data)



        # print(reconstructed_list_title)
        # for row in reconstructed_list:
        #     print(row)  # Each row is a list of strings

        # print(reconstructed_summary_title)
        # print(reconstructed_summary_data)

        return first_summary_solver_row,reconstructed_list_title,reconstructed_list,reconstructed_summary_title,reconstructed_summary_data

