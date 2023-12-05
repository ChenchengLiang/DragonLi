from src.solver.Constants import shell_timeout, solver_command_map,project_folder
import os
import time
import subprocess
from src.solver.Constants import UNKNOWN, SAT, UNSAT
from src.solver.independent_utils import strip_file_name_suffix
import csv
from typing import List, Dict, Tuple
import glob
from src.solver.Constants import INTERNAL_TIMEOUT, BRANCH_CLOSED, MAX_PATH_REACHED, RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR,RED,GREEN,COLORRESET
import random
from src.solver.independent_utils import mean,check_list_consistence


def run_on_one_track(benchmark_name: str, benchmark_folder: str, parameters_list, solver, suffix_dict, summary_folder_name,
                     solver_log: bool = False):
    track_result_list = []

    file_list = glob.glob(benchmark_folder + "/*" + suffix_dict[solver])
    file_list_num = len(file_list)
    for i, file in enumerate(file_list):
        print("processing progress:", i, "/", file_list_num)
        result_dict = run_on_one_problem(file, parameters_list, solver, solver_log=solver_log)
        track_result_list.append(
            (os.path.basename(file), result_dict["result"], result_dict["used_time"], result_dict["split_number"]))

    result_summary_dict = result_summary(track_result_list)
    write_to_cvs_file(track_result_list, result_summary_dict, benchmark_name, solver, parameters_list,summary_folder_name)


def result_summary(track_result_list: List[Tuple[str, str, float,float]]):
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


def write_to_cvs_file(track_result_list: List[Tuple[str, str,float, float]], summary_dict: Dict, benchmark_name: str,
                      solver: str, parameters_list: List[str],summary_folder_name):
    summary_folder = project_folder + "/src/process_benchmarks/summary/"+summary_folder_name
    if os.path.exists(summary_folder) == False:
        os.mkdir(summary_folder)
    # Name of the CSV file to write to
    if len(parameters_list)>1:
        parameters_str_list=[parameters_list[0]] + [parameters_list[1].replace("--graph_type ", "")] + [parameters_list[2][parameters_list[2].rfind("_")+1:parameters_list[2].rfind(".")]]
    elif len(parameters_list)==1:
        parameters_str_list = [parameters_list[0]]
    else:
        parameters_str_list=[]
    parameters_list_str = "_".join(parameters_str_list)


    if parameters_list_str == "":
        summary_name = solver + "_" + benchmark_name + "_summary.csv"
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
            csvwriter.writerow([track_result_list[0][0], track_result_list[0][1], track_result_list[0][2],track_result_list[0][3]] + list(summary_dict.values()))
        else:
            csvwriter.writerow(["File Name", "Result", "Used Time", "split_number", ] + list(summary_dict.keys()))
            csvwriter.writerow([track_result_list[0][0], track_result_list[0][1], track_result_list[0][2],"0"] + list(summary_dict.values()))

        # Writing the following rows
        csvwriter.writerows(track_result_list[1:])



def run_on_one_problem(file_path:str, parameters_list:List[str], solver:str, solver_log:bool=False):
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
    random_integer = random.randint(1, 100000)
    shell_file_name = "run-" + os.path.basename(file_path)+"-" +str(random_integer)+ ".sh"
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
    # print(completed_process)
    # print("Output from script:", completed_process.stdout)
    result_dict = process_solver_output(completed_process.stdout, problem_file_path, solver,log=log)
    result_dict["used_time"] = used_time
    print("Finished", "use time: ", used_time)
    return result_dict


def process_solver_output(solver_output: str, problem_file_path: str, solver:str,log:bool=False):
    result = UNKNOWN
    split_number = 0

    if solver == "this":
        lines:List[str] = solver_output.split('\n')
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

    satisfiability_dict = {}

    for solver, summary_file in summary_file_dict.items():
        #first_summary_solver_row.extend([solver, solver])

        first_summary_solver_row,reconstructed_list_title, reconstructed_list, reconstructed_summary_title, reconstructed_summary_data = extract_one_csv_data(summary_folder,
            summary_file,first_summary_solver_row,solver)
        first_summary_title_row.extend(reconstructed_list_title[1:])
        if len(first_summary_data_rows) == 0:
            first_summary_data_rows = [[] for x in reconstructed_list]


        #print("solver",solver)
        #print(reconstructed_list)
        for f, r in zip(first_summary_data_rows, reconstructed_list):
            for i,x in enumerate(r):
                if x=="":
                   r[i]=0
            if len(f)==0:
                f.extend(r)
                satisfiability_dict[strip_file_name_suffix(r[0])] = []
                satisfiability_dict[strip_file_name_suffix(r[0])].append((solver, r[1]))
            else:
                file_name_1 = strip_file_name_suffix(f[0])
                for rr in reconstructed_list:
                    rr=[x for x in rr if x!=""]
                    file_name_2 = strip_file_name_suffix(rr[0])
                    if file_name_1 == file_name_2:
                        f.extend(rr[1:])
                        satisfiability_dict[strip_file_name_suffix(rr[0])].append((solver, rr[1]))



        if len(second_summary_title_row) == 1:
            second_summary_title_row.extend(reconstructed_summary_title)

        second_summary_data_rows.append([solver] + reconstructed_summary_data)




    compute_split_number_for_common_solved_problems(first_summary_data_rows, first_summary_title_row,
                                                    first_summary_solver_row, second_summary_title_row,
                                                    second_summary_data_rows)

    ################### check satisfiability consistensy between solvers########################

    print("----------------------- check satisfiability consistensy ----------------------------")


    print(f"row numebr: {len(satisfiability_dict)}")
    consistent_list=[]
    inconsistent_list=[]

    for k, v in satisfiability_dict.items():
        #print(k, v)
        solved = []
        for vv in v:
            if vv[1]!=UNKNOWN:
                solved.append(vv)
        if len(solved) > 1:
            satisfiability_list=[]
            for s in solved:
                satisfiability_list.append(s[1])

            consistence=check_list_consistence(satisfiability_list)

            if consistence==False:
                #print(RED,"inconsitensy",COLORRESET,k,solved)
                inconsistent_list.append((k,solved))
            else:
                #print(GREEN,"consitensy",COLORRESET,k,solved)
                consistent_list.append((k,solved))

    print(f"number of consistent problems: {len(consistent_list)}")
    print(f"number of inconsistent problems: {len(inconsistent_list)}")
    if len(inconsistent_list)>0:
        print("inconsistent problems:")
        for i in inconsistent_list:
            print(i)

    print("----------------------- check satisfiability consistensy done ----------------------------")




    #################### write to csv ########################

    summary_path = os.path.join(summary_folder, track_name+"_reconstructed_summary_1.csv")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    # Writing to csv file
    with open(summary_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(first_summary_title_row)
        csvwriter.writerow(first_summary_solver_row)

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



def compute_split_number_for_common_solved_problems(first_summary_data_rows,first_summary_title_row,first_summary_solver_row,second_summary_title_row,second_summary_data_rows):
    # compute sat_average_split_number for commonly solved problem
    # find common solved problems
    common_problem_list = []
    for row in first_summary_data_rows:
        result_count = 0
        sat_configuration = 0
        file_name = row[0]
        for measurement, solver, value in zip(first_summary_title_row, first_summary_solver_row, row):
            if measurement == "Result":
                result_count += 1
                if value == SAT:
                    sat_configuration += 1
        if result_count == sat_configuration:
            common_problem_list.append(file_name)

    # compute sat_average_split_number_common_solved
    sat_average_split_number_common_solved_dict = {solver_dict[0]: [] for solver_dict in second_summary_data_rows}
    for row in first_summary_data_rows:
        file_name = row[0]
        if file_name in common_problem_list:
            for measurement, solver, value in zip(first_summary_title_row, first_summary_solver_row, row):
                if measurement == "split_number":
                    sat_average_split_number_common_solved_dict[solver].append(value)

    # write to summary 2 file
    second_summary_title_row +=  ["sat_average_split_number_common_solved " + str(len(common_problem_list))]
    for summary_row in second_summary_data_rows:
        summary_row.append(mean([int(x) for x in sat_average_split_number_common_solved_dict[summary_row[0]]]))


def extract_one_csv_data(summary_folder,summary_file,first_summary_solver_row,solver):
    first_summary_solver_row.extend([solver, solver, solver])
    column_index = 4

    # if "this" in solver:
    #     first_summary_solver_row.extend([solver, solver, solver])
    #     column_index = 4
    # else:
    #     first_summary_solver_row.extend([solver, solver])
    #     column_index = 3

    summary_path = os.path.join(summary_folder+"/to_summary/", summary_file)
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
            if len(split_number_list)!=0:
                sat_average_split_number = sum([int(x) for x in split_number_list])/len(split_number_list)
            else:
                sat_average_split_number=0
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

