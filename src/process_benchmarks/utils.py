from src.solver.Constants import shell_timeout
import os
import time
import subprocess


def run_on_one_benchmark(file_path):
    # create a shell file to run the main_parameter.py
    shell_file_path = create_a_shell_file(file_path)

    # run the shell file
    result, used_time = run_a_shell_file(shell_file_path,file_path)

    # delete the shell file
    if os.path.exists(shell_file_path):
        os.remove(shell_file_path)

    return result, used_time


def create_a_shell_file(file_path, parameter_list=""):
    shell_folder = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/process_benchmarks/temp_shell"
    shell_file_name = "run-" + os.path.basename(file_path) + ".sh"
    shell_file_path = os.path.join(shell_folder, shell_file_name)
    timeout_command = "timeout " + str(shell_timeout)
    script_location = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/process_benchmarks/main_parameter.py"
    if os.path.exists(shell_file_path):
        os.remove(shell_file_path)
    with open(shell_file_path, 'w') as file:
        file.write("#!/bin/sh\n")
        file.write(timeout_command + " python3 " + script_location + " " + file_path + " " + parameter_list + "\n")
    return shell_file_path


def run_a_shell_file(shell_file_path:str,problem_file_path:str):
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
    result = process_solver_output(completed_process.stdout,problem_file_path)
    print("Finished", "use time: ", used_time)
    return result, used_time


def process_solver_output(solver_output:str,problem_file_path:str):
    result = "UNKNOWN"
    lines = solver_output.split('\n')
    for line in lines:
        if "result:" in line:
            result = line.split("result:")[1].strip(" ")
        #print(line)

    #write to log file
    if result == "SAT" or result == "UNSAT":
        log_file=problem_file_path+".log"
        if os.path.exists(log_file):
            os.remove(log_file)
        with open(log_file, 'w') as file:
            file.write(solver_output)

    return result
