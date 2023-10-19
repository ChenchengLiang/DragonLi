from src.solver.Constants import shell_timeout, solver_command_map
import os
import time
import subprocess
from src.solver.Constants import UNKNOWN, SAT, UNSAT
from src.solver.independent_utils import strip_file_name_suffix


def run_on_one_benchmark(file_path, parameters_list, solver):
    # create a shell file to run the main_parameter.py
    shell_file_path = create_a_shell_file(file_path, parameters_list, solver)

    # run the shell file
    result, used_time = run_a_shell_file(shell_file_path, file_path, solver)

    # delete the shell file
    if os.path.exists(shell_file_path):
        os.remove(shell_file_path)

    return result, used_time


def create_a_shell_file(file_path, parameter_list="", solver=""):
    parameter_str = "".join(parameter_list)
    shell_folder = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/process_benchmarks/temp_shell"
    shell_file_name = "run-" + os.path.basename(file_path) + ".sh"
    shell_file_path = os.path.join(shell_folder, shell_file_name)
    timeout_command = "timeout " + str(shell_timeout)

    solver_command = solver_command_map[solver]
    if os.path.exists(shell_file_path):
        os.remove(shell_file_path)
    with open(shell_file_path, 'w') as file:
        file.write("#!/bin/sh\n")
        file.write(timeout_command + " " + solver_command + " " + file_path + " " + parameter_str + "\n")
    return shell_file_path


def run_a_shell_file(shell_file_path: str, problem_file_path: str, solver):
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
    result = process_solver_output(completed_process.stdout, problem_file_path, solver)
    print("Finished", "use time: ", used_time)
    return result, used_time


def process_solver_output(solver_output: str, problem_file_path: str, solver):
    result = UNKNOWN

    if solver == "this":
        lines = solver_output.split('\n')
        for line in lines:
            if "result:" in line:
                result = line.split("result:")[1].strip(" ")
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
    if result == SAT or result == UNSAT:
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

    return result
