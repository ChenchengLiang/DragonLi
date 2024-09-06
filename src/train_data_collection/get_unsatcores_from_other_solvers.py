import glob
import os
import shutil
import sys
import configparser
import time

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)


from tqdm import tqdm

from src.process_benchmarks.eq2smt_utils import one_eq_file_to_smt2
from src.solver.DataTypes import Equation, Formula
from src.solver.utils_parser import perse_eq_file
from src.solver.independent_utils import strip_file_name_suffix, delete_relative_files, log_print_to_file
from src.train_data_collection.generate_tracks import formatting_results,formatting_results_v2


from typing import List, Tuple, Dict
from src.solver.Constants import bench_folder,suffix_dict
from src.process_benchmarks.utils import run_on_one_track,run_on_one_problem
from itertools import combinations
from src.solver.independent_utils import create_folder
import argparse

#@log_print_to_file
def main():
    # parse argument
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('benchmark', type=str,
                            help='benchmark name')
    arg_parser.add_argument('folder', type=str,
                            help='divided_i or valid_data folder')
    args = arg_parser.parse_args()
    # Accessing the arguments
    benchmark = args.benchmark
    folder = args.folder



    working_folder = f"{bench_folder}/{benchmark}/{folder}"


    solver_log = False
    source_solver_initial_shell_timeout=10
    this_solver_shell_timeout=60
    source_solver_parameter_list_map={"z3":[],"z3-noodler":["smt.string_solver=\"noodler\""],"cvc5":[],"ostrich":[],"woorpje":[]}


    z3_folder_file_basename_list= [os.path.basename(f) for f in glob.glob(f"{working_folder}/z3/UNSAT/*.eq")]
    cvc5_folder_file_basename_list = [os.path.basename(f) for f in glob.glob(f"{working_folder}/cvc5/UNSAT/*.eq")]
    ostrich_folder_file_basename_list = [os.path.basename(f) for f in glob.glob(f"/{working_folder}/ostrich/UNSAT/*.eq")]
    z3_noodler_folder_file_basename_list = [os.path.basename(f) for f in glob.glob(f"{working_folder}/z3-noodler/UNSAT/*.eq")]
    


    benchmark_folder=f"{working_folder}/merged_new_trainable_data/UNSAT"
    file_list = glob.glob(benchmark_folder+ "/*.eq")
    for file in tqdm(file_list,desc="processing progress"):

        #decide the solver
        file_basename=os.path.basename(file)
        if file_basename in z3_folder_file_basename_list:
            solver="z3"
        elif file_basename in cvc5_folder_file_basename_list:
            solver="cvc5"
        elif file_basename in ostrich_folder_file_basename_list:
            solver="ostrich"
        elif file_basename in z3_noodler_folder_file_basename_list:
            solver="z3-noodler"


        parameters_list = source_solver_parameter_list_map[solver]

        print(f"file_basename {file_basename}, solver {solver}")



        # solve the original problem
        smt2_file=strip_file_name_suffix(file)+".smt2"
        if not os.path.exists(smt2_file):
            one_eq_file_to_smt2(file)
        start=time.time()
        run_on_one_problem(smt2_file, parameters_list, solver, solver_log=solver_log,
                           shell_timeout=source_solver_initial_shell_timeout)
        initial_run_time = time.time() - start
        if initial_run_time>source_solver_initial_shell_timeout:
            print(f"original problem timeout with {initial_run_time}")
        else:
            extract_unsatcores(file,initial_run_time,solver,parameters_list,this_solver_shell_timeout,solver_log)



def extract_unsatcores(file,initial_run_time,solver,parameters_list,this_solver_shell_timeout,solver_log):
    shell_timeout_factor = 10
    shell_timeout = initial_run_time * shell_timeout_factor
    print("shell_timeout:", shell_timeout)

    # parse .eq
    parsed_content = perse_eq_file(file)

    # delete eq one by one to create unsat core file list
    unsat_core_file_folder = strip_file_name_suffix(file) + "_unsat_cores"
    create_folder(unsat_core_file_folder)

    # todo is the smallest unsatcore always the best?
    # todo keep all smallest unsatcores?
    # todo change rank to solve the unsolved problems only provide more the same kind training data or very different training data?
    found_unsatcore = False
    total_eq_number = len(parsed_content["equation_list"])
    delete_eq_number_list = list(reversed(range(1, total_eq_number)))

    for eq_number_to_delete in tqdm(delete_eq_number_list, desc="deleting progress"):
        if found_unsatcore:
            break

        combination_list = list(combinations(parsed_content["equation_list"], eq_number_to_delete))

        for index, eq_list_to_delete in enumerate(combination_list):
            print(f"Delete {eq_number_to_delete} from {total_eq_number}, {index}/{len(combination_list)}")

            unsat_core: List[Equation] = [eq for eq in parsed_content["equation_list"] if eq not in eq_list_to_delete]

            # store to eq file
            unsat_core_formula = Formula(unsat_core)
            eq_string_to_file = unsat_core_formula.eq_string_for_file()
            # create eq file
            unsat_core_eq_file = unsat_core_file_folder + f"/delete_{eq_number_to_delete}_{index}.eq"
            with open(unsat_core_eq_file, "w") as f:
                f.write(eq_string_to_file)
            # store to smt2 file
            unsat_core_smt2_file = one_eq_file_to_smt2(unsat_core_eq_file)

            # check whether it is an unsatcore
            result_dict = run_on_one_problem(unsat_core_smt2_file, parameters_list, solver, solver_log=solver_log,
                                             shell_timeout=shell_timeout)
            # print(result_dict)
            satisfiability = result_dict["result"]

            if satisfiability == "UNSAT":
                print("UNSAT core:")
                for eq in unsat_core:
                    print(eq.eq_str)

                # check weather useful to DragonLi
                # run DragonLi with the unsatcore
                print("run this solver")
                parameter_list = ["fixed",
                                  f"--termination_condition termination_condition_0", f"--algorithm SplitEquations",
                                  f"--graph_type graph_1",
                                  f"--order_equations_method unsatcore_shortest",
                                  f"--unsat_core_file {unsat_core_eq_file}"]

                solver = "this"

                result_dict = run_on_one_problem(file, parameter_list, solver,
                                                 solver_log=solver_log, shell_timeout=this_solver_shell_timeout)
                satisfiability_this_solver = result_dict["result"]
                print(result_dict["result"])
                print(result_dict["raw"])
                if satisfiability_this_solver == "UNSAT":
                    print("Found an available unsatcore")
                    shutil.copy(unsat_core_eq_file, strip_file_name_suffix(file) + ".unsatcore")
                    found_unsatcore = True
                    break
                else:
                    print("this solver cannot solve it with the unsatcore")

            else:
                print(f"{satisfiability}, delete relative files")
                delete_relative_files(unsat_core_eq_file)





if __name__ == '__main__':
    main()