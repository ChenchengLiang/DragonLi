
import glob
import os
import sys
import configparser
import time

from tqdm import tqdm

from src.process_benchmarks.eq2smt_utils import one_eq_file_to_smt2
from src.solver.DataTypes import Equation, Formula
from src.solver.Parser import EqParser, Parser
from src.solver.independent_utils import strip_file_name_suffix
from src.train_data_collection.generate_tracks import formatting_results,formatting_results_v2

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')

sys.path.append(path)
from typing import List, Tuple, Dict
from src.solver.Constants import bench_folder,suffix_dict
from src.process_benchmarks.utils import run_on_one_track,run_on_one_problem
from itertools import combinations
from src.solver.independent_utils import create_folder

def main():
    benchmark_name = "01_track_multi_word_equations_generated_eval_1001_2000_new_trainable_data_test"


    # extract unsat cores
    solver="z3"
    solver_log = False
    shell_timeout=10

    config = {
        "benchmark_name": benchmark_name,
        "benchmark_folder": f"{bench_folder}/{benchmark_name}/{solver}",
        "solver":solver,
        "parameters_list":[],
        "summary_folder_name": f"{benchmark_name}_summary"
    }

    print("config:", config)


    file_list = glob.glob(config["benchmark_folder"] + "/*.eq")
    for file in tqdm(file_list,desc="processing progress"):

        # solve the original problem
        smt2_file=strip_file_name_suffix(file)+".smt2"
        start=time.time()
        run_on_one_problem(smt2_file, config["parameters_list"], solver, solver_log=solver_log,
                           shell_timeout=shell_timeout)
        initial_run_time = time.time() - start
        shell_timeout_factor=10
        shell_timeout=initial_run_time*shell_timeout_factor
        print("shell_timeout:",shell_timeout)

        # parse .eq
        parser_type = EqParser()
        parser = Parser(parser_type)
        parsed_content = parser.parse(file)

        # delete eq one by one to create unsat core file list
        unsat_core_file_folder=strip_file_name_suffix(file)+"_unsat_cores"
        create_folder(unsat_core_file_folder)


        for eq_number_to_delete in tqdm(range(1,len(parsed_content["equation_list"])),desc="deleting progress"):
            eq_list_to_delete = list(combinations(parsed_content["equation_list"], eq_number_to_delete))
            for i,eq_list_to_delete in enumerate(eq_list_to_delete):

                unsat_core:List[Equation]= [eq for eq in parsed_content["equation_list"] if eq not in eq_list_to_delete]

                #store to eq file
                unsat_core_formula=Formula(unsat_core)
                eq_string_to_file=unsat_core_formula.eq_string_for_file()
                print(eq_string_to_file)
                # create eq file
                unsat_core_eq_file=unsat_core_file_folder+f"/delete_{eq_number_to_delete}_{i}.eq"
                with open(unsat_core_eq_file, "w") as f:
                    f.write(eq_string_to_file)
                #store to smt2 file
                unsat_core_smt2_file=one_eq_file_to_smt2(unsat_core_eq_file)

                # run the unsat core problem
                result_dict = run_on_one_problem(unsat_core_smt2_file, config["parameters_list"], solver, solver_log=solver_log,shell_timeout=shell_timeout)
                print(result_dict)
                satisfiability=result_dict["result"]

                if satisfiability=="UNSAT":
                    print("UNSAT core:")
                    for eq in unsat_core:
                        print(eq.eq_str)
                else:
                    print(f"{satisfiability} , delete files")
                    os.remove(unsat_core_eq_file)
                    os.remove(unsat_core_smt2_file)


        #check weather useful to DragonLi
        unsat_core_file_list = glob.glob(unsat_core_file_folder + "/*.eq")

        # benchmark_name = "01_track_multi_word_equations_generated_eval_1001_2000_new_trainable_data_test"



    #
    #
    # # run Dragon with SplitEquations algorithm
    # termination_condition = "termination_condition_0"
    # algorithm="SplitEquations"
    # config = {
    #     "benchmark_name": benchmark_name,
    #     "benchmark_folder": bench_folder + "/" + benchmark_name + "/ALL/ALL",
    #     # "solver":"ostrich",
    #     # "parameters_list":[],
    #     "solver": "this",
    #     "parameters_list": ["fixed",
    #                         f"--termination_condition {termination_condition}",
    #                         f"--algorithm {algorithm}",
    #                         f"--order_equations_method category"
    #                         ],
    #     "summary_folder_name": f"{benchmark_name}_summary"
    # }
    #
    # solver_log = False
    #
    # print("config:", config)
    #
    # run_on_one_track(config["benchmark_name"], config["benchmark_folder"], config["parameters_list"], config["solver"],
    #                  suffix_dict, summary_folder_name=config["summary_folder_name"],
    #                  solver_log=solver_log)


if __name__ == '__main__':
    main()