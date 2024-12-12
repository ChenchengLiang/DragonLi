import os
import shutil
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
from src.solver.independent_utils import strip_file_name_suffix, create_folder
from src.process_benchmarks.utils import summary_one_track
from src.solver.Constants import project_folder,bench_folder
import argparse


def main():
    benchmark_name=f"Benchmark_C_train_eq_1_100_1_10000"
    merged_summary_folder = project_folder+f"/src/process_benchmarks/summary/merge_summary/{benchmark_name}_summary"
    create_folder(project_folder+f"/src/process_benchmarks/summary/{benchmark_name}_new_trainable_data")

    #this_solver_rank_option="graph_1_hybrid_category_gnn_random_each_n_iterations_GCNSplit"
    this_solver_rank_option="category_random"
    summary_file_this_solver = f"this_fixed_termination_condition_0_{this_solver_rank_option}_{benchmark_name}_summary.csv"
    log_string=""
    # learn from these target solvers when targer solver is this solver replace ":" with "_" in the name
    target_solver_list=["z3","z3-noodler","cvc5","ostrich"]
    #target_solver_list = ["this_fixed_termination_condition_0_hybrid_category_fixed_random"]
    for target_solver in target_solver_list:

        summary_file_target_solver=f"{target_solver}_{benchmark_name}_summary.csv"


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



        common_solved_list = []
        common_unsolvable_list = []
        this_solver_unique_solved_list = []
        target_solver_unique_solved_list = []

        for this_solver_row, target_solver_row in zip(this_solver_summary, target_solver_summary):
            filename=strip_file_name_suffix(this_solver_row[0])
            this_solver_satisfiability=this_solver_row[1]
            target_solver_satisfiability=target_solver_row[1]
            if this_solver_satisfiability==target_solver_satisfiability: # get common list
                if this_solver_satisfiability=="UNKNOWN": #both unknown
                    common_unsolvable_list.append((filename,this_solver_satisfiability))
                else: # both SAT or UNSAT
                    common_solved_list.append((filename,this_solver_satisfiability))
            else: # get unique list
                if this_solver_satisfiability=="UNKNOWN" and target_solver_satisfiability!="UNKNOWN":
                    target_solver_unique_solved_list.append((filename,target_solver_satisfiability))
                elif this_solver_satisfiability!="UNKNOWN" and target_solver_satisfiability=="UNKNOWN":
                    this_solver_unique_solved_list.append((filename,this_solver_satisfiability))
                else:
                    pass

            # print(filename)
            # print(this_solver_row, target_solver_row)

        log_string+=f"target_solver: {target_solver}\n"
        log_string+=f"common_solved_list: {len(common_solved_list)}\n"
        log_string+=f"common_unsolvable_list: {len(common_unsolvable_list)}\n"
        log_string+=f"this_solver_unique_solved_list: {len(this_solver_unique_solved_list)}\n"
        log_string+=f"target_solver_unique_solved_list: {len(target_solver_unique_solved_list)}\n"
        log_string+="-"*10
        log_string+="\n"

        print("target_solver",target_solver)
        print("common_solved_list",len(common_solved_list))
        print("common_unsolvable_list",len(common_unsolvable_list))
        print("this_solver_unique_solved_list",len(this_solver_unique_solved_list))
        print("target_solver_unique_solved_list",len(target_solver_unique_solved_list))
        print("-"*10)


        #store the new trainable data
        new_trainable_data_folder = project_folder+f"/src/process_benchmarks/summary/{benchmark_name}_new_trainable_data/{target_solver}"
        create_folder(new_trainable_data_folder)
        create_folder(new_trainable_data_folder + "/SAT")
        create_folder(new_trainable_data_folder + "/UNSAT")


        for file_name,satisfiability in target_solver_unique_solved_list:
            if satisfiability=="SAT":
                shutil.copy(f"{bench_folder}/{benchmark_name}/ALL/ALL/{file_name}.eq", new_trainable_data_folder+"/SAT")
                shutil.copy(f"{bench_folder}/{benchmark_name}/ALL/ALL/{file_name}.smt2", new_trainable_data_folder+"/SAT")
            elif satisfiability=="UNSAT":
                shutil.copy(f"{bench_folder}/{benchmark_name}/ALL/ALL/{file_name}.eq",
                            new_trainable_data_folder + "/UNSAT")
                shutil.copy(f"{bench_folder}/{benchmark_name}/ALL/ALL/{file_name}.smt2",
                            new_trainable_data_folder + "/UNSAT")
            else:
                print(f"Error, satisfiability from target_solver is  UNKNOWN")

    #merge all the new trainable data
    merged_new_trainable_data_folder = project_folder+f"/src/process_benchmarks/summary/{benchmark_name}_new_trainable_data/merged_new_trainable_data"
    create_folder(merged_new_trainable_data_folder)
    create_folder(merged_new_trainable_data_folder + "/SAT")
    create_folder(merged_new_trainable_data_folder + "/UNSAT")


    for target_solver in target_solver_list:
        new_trainable_data_folder = project_folder+f"/src/process_benchmarks/summary/{benchmark_name}_new_trainable_data/{target_solver}"
        shutil.copytree(new_trainable_data_folder+"/SAT",merged_new_trainable_data_folder+"/SAT",dirs_exist_ok=True)
        shutil.copytree(new_trainable_data_folder + "/UNSAT", merged_new_trainable_data_folder+"/UNSAT", dirs_exist_ok=True)
        # for file_name in os.listdir(new_trainable_data_folder):
        #     shutil.copy(f"{new_trainable_data_folder}/{file_name}", merged_new_trainable_data_folder)

    # output log_string to file
    track_info_file = project_folder+f"/src/process_benchmarks/summary/{benchmark_name}_new_trainable_data/log.txt"
    with open(track_info_file, 'w') as file:
        file.write(log_string)









if __name__ == '__main__':
    main()
