import os
import shutil
import sys
import configparser

from src.solver.algorithms.split_equations_extract_data import SplitEquationsExtractData

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.Constants import bench_folder, project_folder, UNKNOWN, rank_task_label_size_map
from src.solver.Parser import Parser, EqParser, SMT2Parser
from src.solver.Solver import Solver
from src.solver.utils import print_results, graph_func_map
from src.solver.algorithms import EnumerateAssignments, EnumerateAssignmentsUsingGenerator, \
    ElimilateVariablesRecursive, SplitEquations
from src.solver.DataTypes import Equation
from src.solver.independent_utils import strip_file_name_suffix, profile_function

import cProfile
import pstats


def main():
    # debug path
    #file_path = bench_folder + "/examples_choose_eq/kepler/48327_quad-002-1-unsat_0.eq" # trap prefix and suffix
    #file_path = bench_folder + "/examples_choose_eq/kepler/48381_quad-001-1-unsat_0.eq"
    #file_path = bench_folder + "/examples_choose_eq/30/30.eq" #suffix timeout
    #file_path = bench_folder + "/examples_choose_eq/31/31.eq" #suffix timeout
    #file_path = bench_folder + "/examples_choose_eq/32/32.eq" #trap prefix. X a = Y X C [X=YY'], [Y=XX'], [X=Y]
    #file_path = bench_folder + "/examples_choose_eq/33/33.eq"  # trap suffix. a X = C X Y [X=Y'Y] -> a Y' = CY'Y
    #file_path = bench_folder + "/examples_choose_eq/34/34.eq"  # trap prefix and suffix a X = Y X b
    #file_path = bench_folder + "/examples_choose_eq/35/35.eq"  # trap prefix and suffix X a = b X Y
    #file_path = bench_folder + "/examples_choose_eq/36/36.eq"  # X = Y a X
    #file_path = bench_folder + "/examples_choose_eq/37/37.eq"  # X = Y X b # trap prefix and suffix
    #file_path = bench_folder + "/examples_choose_eq/37/37-1.eq"  # rewrite X = Y X b to X = Y Z b \wedge X=Z
    #file_path = bench_folder + "/examples_choose_eq/37/37-2.eq"  # rewrite X = Y X b to X = Y Z b \wedge X=Z
    #file_path = bench_folder + "/examples_choose_eq/37/37-3.eq"  # rewrite X = Y X b to X = Y Z b \wedge X=Z
    #file_path = bench_folder + "/examples_choose_eq/38/g_01_track_multi_word_equations_eq_2_50_generated_train_1_1000_124.eq"
    #file_path = bench_folder + "/examples_choose_eq/39/g_01_track_multi_word_equations_eq_2_50_generated_bootstrapping_1_10000_329.eq"
    #file_path = bench_folder + "/examples_choose_eq/40/g_01_track_multi_word_equations_eq_2_50_generated_bootstrapping_1_10000_5307.eq"
    #file_path = bench_folder + "/examples_choose_eq/41/g_01_track_multi_word_equations_eq_2_50_generated_train_1_1000_183.eq"
    #file_path = bench_folder + "/examples_choose_eq/42/g_01_track_multi_word_equations_eq_2_50_generated_train_1_1000_747.eq"
    #file_path = bench_folder + "/examples_choose_eq/suffix-1/g_01_track_multi_word_equations_eq_2_50_generated_train_1_1000_1.eq"
    #file_path = bench_folder + "/examples_choose_eq/suffix-2/g_01_track_multi_word_equations_eq_2_50_generated_train_1_1000_10.eq"
    #file_path = bench_folder + "/examples_choose_eq/suffix-3/g_01_track_multi_word_equations_eq_2_50_generated_train_1_1000_1000.eq"
    #file_path = bench_folder + "/examples_choose_eq/suffix-4/g_01_track_multi_word_equations_eq_2_50_generated_train_1_1000_102.eq"
    #file_path = bench_folder + "/examples_choose_eq/suffix-5/g_01_track_multi_word_equations_eq_2_50_generated_train_1_1000_103.eq"
    #file_path = bench_folder + "/examples_choose_eq/suffix-6/g_01_track_multi_word_equations_eq_2_50_generated_train_1_1000_104.eq"
    #file_path = bench_folder + "/examples_choose_eq/suffix-7/g_01_track_multi_word_equations_eq_2_50_generated_train_1_1000_106.eq"
    file_path = "webapp/example.eq"


    #file_path = bench_folder + "/examples_choose_eq/extract_unsatcore_proof_tree/g_01_track_multi_word_equations_eq_2_50_generated_train_1_10000_3689.eq"

    """
    X = Y X b
    number of X in right side >= number of X in left side
    """
    #conjunctive 03 track
    #file_path = bench_folder + "/Benchmark_C_eval_eq_1_100_1_1000/ALL/ALL/g_Benchmark_C_eval_eq_1_100_1_1000_69.eq"

    # inconsistent check
    #file_path = bench_folder + "/examples_choose_eq/inconsistent/1/g_Benchmark_D_max_replace_length_bounded_16_train_1_10000_1047.eq"
    #file_path = bench_folder + "/examples_choose_eq/inconsistent/1/g_Benchmark_D_max_replace_length_bounded_16_train_1_10000_1789.eq"
    #file_path = bench_folder + "/examples_choose_eq/inconsistent/1/g_Benchmark_D_max_replace_length_bounded_16_train_1_10000_6520.eq"
    #file_path = bench_folder + "/examples_choose_eq/inconsistent/1/g_Benchmark_D_max_replace_length_bounded_16_train_1_10000_8681.eq"
    #file_path = bench_folder + "/examples_choose_eq/inconsistent/1/g_Benchmark_D_max_replace_length_bounded_16_train_1_10000_8765.eq"
    #file_path = bench_folder + "/examples_choose_eq/inconsistent/1/g_Benchmark_D_max_replace_length_bounded_16_train_1_10000_8808.eq"

    # gnn test
    #file_path = bench_folder + "/examples_choose_eq/gnn-1/g_01_track_multi_word_equations_eq_2_50_generated_train_1_1000_690.eq"

    # multiple equations
    #file_path = bench_folder + "/examples_choose_eq/1/test1.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/2/test2.eq"  # UNSAT
    #file_path = bench_folder + "/examples_choose_eq/3/test3.eq"  # SAT
    # file_path = bench_folder + "/examples_choose_eq/4/test4.eq"  # SAT
    # file_path = bench_folder + "/examples_choose_eq/5/test5.eq"  # SAT
    # file_path = bench_folder + "/examples_choose_eq/6/test6.eq"  # SAT
    # file_path = bench_folder + "/examples_choose_eq/7/test7.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/8/test8.eq"  # SAT
    # file_path = bench_folder + "/examples_choose_eq/9/test9.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/10/test10.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/11/test11.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/12/test12.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/13/test13.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/14/g_conjunctive_01_track_train_rank_task_1_100_1.eq"  # UNSAT
    #file_path = bench_folder + "/examples_choose_eq/15/g_01_track_multi_word_equations_generated_eval_1001_2000_1496.eq"  # SAT #reverse helps
    #file_path = bench_folder + "/examples_choose_eq/16/g_01_track_multi_word_equations_generated_eval_1001_2000_1131.eq"
    #file_path = bench_folder + "/examples_choose_eq/17/17.eq"
    #file_path = bench_folder + "/examples_choose_eq/18/18.eq" #UNSAT, big graph
    #file_path = bench_folder + "/examples_choose_eq/19/g_conjunctive_01_track_train_eq_number_20_rank_task_1_20000_638.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/20/g_01_track_multi_word_equations_generated_eval_eq_number_5_rank_task_1_5_3.eq"  # unknown
    #file_path = bench_folder + "/examples_choose_eq/20/g_01_track_multi_word_equations_generated_eval_eq_number_5_rank_task_1_5_1.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/21/21.eq"  # UNSAT
    #file_path = bench_folder + "/examples_choose_eq/22/22.eq"  # UNSAT
    #file_path = bench_folder + "/examples_choose_eq/23/23.eq"  # UNSAT
    #file_path = bench_folder + "/examples_choose_eq/24/24.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/25/25.eq"  # UNSNAT
    #file_path = bench_folder + "/examples_choose_eq/26/26.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/27/g_01_track_multi_word_equations_eq_5_20_generated_train_20001_30000_25376.eq"  #
    #file_path = bench_folder + "/temp/output.eq"  #parse


    # file_path = bench_folder + "/examples/multi_eqs/4/g_04_track_generated_train_1_1000_4.eq"  # UNSAT
    # file_path = bench_folder + "/examples/multi_eqs/5/g_04_track_generated_train_1_1000_5.eq"  # UNSAT
    # file_path = bench_folder + "/examples/multi_eqs/26/04_track_26.eq"  # SAT
    # file_path=bench_folder +"/examples/multi_eqs/test3.eq" #UNSAT
    # file_path=bench_folder +"/examples/multi_eqs/04_track_6.eq" #SAT
    # file_path=bench_folder +"/examples/multi_eqs/04_track_59.eq" #UNSAT
    # file_path=bench_folder +"/examples/multi_eqs/04_track_172.eq" #SAT
    # file_path = bench_folder + "/examples/multi_eqs/04_track_189.eq"  # SAT
    # file_path = bench_folder + "/examples/multi_eqs/04_track_19.eq"  # UNSAT
    # file_path = bench_folder + "/examples/multi_eqs/04_track_80.eq"  # UNSAT
    # file_path = bench_folder + "/examples/multi_eqs/04_track_180.eq"  # UNSAT
    # file_path = bench_folder + "/examples/multi_eqs/04_track_183.eq"  # UNSAT
    # file_path=bench_folder +"/debug/19949.corecstrs.readable.eq" #UNSAT
    # file_path = bench_folder + "/debug/slent_kaluza_458_sink.eq"  # UNSAT
    # file_path = bench_folder + "/debug/slent_kaluza_569_sink.eq"  # UNSAT
    # file_path = bench_folder + "/debug/slent_kaluza_1325_sink.eq"  # UNSAT

    # smt format
    # file_path=bench_folder +"/example_smt/1586.corecstrs.readable.smt2"

    parser_type = EqParser() if file_path.endswith(".eq") else SMT2Parser()
    parser = Parser(parser_type)
    parsed_content = parser.parse(file_path)
    print("parsed_content:", parsed_content)

    graph_type = "graph_3"
    task = "task_3"
    rank_task = 1
    label_size = rank_task_label_size_map[rank_task]
    model_type = "GCNSplit"
    gnn_model_path = f"{project_folder}/Models/model_0_{graph_type}_{model_type}.pth"
    eq_satisfiability="UNSAT"



    algorithm_parameters_ElimilateVariablesRecursive = {"branch_method": "fixed", "task": task,
                                                        "graph_type": graph_type,
                                                        "graph_func": graph_func_map[graph_type],
                                                        "gnn_model_path": gnn_model_path, "extract_algorithm": "fixed",
                                                        "termination_condition": "termination_condition_0",
                                                        "label_size": label_size,"rank_task":rank_task}  # branch_method [extract_branching_data_task_2,random,fixed,gnn,gnn:fixed,gnn:random]

    algorithm_parameters_SplitEquations = {"branch_method": "fixed",
                                           "order_equations_method": "category_random",  # "gnn_first_n_iterations_category"
                                           "termination_condition": "termination_condition_0",
                                           "graph_type": graph_type, "graph_func": graph_func_map[graph_type],
                                           "label_size": label_size,"rank_task":rank_task}

    algorithm_parameters_SplitEquations_gnn = {"branch_method": "fixed",
                                               "order_equations_method": "category_gnn_first_n_iterations",# "gnn_first_n_iterations_category"
                                               "gnn_model_path": gnn_model_path,
                                               "termination_condition": "termination_condition_0",
                                               "graph_type": graph_type, "graph_func": graph_func_map[graph_type],
                                               "label_size": label_size,"rank_task":rank_task}

    algorithm_parameters_SplitEquationsExtractData = {"branch_method": "fixed",
                                                      "order_equations_method": "unsatcore_shortest_first_n_iterations_category",
                                                      #"termination_condition": "termination_condition_4",# for sat
                                                      "termination_condition": "termination_condition_7",#for unsat
                                                      "graph_type": graph_type,
                                                      "graph_func": graph_func_map[graph_type],
                                                      "task": "dynamic_embedding", "label_size": label_size,
                                                      "rank_task":rank_task,"eq_satisfiability":eq_satisfiability}
    log_file=f"{bench_folder}/last_run.log"
    if os.path.exists(log_file):
        os.remove(log_file)


    #solver = Solver(algorithm=SplitEquations, algorithm_parameters=algorithm_parameters_SplitEquations_gnn)
    #solver = Solver(algorithm=SplitEquationsExtractData, algorithm_parameters=algorithm_parameters_SplitEquationsExtractData)
    solver = Solver(algorithm=SplitEquations, algorithm_parameters=algorithm_parameters_SplitEquations)

    #result_dict = profile_function(solver.solve, parsed_content, visualize=False, output_train_data=True)

    result_dict = solver.solve(parsed_content, visualize=False, output_train_data=True)

    print_results(result_dict)

    if os.path.exists(log_file):
        shutil.move(log_file,os.path.dirname(file_path))


if __name__ == '__main__':
    main()
