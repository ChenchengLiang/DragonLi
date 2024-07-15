import os
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
from src.solver.independent_utils import strip_file_name_suffix


def main():
    # debug path
    # file_path = bench_folder + "/debug/04_track_5.eq"
    # file_path = bench_folder + "/debug/g_03_track_9153.eq"
    # file_path = bench_folder + "/debug/g_03_track_9596.eq"
    # file_path = bench_folder + "/debug/g_03_track_train_task_3_1_5000_839.eq"
    # file_path = bench_folder + "/debug/g_03_track_train_task_3_1_5000_4104.eq"
    # file_path = bench_folder + "/debug-eval/g_03_track_27.eq"
    # file_path = "/home/cheli243/Desktop/CodeToGit/Woorpje_benchmarks/debug-eval-uppmax/ALL/divided_1/04_track_1.eq"
    # file_path = bench_folder + "/debug-rank/1/g_01_track_multi_word_equations_generated_eval_1001_2000_1685.eq" #UNSAT 83 eqs
    #file_path = bench_folder + "/debug-rank/2/g_01_track_multi_word_equations_generated_eval_1001_2000_1004.eq"  # UNSAT

    # file_path = bench_folder +"/test/03_track_11.eq"
    # Woorpje_benchmarks path
    # SAT
    # file_path = bench_folder +"/01_track/01_track_1.eq"
    # file_path = bench_folder +"/01_track/01_track_2.eq"
    # file_path = bench_folder +"/01_track/01_track_3.eq"
    # file_path = bench_folder +"/01_track/01_track_4.eq"
    # file_path = bench_folder +"/01_track/01_track_5.eq"
    # file_path = bench_folder +"/01_track/01_track_36.eq"
    # file_path = bench_folder +"/01_track/01_track_37.eq"
    # file_path = bench_folder +"/01_track/01_track_58.eq"
    # file_path = bench_folder +"/01_track/01_track_93.eq"
    # file_path = bench_folder +"/01_track/01_track_192.eq"

    # UNSAT
    # file_path = bench_folder +"/03_track/03_track_14.eq"
    # file_path = bench_folder +"/03_track/03_track_7.eq"
    # file_path = bench_folder +"/03_track/03_track_11.eq"
    # file_path = bench_folder +"/03_track/03_track_17.eq"

    # multiple equations
    #file_path = bench_folder + "/examples_choose_eq/1/test1.eq"  # SAT
    #file_path = bench_folder + "/examples_choose_eq/2/test2.eq"  # UNSAT
    # file_path = bench_folder + "/examples_choose_eq/3/test3.eq"  # SAT
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
    #file_path = bench_folder + "/examples_choose_eq/15/g_01_track_multi_word_equations_generated_eval_1001_2000_1496.eq"  # SAT
    file_path = bench_folder + "/examples_choose_eq/16/g_01_track_multi_word_equations_generated_eval_1001_2000_1131.eq"


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

    #file_path = bench_folder + "/conjunctive_03_track_eval_rank_task_1_1000/ALL/ALL/g_conjunctive_03_track_train_rank_task_1_1000_1.eq"
    #file_path = bench_folder + "/01_track_multi_word_equations_generated_eval_1001_2000/ALL/ALL/g_01_track_multi_word_equations_generated_eval_1001_2000_1001.eq"

    # smt format
    # file_path=bench_folder +"/example_smt/1586.corecstrs.readable.smt2"

    parser_type = EqParser() if file_path.endswith(".eq") else SMT2Parser()
    parser = Parser(parser_type)
    parsed_content = parser.parse(file_path)
    print("parsed_content:", parsed_content)

    graph_type = "graph_1"
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
                                           "order_equations_method": "category",
                                           "termination_condition": "termination_condition_0",
                                           "graph_type": graph_type, "graph_func": graph_func_map[graph_type],
                                           "label_size": label_size,"rank_task":rank_task}

    algorithm_parameters_SplitEquations_gnn = {"branch_method": "fixed",
                                               "order_equations_method": "gnn",
                                               "gnn_model_path": gnn_model_path,
                                               "termination_condition": "termination_condition_0",
                                               "graph_type": graph_type, "graph_func": graph_func_map[graph_type],
                                               "label_size": label_size,"rank_task":rank_task}

    algorithm_parameters_SplitEquationsExtractData = {"branch_method": "fixed",
                                                      "order_equations_method": "category_random",
                                                      "termination_condition": "termination_condition_7",
                                                      "graph_type": graph_type,
                                                      "graph_func": graph_func_map[graph_type],
                                                      "task": "dynamic_embedding", "label_size": label_size,
                                                      "rank_task":rank_task,"eq_satisfiability":eq_satisfiability}

    solver = Solver(algorithm=SplitEquations, algorithm_parameters=algorithm_parameters_SplitEquations_gnn)
    #solver = Solver(algorithm=SplitEquationsExtractData, algorithm_parameters=algorithm_parameters_SplitEquationsExtractData)
    #solver = Solver(algorithm=SplitEquations, algorithm_parameters=algorithm_parameters_SplitEquations)

    # solver = Solver(algorithm=ElimilateVariablesRecursive,algorithm_parameters=algorithm_parameters_ElimilateVariablesRecursive)
    # solver = Solver(EnumerateAssignmentsUsingGenerator, max_variable_length=max_variable_length,algorithm_parameters=algorithm_parameters)
    # solver = Solver(algorithm=EnumerateAssignments,max_variable_length=max_variable_length,algorithm_parameters=algorithm_parameters)
    result_dict = solver.solve(parsed_content, visualize=True, output_train_data=True)

    print_results(result_dict)


if __name__ == '__main__':
    main()
