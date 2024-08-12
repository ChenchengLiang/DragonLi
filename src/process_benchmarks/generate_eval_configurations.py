import configparser
import os
import sys

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')

sys.path.append(path)
from src.solver.Constants import bench_folder, project_folder
from src.solver.independent_utils import write_configurations_to_json_file


def main():
    benchmark_name = "01_track_multi_word_equations_generated_eval_1001_2000"
    #benchmark_name = "conjunctive_03_track_train_rank_task_1_5000"
    #benchmark_name = "conjunctive_03_track_eval_rank_task_1_1000"
    #benchmark_name = "conjunctive_01_track_train_rank_task_1_100"
    #benchmark_name = "01_track_multi_word_equations_generated_train_1_40000_for_rank_task_2"
    rank_task = 0
    graph_type = "graph_1"
    model_type = "GCNSplit"  # "GINSplit"


    model_folder = project_folder + "/" + "Models/"
    task = "rank_task_1"  # "task_3"
    algorithm = "SplitEquations"
    solver_param_list = [
        # branch:fixed, order equations: no category
        # ["this", ["fixed", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method fixed"
        #           ]],
        # ["this", ["fixed", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method random"
        #           ]],
        # ["this", ["fixed", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method hybrid_fixed_random"
        #           ]],

        # branch:fixed, order equations: category
        # ["this", ["fixed", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method category"
        #           ]],
        # ["this", ["fixed", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method category_random"
        #           ]],
        # ["this", ["fixed", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method hybrid_category_fixed_random"
        #           ]],
        #
        # # branch:fixed, order equations: shortest, longest
        # ["this", ["fixed", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method shortest"
        #           ]],
        # ["this", ["fixed", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method longest"
        #           ]],
        #
        # # branch:random, order equations: no category
        # ["this", ["random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method fixed"
        #           ]],
        # ["this", ["random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method random"
        #           ]],
        # ["this", ["random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method hybrid_fixed_random"
        #           ]],
        #
        # # branch:random, order equations: category
        # ["this", ["random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method category"
        #           ]],
        # ["this", ["random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method category_random"
        #           ]],
        # ["this", ["random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method hybrid_category_fixed_random"
        #           ]],
        #
        # # branch:random, order equations: shortest, longest
        # ["this", ["random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method shortest"
        #           ]],
        #
        # ["this", ["random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method longest"
        #           ]],
        #
        # # branch:hybrid, order equations: no category
        # ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method fixed"
        #           ]],
        # ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method random"
        #           ]],
        # ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method hybrid_fixed_random"
        #           ]],
        #
        # # branch:hybrid, order equations: category
        # ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method category"
        #           ]],
        # ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method category_random"
        #           ]],
        # ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method hybrid_category_fixed_random"
        #           ]],
        #
        # # branch:hybrid, order equations: shortest, longest
        # ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method shortest"
        #           ]],
        # ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
        #           f"--graph_type {graph_type}",
        #           f"--algorithm {algorithm}",
        #           f"--order_equations_method longest"
        #           ]],

        # ["z3", []],
        # ["z3-noodler", ["smt.string_solver=\"noodler\""]],
        # ["woorpje", []],
        # ["ostrich", []],
        # ["cvc5", []],

        # gnn based configurations
        # branch: fixed
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method gnn",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method gnn_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method gnn_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_gnn_random",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_gnn_random_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_gnn_random_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method category_gnn",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method category_gnn_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method category_gnn_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_category_gnn_random",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_category_gnn_random_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["fixed", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_category_gnn_random_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],

        # branch: random
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method gnn",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method gnn_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method gnn_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_gnn_random",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_gnn_random_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_gnn_random_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method category_gnn",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method category_gnn_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method category_gnn_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_category_gnn_random",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_category_gnn_random_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_category_gnn_random_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],

        # branch: hybrid
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method gnn",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method gnn_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method gnn_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_gnn_random",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_gnn_random_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_gnn_random_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method category_gnn",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method category_gnn_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method category_gnn_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_category_gnn_random",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_category_gnn_random_first_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],
        ["this", ["hybrid_fixed_random", "--termination_condition termination_condition_0",
                  f"--graph_type {graph_type}",
                  f"--algorithm {algorithm}",
                  f"--order_equations_method hybrid_category_gnn_random_each_n_iterations",
                  f"--gnn_model_path " + model_folder + f"model_0_{graph_type}_{model_type}.pth",
                  f"--gnn_task {task}",
                  f"--rank_task {rank_task}"]],




    ]

    benchmark_dict = {
        # "test_track": bench_folder + "/test",
        # "example_track": bench_folder + "/examples",
        # "track_01": bench_folder + "/01_track",
        # "g_track_01_sat":bench_folder + "/01_track_generated/SAT",
        # "g_track_01_mixed": bench_folder + "/01_track_generated/mixed",
        # "g_track_01_eval":bench_folder + "/01_track_generated_eval_data",
        # "track_02": bench_folder + "/02_track",
        # "track_03": bench_folder + "/03_track",
        # "track_04": bench_folder + "/04_track",
        # "track_05": bench_folder + "/05_track",
        # "track_random_train": bench_folder + "/random_track_train",
        # "track_random_eval": bench_folder + "/random_track_eval",
        # "track_01_generated_SAT_train": bench_folder + "/01_track_generated_SAT_train/ALL",
        # "track_01_generated_SAT_eval": bench_folder + "/01_track_generated_SAT_eval",
    }

    benchmark_folder = benchmark_name + "/ALL"
    folder_number = sum(
        [1 for fo in os.listdir(bench_folder + "/" + benchmark_folder) if "divided" in os.path.basename(fo)])
    for i in range(folder_number):
        divided_folder_index = i + 1
        benchmark_dict[benchmark_name + "_divided_" + str(
            divided_folder_index)] = bench_folder + "/" + benchmark_folder + "/divided_" + str(divided_folder_index)

    configuration_list = []
    for solver_param in solver_param_list:
        solver = solver_param[0]
        parameters_list = solver_param[1]

        for benchmark_name, benchmark_folder in benchmark_dict.items():
            configuration_list.append(
                {"solver": solver, "parameters_list": parameters_list, "benchmark_name": benchmark_name,
                 "benchmark_folder": benchmark_folder, "summary_folder_name": benchmark_name + "_summary"})

    # Writing the dictionary to a JSON file
    configuration_folder = project_folder + "/src/process_benchmarks/configurations"
    write_configurations_to_json_file(configuration_folder=configuration_folder, configurations=configuration_list)

    print("done")


if __name__ == '__main__':
    main()
