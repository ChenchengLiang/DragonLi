import configparser
import os
import sys

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.independent_utils import zip_folder,get_folders
from src.solver.utils import graph_func_map
from src.solver.Constants import bench_folder,recursion_limit
from src.train_data_collection.utils import output_eq_graphs, output_pair_eq_graphs, output_split_eq_graphs,output_rank_eq_graphs
import shutil
import argparse


def main():
    # draw graphs from train folder
    sys.setrecursionlimit(recursion_limit)

    # read graph type from command line
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('graph_type', type=str, help='graph_type')
    args = arg_parser.parse_args()

    # draw graphs for all folders
    # benchmark = "01_track_train_task_3_1_2000"
    # task = "task_3"

    benchmark_name = "01_track_multi_word_equations_generated_train_1_40000_new_small_test"#"choose_eq_train"
    benchmark_path=f"{bench_folder}/{benchmark_name}"

    task="rank_task_1"




    folder_list = [folder for folder in get_folders(benchmark_path) if
                   "divided" in folder or "valid" in folder]
    print(folder_list)
    if len(folder_list) != 0:
        for folder in folder_list:
            draw_graph_for_one_folder(args,benchmark_path + "/" + folder,task)
    else:
        draw_graph_for_one_folder(args,benchmark_path,task)

def draw_graph_for_one_folder(args,benchmark_path,task):

    if task == "task_1":
        draw_func = output_eq_graphs  # task 1
    elif task == "task_2":
        draw_func = output_pair_eq_graphs  # task 2
    elif task == "task_3":
        draw_func = output_split_eq_graphs  # task 3
    elif task == "rank_task_1": #G:List[graph]
        draw_func = output_rank_eq_graphs


    train_eq_folder = benchmark_path+ "/train"
    train_zip_file=train_eq_folder+".zip"
    for graph_type in [args.graph_type]:
        # prepare folder
        graph_folder = benchmark_path + "/" + graph_type

        if os.path.exists(graph_folder):
            shutil.rmtree(graph_folder)
            os.mkdir(graph_folder)
        else:
            os.mkdir(graph_folder)

        # draw one type graphs
        print(f"- draw {graph_type} -")
        draw_func(zip_file=train_zip_file,graph_folder=graph_folder, graph_func=graph_func_map[graph_type], visualize=False)

        # compress
        zip_folder(folder_path=graph_folder, output_zip_file=graph_folder + ".zip")
        shutil.rmtree(graph_folder)

    print("done")


if __name__ == '__main__':
    main()
