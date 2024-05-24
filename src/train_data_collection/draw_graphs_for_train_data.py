import configparser
import sys

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.independent_utils import get_folders
from src.solver.Constants import bench_folder,recursion_limit
from src.train_data_collection.utils import draw_graph_for_one_folder
import argparse


def main():
    # draw graphs from train folder
    sys.setrecursionlimit(recursion_limit)

    # read graph type from command line
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('graph_type', type=str, help='graph_type')
    args = arg_parser.parse_args()
    graph_type= args.graph_type

    # draw graphs for all folders
    # benchmark = "01_track_train_task_3_1_2000"
    # task = "task_3"

    benchmark_name = "choose_eq_train"#"choose_eq_train"
    benchmark_path=f"{bench_folder}/{benchmark_name}"

    task="rank_task_1"




    folder_list = [folder for folder in get_folders(benchmark_path) if
                   "divided" in folder or "valid" in folder]
    print(folder_list)
    if len(folder_list) != 0:
        for folder in folder_list:
            draw_graph_for_one_folder(graph_type, benchmark_path + "/" + folder, task)
    else:
        draw_graph_for_one_folder(graph_type, benchmark_path, task)


if __name__ == '__main__':
    main()
