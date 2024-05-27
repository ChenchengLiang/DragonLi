import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

import sys
from src.solver.Constants import project_folder

sys.path.append(project_folder)
from src.solver.Constants import bench_folder, recursion_limit
from src.solver.independent_utils import get_folders

import zipfile
import fnmatch
from tqdm import tqdm
import json

def main():
    benchmark = "choose_eq_train"

    benchmark_path = bench_folder + "/" + benchmark
    folder_list = [folder for folder in get_folders(benchmark_path) if
                   "divided" in folder or "valid" in folder]
    for folder in folder_list:
        current_folder = benchmark_path + "/" + folder
        train_data_zip_file = current_folder + "/graph_1.zip"
        with zipfile.ZipFile(train_data_zip_file, 'r') as zip_file_content:
            for f in tqdm(zip_file_content.namelist(), desc="output_rank_eq_graphs"):  # scan all files in zip
                if fnmatch.fnmatch(f, '*.graph.json'):
                    with zip_file_content.open(f) as json_file:
                        json_dict = json.loads(json_file.read())
                        print(json_dict)



def hash_graph(graph_dict):
    pass




if __name__ == '__main__':
    main()
