import sys
import configparser
from typing import List, Dict

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

import sys
from src.solver.Constants import project_folder

sys.path.append(project_folder)
from src.solver.Constants import bench_folder, recursion_limit
from src.solver.independent_utils import get_folders, color_print

import zipfile
import fnmatch
from tqdm import tqdm
import json
import hashlib

def main():
    benchmark = "choose_eq_train"
    graph_type="graph_1"

    benchmark_path = bench_folder + "/" + benchmark
    folder_list = [folder for folder in get_folders(benchmark_path) if
                   "divided" in folder or "valid" in folder]

    zip_file_name=graph_type
    hash_table=get_data_label_hash_table(benchmark_path, folder_list, zip_file_name)

    check_hash_table_label_consistency(hash_table)

    unify_labels(benchmark_path, folder_list, graph_type, hash_table)

    zip_file_name = f"modified_{graph_type}"
    hash_table = get_data_label_hash_table(benchmark_path, folder_list, zip_file_name)
    check_hash_table_label_consistency(hash_table)


def unify_labels(benchmark_path, folder_list, graph_type, hash_table):
    for folder in folder_list:
        current_folder = benchmark_path + "/" + folder
        input_zip_file_path = current_folder + f"/{graph_type}.zip"
        output_zip_file_path = current_folder + f"/modified_{graph_type}.zip"

        # Open the input zip file to read and a new zip file to write simultaneously
        with zipfile.ZipFile(input_zip_file_path, 'r') as input_zip, \
                zipfile.ZipFile(output_zip_file_path, 'w') as output_zip:
            for file_name in tqdm(input_zip.namelist(), desc="Modifying Files"):
                with input_zip.open(file_name) as file_content:
                    if fnmatch.fnmatch(file_name, '*.graph.json'):
                        # Load JSON data
                        json_data = json.loads(file_content.read().decode('utf-8'))

                        # Check hash table
                        G = []
                        for key, value in json_data.items():
                            if isinstance(value, dict):
                                G.append(value)
                        # check each data
                        for g in G:
                            one_data = [g] + G
                            hashed_data = hash_one_data(one_data)
                            # Check hash table
                            if hashed_data in hash_table:
                                label_count = hash_table[hashed_data]
                                # Modify the JSON data as needed
                                if label_count[0] > label_count[1]:
                                    g["label"] = 0
                                else:
                                    g["label"] = 1

                        # Convert modified dict back to JSON bytes
                        modified_content = json.dumps(json_data).encode('utf-8')
                    else:
                        # If not a JSON file, just read the content to write back unmodified
                        modified_content = file_content.read()

                    # Write either modified or original content to the new zip file
                    output_zip.writestr(file_name, modified_content, compress_type=zipfile.ZIP_DEFLATED)

    print("Modification complete and new zip file created.")

def get_data_label_hash_table(benchmark_path, folder_list,zip_file_name:str)->Dict:
    # get hash table
    hash_table: Dict = {}  # {hashed_data:label_count}, label_count={0:0,1:0}
    for folder in folder_list:
        current_folder = benchmark_path + "/" + folder
        train_data_zip_file = current_folder + f"/{zip_file_name}.zip"
        with zipfile.ZipFile(train_data_zip_file, 'r') as zip_file_content:
            for f in tqdm(zip_file_content.namelist(), desc="make hash table"):  # scan all files in zip
                if fnmatch.fnmatch(f, '*.graph.json'):
                    with zip_file_content.open(f) as json_file:
                        json_dict = json.loads(json_file.read())
                        # print(json_dict)
                        # Get all graphs to G
                        G = []
                        for key, value in json_dict.items():
                            if isinstance(value, dict):
                                G.append(value)
                        # hash one data to hash table
                        for g in G:
                            one_data = [g] + G
                            hashed_data = hash_one_data(one_data)
                            label = g["label"]

                            if hashed_data in hash_table:
                                label_count = hash_table[hashed_data]
                                if label in label_count:
                                    label_count[label] += 1
                                else:
                                    label_count[label] = 1
                                hash_table[hashed_data] = label_count
                            else:
                                label_count = {0: 0, 1: 0}
                                label_count[label] = 1
                                hash_table[hashed_data] = label_count
    return hash_table


def check_hash_table_label_consistency(hash_table,log=False):
    insistent_count=0
    consistent_count=0
    for key, value in hash_table.items():
        label_count = value
        if label_count[0] != 0 and label_count[1]!=0:
            insistent_count+=1
            if log==True:
                color_print(f"Inconsistent label count, {key}", "red")
                print(label_count)
        else:
            consistent_count+=1
            if log == True:
                color_print(f"Consistent label count, {key}", "green")
                print(label_count)
    print(f"insistent_count:{insistent_count}")
    print(f"consistent_count:{consistent_count}")



def hash_one_data(graph_list:List[Dict])->str:
    data_str = ""
    for index, g in enumerate(graph_list):
        data_str += f"nodes:{str(g['nodes'])}|node_types:{str(g['node_types'])}|edges:{str(g['edges'])},"
    data_str=remove_last_comma(data_str)

    # Hash the string representation
    hashed_data:str=hashlib.md5(data_str.encode()).hexdigest()


    return hashed_data


def remove_last_comma(s):
    # Check if the string ends with a comma
    if s.endswith(','):
        # Remove the last character (the comma)
        return s[:-1]
    return s

if __name__ == '__main__':
    main()
