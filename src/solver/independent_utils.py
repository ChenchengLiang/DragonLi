from typing import Iterable, List,Callable
import json
import re
import os
import shutil
import psutil
import zipfile
import pickle
import time
import hashlib
from collections import OrderedDict
def check_list_consistence(target_list):
    consitence_list = []
    for one_answer in target_list:
        consitence_list.append(all(element == one_answer for element in target_list))
    return all(consitence_list)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage_bytes = process.memory_info().rss

    # Convert bytes to KB, MB, and GB
    memory_usage_kb = memory_usage_bytes / 1024
    memory_usage_mb = memory_usage_kb / 1024
    memory_usage_gb = memory_usage_mb / 1024

    if memory_usage_gb >= 1:
        # Format as GB + MB + KB
        return f"{memory_usage_gb:.2f} GB, {memory_usage_mb % 1024:.2f} MB, {memory_usage_kb % 1024:.2f} KB",memory_usage_gb
    elif memory_usage_mb >= 1:
        # Format as MB + KB
        return f"{memory_usage_mb:.2f} MB, {memory_usage_kb % 1024:.2f} KB",memory_usage_gb
    else:
        # Format as KB
        return f"{memory_usage_kb:.2f} KB",memory_usage_gb

def remove_duplicates(lst:Iterable)->List:
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def flatten_list(nested_list:Iterable)->List:
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def strip_file_name_suffix(file_name):
    return file_name.replace(".eq", "").replace(".smt2", "").replace(".smt", "")

def dump_to_json_with_format(graph_dict,file_path):
    # Dump each top-level item in graph_dict to a string
    items = [f'"{key}": {json.dumps(value)}' for key, value in graph_dict.items()]

    # Combine the items into the final JSON string
    formatted_json = "{\n" + ",\n".join(items) + "\n}"

    with open(file_path, 'w') as f:
        f.write(formatted_json)


def replace_primed_vars(v, e):
    # Find all unique capital letters followed by primes in both v and e
    patterns = set(re.findall(r"[A-Z]'+", v + e))

    # Identify available capital letters for replacement
    available_caps = identify_available_capitals(v + e)

    # Create a mapping from the old pattern to a new capital letter
    mapping = {pattern: available_caps.pop() for pattern in patterns}

    # Replace occurrences in v and e
    for old, new in mapping.items():
        #print("replace",old,"by", new)
        v = v.replace(old, new)
        e = e.replace(old, new)

    return v, e

def identify_available_capitals(s:str):
    # Identify available capital letters for replacement
    all_used_vars = set(re.findall(r"[A-Z]", s))
    all_caps = set(chr(i) for i in range(65, 91))
    available_caps = list(all_caps - all_used_vars)
    return available_caps

def write_configurations_to_json_file(configuration_folder,configurations):
    if os.path.exists(configuration_folder) == False:
        os.mkdir(configuration_folder)
    else:
        shutil.rmtree(configuration_folder)
        os.mkdir(configuration_folder)

    for i, config in enumerate(configurations):
        file_name = configuration_folder + "/config_" + str(i) + ".json"
        with open(file_name, 'w') as f:
            json.dump(config, f, indent=4)

def mean(l):
    if len(l)==0:
        return 0
    else:
        return sum(l)/len(l)

def compress_to_zip(pickle_file):
    zip_file = pickle_file + '.zip'
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pickle_file, os.path.basename(pickle_file))
    os.remove(pickle_file)  # Optionally remove the original pickle file


def zip_folder(folder_path, output_zip_file):
    # Create a ZIP file in write mode
    with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through directory
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create a relative path to the file to preserve the folder structure
                relative_path = os.path.relpath(os.path.join(root, file), os.path.dirname(folder_path))
                # Add the file to the ZIP file
                zipf.write(os.path.join(root, file), arcname=relative_path)


def save_to_pickle(dataset, file_path):
    """
    Save a dataset to a pickle file.

    :param dataset: The dataset to be saved.
    :param file_path: Path to the pickle file where the dataset will be stored.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file)

def load_from_pickle(file_path):
    """
    Load a dataset from a pickle file.

    :param file_path: Path to the pickle file to be loaded.
    :return: Loaded dataset or None if file does not exist.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        return None

def load_from_pickle_within_zip(zip_file_path, pickle_file_name):
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(pickle_file_name, 'r') as file:
            return pickle.load(file)

def time_it(func:Callable):
    def wrapper(*args, **kwargs):
        print(f"----- Function {func.__name__} starts -----")
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        print(f"----- Function {func.__name__} finished, took {end_time - start_time} seconds to run -----")
        return result
    return wrapper

def color_print(text,color):
    if color=="red":
        print("\033[31m"+text+"\033[0m")
    elif color=="green":
        print("\033[32m"+text+"\033[0m")
    elif color=="yellow":
        print("\033[33m"+text+"\033[0m")
    elif color=="blue":
        print("\033[34m"+text+"\033[0m")
    elif color=="purple":
        print("\033[35m"+text+"\033[0m")
    elif color=="cyan":
        print("\033[36m"+text+"\033[0m")
    elif color=="white":
        print("\033[37m"+text+"\033[0m")
    elif color=="black":
        print("\033[30m"+text+"\033[0m")
    else:
        print(text)

@time_it
def handle_duplicate_files(directory, log=False):
    """
    Delete duplicate files in the specified directory, print the number of deletions,
    and return a list of deleted file paths.

    :param directory: Path to the directory to search for duplicate files.
    :return: List of paths of the deleted duplicate files.
    """

    def file_hash(filepath):
        """Compute hash of a file."""
        hash_func = hashlib.md5()
        with open(filepath, 'rb') as file:
            for chunk in iter(lambda: file.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    hashes = {}
    deleted_files = []

    duplicate_folder=os.path.dirname(directory)+"/duplicated_eqs"
    if os.path.exists(duplicate_folder):
        shutil.rmtree(duplicate_folder)
    os.mkdir(duplicate_folder)

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            filehash = file_hash(filepath)
            if filehash in hashes:
                original_files = ", ".join(hashes[filehash])
                if log==True:
                    print(f"Move duplicate file: {filepath} to {duplicate_folder} (duplicates: {original_files})")
                #os.remove(filepath)
                shutil.move(filepath,duplicate_folder)
                deleted_files.append(filepath)
            else:
                hashes[filehash] = hashes.get(filehash, []) + [filepath]

    # Print the total number of duplicate files deleted
    print(f"Total duplicate files: {len(deleted_files)}")

    return deleted_files
@time_it
def handle_files_with_target_string(directory, target_string,move_to_folder_name="empty_eq", log=False):
    """
    Delete files in the specified directory that contain the given target string, print the number of deletions,
    and return a list of deleted file paths.

    :param directory: Path to the directory to search in.
    :param target_string: The content to search for within the files.
    :return: List of paths of the deleted files.
    """

    def contains_target(file_path, target):
        """Check if a file contains the target string."""
        with open(file_path, 'r') as file:
            return target in file.read()

    deleted_files = []

    target_folder=os.path.dirname(directory)+"/"+move_to_folder_name
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.mkdir(target_folder)


    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and contains_target(file_path, target_string):
            if log == True:
                print(f"Move file: {filename} to {target_folder}")
            #os.remove(file_path)
            shutil.move(file_path,target_folder)
            deleted_files.append(file_path)

    # Print the total number of files deleted
    print(f"Total files moved: {len(deleted_files)}\nTarget string:\n{target_string}")


    return deleted_files

def apply_to_all_files(directory, operation_function):
    """
    Apply a given operation to all files in a specified directory.

    :param directory: Path to the directory.
    :param operation_function: Function that defines the operation to be applied to each file.
    """
    print(f"Applying operation {operation_function.__name__} to all files in {directory}")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            operation_function(file_path)
            #print(f"Processed file: {filename}")

def delete_duplicate_lines(file_path):
    """Remove duplicate lines from a file while preserving the original order."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Using OrderedDict to preserve order and remove duplicates
    unique_lines = list(OrderedDict.fromkeys(lines))

    with open(file_path, 'w') as file:
        file.writelines(unique_lines)


def get_folders(path):
    """
    Returns a list of folder names found in the given directory path.
    Only folders will be returned, not files.
    """
    folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return folders


def find_leaf_folders(root_folder):
    """
    Find all leaf folders in the given directory.

    :param root_folder: The root directory to search in.
    :return: A list of paths to all leaf folders.
    """
    leaf_folders = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        # If 'dirnames' is empty, this is a leaf folder
        if not dirnames:
            leaf_folders.append(dirpath)

    return leaf_folders

def create_folder(folder_path):
    if os.path.exists(folder_path)==False:
        print("create folder",folder_path)
        os.mkdir(folder_path)
    else:
        print("folder existed, remove and create new one",folder_path)
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    return folder_path