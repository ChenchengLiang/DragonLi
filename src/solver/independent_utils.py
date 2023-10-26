from typing import Iterable, List
import json
import re
import os
import shutil

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

def identify_available_capitals(s):
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
