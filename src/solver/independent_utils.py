from typing import Iterable, List
import json

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
