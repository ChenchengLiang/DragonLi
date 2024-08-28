import configparser
import os
import sys

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

import random
import string
from src.solver.Constants import bench_folder
from copy import deepcopy
from src.solver.independent_utils import remove_duplicates, identify_available_capitals, strip_file_name_suffix
from src.train_data_collection.utils import dvivde_track_for_cluster
import shutil
from src.process_benchmarks.eq2smt_utils import one_eq_file_to_smt2
from typing import List, Tuple


def main():
    # generate track
    start_idx = 1
    end_idx = 5
    track_name = f"01_track_multi_word_equations_generated_eval_eq_number_20_rank_task_{start_idx}_{end_idx}"
    track_folder = bench_folder + "/" + track_name
    save_equations(start_idx, end_idx, track_folder, track_name, generate_one_track_4)
    # divide tracks
    dvivde_track_for_cluster(track_folder, chunk_size=50)

    print("done")


def save_equations(start_index, end_index, folder, track_name, equation_generator):
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        shutil.rmtree(folder)
        os.mkdir(folder)
    all_folder = folder + "/ALL"
    if not os.path.exists(all_folder):
        os.makedirs(all_folder)
    for i in range(start_index, end_index + 1):  # +1 because range is exclusive at the end
        print("---", str(i), "----")
        filename = os.path.join(all_folder, f"g_{track_name}_{i}.eq")
        equation_str, _, _, _ = equation_generator(filename, i)
        with open(filename, 'w') as file:
            file.write(equation_str)
        # generate smt2 file
        one_eq_file_to_smt2(filename)


def generate_one_track_1(file_name, index, max_variables=15, max_terminals=10, max_length=300,
                         write_replacement_log=False):
    _, terminals = get_variables_and_terminals(max_variables=max_variables, max_terminals=max_terminals)

    # Create a random string of the terminals
    random_string = ''.join(random.choice(terminals) for _ in range(random.randint(1, max_length)))
    equation_left = deepcopy(random_string)
    equation_right = deepcopy(random_string)

    replacement_log = ""
    replaced_left, replaced_right, variables, replacement_log = replace_substring_with_new_variables(equation_left,
                                                                                                     equation_right,
                                                                                                     replacement_log)
    replacement_log = replacement_log + "\n ----- results ----- \n"
    replacement_log = replacement_log + f"random_string {len(random_string)}: \n {random_string} \n"
    replacement_log = replacement_log + f"replaced_left {len(replaced_left)}: \n {replaced_left} \n"
    replacement_log = replacement_log + f"replaced_right {len(replaced_right)}: \n {replaced_right} \n"
    replacement_log = replacement_log + f"variables: {variables}"

    replacement_log_file = strip_file_name_suffix(file_name) + ".replacement_log"
    if write_replacement_log == True:
        with open(replacement_log_file, 'w') as file:
            file.write(replacement_log)

    # Format the result
    result = formatting_results(''.join(variables), ''.join(terminals), [(replaced_left, replaced_right)])

    return result, variables, terminals, [(replaced_left, replaced_right)]


def replace_substring_with_new_variables(left, right, replacement_log):
    variables = []
    left_length = len(left)
    right_length = len(right)
    max_replace_variable_length = 5
    max_replace_time = 5

    replacement_log = replacement_log + f"-lhs- \n"
    replace_time = random.randint(0, max_replace_time)
    replacement_log = replacement_log + f"replace_time {replace_time}\n"
    for i in range(replace_time):
        replacement_log = replacement_log + f"- {i} - \n"
        # substring index
        random_start_index = random.randint(0, left_length)
        random_substring_length = random.randint(0, left_length - random_start_index)
        random_end_index = random_start_index + random_substring_length
        # generate random variables
        available_variables = identify_available_capitals("".join(variables))
        if len(available_variables) <= 1:
            break
        random_variables = generate_random_variables(available_variables, max_random_variables_length=len(
            available_variables) % max_replace_variable_length)
        # replace substring
        replacement_log = replacement_log + f"before: {left}  \n"
        replacement_log = replacement_log + f"replace: {left[random_start_index:random_end_index]} by {random_variables} \n"
        left = left[:random_start_index] + random_variables + left[random_end_index:]
        replacement_log = replacement_log + f"after: {left}  \n"

        # update length
        left_length = len(left)
        # update variables
        variables = remove_duplicates([x for x in left + right if x.isupper()])
        variable_str = "".join(variables)
        replacement_log = replacement_log + f"variables: {variable_str}  \n"

    replacement_log = replacement_log + f"\n"

    replacement_log = replacement_log + f"-rhs- \n"
    replace_time = random.randint(0, max_replace_time)
    replacement_log = replacement_log + f"replace_time {replace_time}\n"
    for i in range(replace_time):
        replacement_log = replacement_log + f"- {i} - \n"
        # substring index
        random_start_index = random.randint(0, right_length)
        random_substring_length = random.randint(0, right_length - random_start_index)
        random_end_index = random_start_index + random_substring_length
        # generate random variables
        available_variables = identify_available_capitals("".join(variables))
        if len(available_variables) <= 1:
            break
        random_variables = generate_random_variables(available_variables, max_random_variables_length=len(
            available_variables) % max_replace_variable_length)

        # replace substring
        replacement_log = replacement_log + f"before: {right}  \n"
        replacement_log = replacement_log + f"replace: {right[random_start_index:random_end_index]} by {random_variables} \n"
        right = right[:random_start_index] + random_variables + right[random_end_index:]
        replacement_log = replacement_log + f"after: {right}  \n"
        # update length
        right_length = len(right)
        # update variables
        variables = remove_duplicates([x for x in left + right if x.isupper()])
        variable_str = "".join(variables)
        replacement_log = replacement_log + f"variables: {variable_str}  \n"

    return left, right, variables, replacement_log


def generate_random_variables(available_variables, max_random_variables_length=5):
    if max_random_variables_length == 0:
        max_random_variables_length = 1
    random_variables_length = random.randint(1, max_random_variables_length)
    random_variables = random.sample(available_variables, random_variables_length)
    random_variables = "".join(random_variables)
    return random_variables


def generate_one_random(max_variables=15, max_terminals=10, max_length=50):
    variables, terminals = get_variables_and_terminals(max_variables=max_variables, max_terminals=max_terminals)

    left_side_length = random.randint(1, max_length / 2)
    right_side_length = random.randint(1, max_length / 2)
    # Create a random string of the terminals
    random_left_string = ''.join(random.choice(terminals + variables) for _ in range(left_side_length))
    random_right_string = ''.join(random.choice(terminals + variables) for _ in range(right_side_length))

    # Format the result
    result = formatting_results(''.join(variables), ''.join(terminals), [(random_left_string, random_right_string)])

    return result, variables, terminals, [(random_left_string, random_right_string)]


def generate_one_track_2(file_name, index):
    # Generate variable and terminal sets
    num_variables = index
    variables = [string.ascii_uppercase[i] for i in range(num_variables)]
    terminals = ["a", "b"]

    # Create a equation with the partten "X_{n}aX_{n}bX_{n-1}bX_{n-2}\cdots bX_{1} = aX_{n}X_{n-1}X_{n-1}bX_{n-2}X_{n-2}b \cdots b X_{1}X_{1}baa"
    left_terms = []
    right_terms = []
    for i, v in enumerate(variables):
        if i == 0:
            left_terms.extend([v, "a", v])
            right_terms.extend(["a", v])
        else:
            left_terms.extend(["b", v])
            right_terms.extend([v, v, "b"])
    right_terms.extend(["a", "a"])
    equation_left = "".join(left_terms)
    equation_right = "".join(right_terms)

    # Format the result
    result = formatting_results(''.join(variables), ''.join(terminals), [(equation_left, equation_right)])

    return result, variables, terminals, [(equation_left, equation_right)]


def generate_one_track_3(file_name, index):
    track_2_eq, variables, terminals, eqs = generate_one_track_2(file_name, random.randint(2, 15))
    equation_left = eqs[0][0]
    equation_right = eqs[0][1]

    def process_one_hand_side(one_hand_side_str):
        new_one_hand_side_list = []
        # replace each b with lhs or rhs of eq from track 1
        for item in one_hand_side_str:
            if item == "b":
                _, _, _, eqs = generate_one_track_1(file_name, index, max_variables=24,max_terminals=24,max_length=20,write_replacement_log=False)
                l = eqs[0][0]
                r = eqs[0][1]
                replaced_item = random.choice([l, r])
                new_one_hand_side_list.append(replaced_item)
            else:
                new_one_hand_side_list.append(item)
        new_one_hand_side_str = "".join(new_one_hand_side_list)
        return new_one_hand_side_str

    # replace each b with lhs or rhs of eq from track 1

    new_equation_left_str = process_one_hand_side(equation_left)
    new_equation_right_str = process_one_hand_side(equation_right)
    variables = remove_duplicates([x for x in new_equation_left_str + new_equation_right_str if x.isupper()])
    terminals = remove_duplicates([x for x in new_equation_left_str + new_equation_right_str if x.islower()])

    result = formatting_results(''.join(variables), ''.join(terminals),
                                [(new_equation_left_str, new_equation_right_str)])

    return result, variables, terminals, [(new_equation_left_str, new_equation_right_str)]


def generate_conjunctive_track_03(file_name, index):
    eq_number = random.randint(2, 20)
    eq_list = []
    variable_list = []
    terminal_list = []
    for i in range(eq_number):
        result, variables, terminals, eqs = generate_one_track_3(file_name, index)
        left_str = eqs[0][0]
        right_str = eqs[0][1]
        variable_list.extend(variables)
        terminal_list.extend(terminals)
        variable_list = remove_duplicates(variable_list)
        terminal_list = remove_duplicates(terminal_list)
        eq_list.append((left_str, right_str))
    result = formatting_results(''.join(variable_list), ''.join(terminal_list), eq_list)
    return result, variable_list, terminal_list, eq_list

def generate_one_track_4(file_name, index):
    # "01_track_multi_word_equations_generated_eval_1001_2000"
    max_eq=20
    max_variables=10
    max_terminals=6
    one_side_max_length=50
    eq_number = random.randint(1, max_eq)
    eq_list = []
    variable_list = []
    terminal_list = []
    for i in range(eq_number):
        # result, variables, terminals, eqs=generate_one_random(max_variables=10, max_terminals=6, max_length=60)
        # result, variables, terminals, eqs = generate_one_track_3(file_name,index)
        result, variables, terminals, eqs = generate_one_track_1(file_name, index, max_variables=max_variables, max_terminals=max_terminals,
                                                                 max_length=one_side_max_length, write_replacement_log=False)
        left_str = eqs[0][0]
        right_str = eqs[0][1]
        variable_list.extend(variables)
        terminal_list.extend(terminals)
        variable_list = remove_duplicates(variable_list)
        terminal_list = remove_duplicates(terminal_list)
        eq_list.append((left_str, right_str))
    result = formatting_results(''.join(variable_list), ''.join(terminal_list), eq_list)

    if index == 1:
        track_info_str = f"max_eq={max_eq} \n max_variables={max_variables}\n max_terminals={max_terminals}\n max_length={one_side_max_length}"
        #output track_info_str to file
        track_info_file = f"{os.path.dirname(os.path.dirname(file_name))}/track_info.txt"
        with open(track_info_file, 'w') as file:
            file.write(track_info_str)

    return result, variable_list, terminal_list, eq_list


def generate_one_SAT_multi_word_equation_track(file_name, index):
    max_one_side_element = 10
    max_assignment_length = 10
    eq_number = random.randint(1, 50)
    eq_list = []
    variable_list = []
    terminal_list = []

    # generate assingments
    terminals = string.ascii_lowercase
    variable_to_terminal_map = {}

    for index, letter in enumerate(string.ascii_uppercase, start=1):
        variable_to_terminal_map[letter] = ''.join(
            random.choice(terminals) for _ in range(random.randint(0, max_assignment_length)))
    terminals_to_variable = {value: key for key, value in variable_to_terminal_map.items()}
    print("variable_to_terminal_map", variable_to_terminal_map)

    eq_string_list = []
    for i in range(eq_number):
        left_side_length = random.randint(1, max_one_side_element)
        left_string_list = [random.choice(list(terminals_to_variable.keys())) for _ in range(left_side_length)]
        right_string_list = left_string_list
        eq_string_list.append((left_string_list, right_string_list))

    for eq in eq_string_list:
        replaced_left_list = [terminals_to_variable[e] if random.randint(0, 10) >= 5 else e for e in eq[0]]
        replaced_right_list = [terminals_to_variable[e] if random.randint(0, 10) >= 5 else e for e in eq[1]]
        left_string = ''.join(replaced_left_list)
        right_string = ''.join(replaced_right_list)
        print("original: ", "".join(eq[0]), " = ", "".join(eq[1]))
        print("replaced: ", "".join(replaced_left_list), " = ", "".join(replaced_right_list))

        if left_string != right_string:
            left_variables = [char for char in left_string if char.isupper()]
            right_variables = [char for char in right_string if char.isupper()]
            left_terminals = [char for char in left_string if char.islower()]
            right_terminals = [char for char in right_string if char.islower()]
            terminal_list.extend(left_terminals)
            terminal_list.extend(right_terminals)
            terminal_list = remove_duplicates(terminal_list)
            variable_list.extend(left_variables)
            variable_list.extend(right_variables)
            variable_list = remove_duplicates(variable_list)
            eq_list.append((left_string, right_string))

    result = formatting_results(''.join(variable_list), ''.join(terminal_list), eq_list)
    return result, variable_list, terminal_list, eq_list


def formatting_results(variables: List[str], terminals: List[str], eq_list: List[Tuple[str, str]]) -> str:
    # Format the result
    result = f"Variables {{{''.join(variables)}}}\n"
    result += f"Terminals {{{''.join(terminals)}}}\n"
    for eq in eq_list:
        result += f"Equation: {eq[0]} = {eq[1]}\n"
    result += "SatGlucose(100)"
    return result


def get_variables_and_terminals(max_variables=15, max_terminals=10):
    # Choose a number of variables and terminals
    num_variables = random.randint(1, max_variables)
    num_terminals = random.randint(1, max_terminals)

    # Generate variable and terminal sets
    variables = [string.ascii_uppercase[i] for i in range(num_variables)]
    terminals = random.sample(string.ascii_lowercase, num_terminals)
    return variables, terminals


if __name__ == '__main__':
    main()
