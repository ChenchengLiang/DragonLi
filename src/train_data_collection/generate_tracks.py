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
from src.solver.DataTypes import formatting_results, formatting_results_v2
import json


def main():
    # generate track
    start_idx = 1
    end_idx = 1000
    # track_name = f"01_track_multi_word_equations_eq_2_50_generated_train_{start_idx}_{end_idx}"
    track_name = f"04_track_DragonLi_test_generate_one_random_{start_idx}_{end_idx}"
    track_folder = bench_folder + "/" + track_name
    save_equations(start_idx, end_idx, track_folder, track_name, generate_one_track_4)
    # save_equations(start_idx, end_idx, track_folder, track_name, generate_one_track_4_v2)
    #save_equations(start_idx, end_idx, track_folder, track_name, generate_one_track_4_v3)
    # save_equations(start_idx, end_idx, track_folder, track_name, generate_one_track_1_v2)

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
        equation_str, variable_list, terminal_list, eq_list = equation_generator(filename, i)
        # generate eq file
        if len(variable_list) > 26 or len(terminal_list) > 26:
            pass
        else:
            equation_str = formatting_results(variable_list, terminal_list, eq_list)
        with open(filename, 'w') as file:
            file.write(equation_str)
        # generate smt2 file
        one_eq_file_to_smt2(filename)


def generate_one_track_1(file_name, index, max_variables=15, max_terminals=10, max_length=300,
                         write_replacement_log=False, terminal_pool=None, variable_pool=None, fixed_length=60,
                         substring_length=5, substring_variable_map={}, eq_string_list=[], eq_index=0):
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


def generate_letter_pool(max_length, use_uppercase=True, custom_letters=None):
    # Choose a number of variables or terminals
    # length = random.randint(1, max_length)
    length = max_length

    if custom_letters:
        letters = custom_letters
    else:
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if use_uppercase else 'abcdefghijklmnopqrstuvwxyz'

    num_letters = len(letters)
    pool = []

    for i in range(length):
        letter_index = i % num_letters
        number = i // num_letters  # generate the index for A_1, A_2, A_3, ...

        if number == 0:
            pool.append(letters[letter_index])
        else:
            pool.append(f"{letters[letter_index]}{number}")

    return pool


def generate_one_track_1_v2(file_name, index, max_variables=15, max_terminals=10,
                            max_length=300, terminal_pool=None,
                            variable_pool=None, fixed_length=60, substring_length=5, substring_variable_map={},
                            eq_string_list=[], eq_index=0):
    terminal_pool = generate_letter_pool(max_terminals, use_uppercase=False)
    variable_pool = generate_letter_pool(max_variables, use_uppercase=True)
    random_terminal_list = [random.choice(terminal_pool) for _ in range(random.randint(1, max_length))]
    eq_left = random_terminal_list
    eq_right = random_terminal_list

    # replace terminals with variables
    replaced_left, replaced_right = replace_substring_with_new_variables_v2(eq_left, eq_right, variable_pool)

    result = formatting_results_v2(variable_pool, terminal_pool, [(replaced_left, replaced_right)])

    return result, variable_pool, terminal_pool, [(replaced_left, replaced_right)]


def replace_one_side_by_substring_map_v1(one_side, substring, substring_variable_map):
    number_of_substring = one_side.count(substring)
    if number_of_substring > 1:
        # substring_indexes = _substring_indexes(one_side, substring)
        # s_list = list(one_side)
        # for i in substring_indexes:
        #     if random.random() < 0.5:
        #         s_list[i:i + len(substring)] = list(substring_variable_map[substring])
        #         substring_indexes = [x - len(substring) for x in substring_indexes]
        # one_side = ''.join(s_list)

        for i in range(number_of_substring):
            if random.random() < 0.5:
                one_side = _random_replace(one_side, substring, substring_variable_map[substring], 1)


    elif number_of_substring == 1:
        if random.random() < 0.5:
            one_side = one_side.replace(substring, substring_variable_map[substring])
    else:
        pass

    return one_side


def replace_one_side_by_substring_map(one_side, substring, substring_variable_map, reversed_substring_variable_map):
    if random.random() < 0.5:
        number_of_substring = one_side.count(substring)
        if number_of_substring > 1:
            replace_time = random.randint(1, number_of_substring)
            one_side = _random_replace(one_side, substring, substring_variable_map[substring], replace_time)
        elif number_of_substring == 1:
            one_side = one_side.replace(substring, substring_variable_map[substring])
        else:
            # recover the variable to substring in substring_variable_map until current substring is found
            recovered_ones_side = one_side
            out_loop_break = False
            for key in reversed_substring_variable_map:
                for i in range(recovered_ones_side.count(key)):
                    recovered_ones_side = recovered_ones_side.replace(key, reversed_substring_variable_map[key], 1)
                    if recovered_ones_side.count(substring) == 1:
                        recovered_ones_side.replace(substring, substring_variable_map[substring])
                        out_loop_break = True
                        break
                if out_loop_break == True:
                    break
            if out_loop_break == True:
                one_side = recovered_ones_side

    return one_side


def _substring_indexes(s, target):
    # Find all positions where the target substring occurs
    indices = []
    start = 0
    while start < len(s):
        start = s.find(target, start)
        if start == -1:
            break
        indices.append(start)
        start += 1
    return indices


def _random_replace(s, target, replacement, n):
    # Find all positions where the target substring occurs
    indices = _substring_indexes(s, target)

    # Ensure we have enough occurrences to replace
    if len(indices) < n:
        # raise ValueError("Not enough occurrences of the substring to replace")
        return s  # don't replace if not enough occurrences

    # Shuffle the indices
    random.shuffle(indices)

    # Perform replacements at the first n shuffled positions
    s_list = list(s)  # Convert string to a list for easier mutation
    for i in range(n):
        index = indices[i]
        s_list[index:index + len(target)] = list(replacement)

    # Convert the list back to a string
    return ''.join(s_list)


def generate_one_track_1_woorpje(file_name, index, max_variables=15,
                                 max_terminals=10, max_length=300, terminal_pool=None, variable_pool=None,
                                 fixed_length=60, substring_length=5, substring_variable_map={}, eq_string_list=[],
                                 eq_index=0):
    # random_terminal_list = [random.choice(terminal_pool) for _ in range(random.randint(1, max_length))]
    # random_terminal_list = [random.choice(terminal_pool) for _ in range(fixed_length)]
    # eq_left = random_terminal_list
    # eq_right = random_terminal_list
    #
    # # replace terminals with variables
    # replaced_left, replaced_right = replace_substring_with_new_variables_v2(eq_left, eq_right, variable_pool,substring_length=substring_length)
    #

    # generate random string with fixed length
    # random_string=generate_random_string(fixed_length, terminal_pool)

    # read random string from pre generated random string
    random_string = eq_string_list[eq_index]
    eq_left = random_string
    eq_right = random_string

    reversed_substring_variable_map = {v: k for k, v in substring_variable_map.items()}
    # for each replacable substring, replace with variable by 0.5 probability
    shuffled_substring_variable_map = list(substring_variable_map.items())
    random.shuffle(shuffled_substring_variable_map)
    for substring, v in shuffled_substring_variable_map:
        # randomly replace substring with variable
        eq_left = replace_one_side_by_substring_map(eq_left, substring, substring_variable_map,
                                                    reversed_substring_variable_map)
        eq_right = replace_one_side_by_substring_map(eq_right, substring, substring_variable_map,
                                                     reversed_substring_variable_map)

    replaced_left = list(eq_left)
    replaced_right = list(eq_right)

    updated_variable_pool = remove_duplicates([x for x in replaced_left + replaced_right if x in variable_pool])
    updated_terminal_pool = remove_duplicates([x for x in replaced_left + replaced_right if x in terminal_pool])

    # formatting the result replaced_left:List[str], replaced_right:List[str]
    result = formatting_results_v2(updated_variable_pool, updated_terminal_pool, [(replaced_left, replaced_right)])

    return result, updated_variable_pool, updated_terminal_pool, [(replaced_left, replaced_right)]


def replace_sublist(terminal_list, replacement_variable_list, start_index, end_index):
    terminal_list[start_index:end_index + 1] = replacement_variable_list
    return terminal_list


def replace_substring_with_new_variables_v2(original_left_list, original_right_list, variable_pool, substring_length=5):
    max_replace_variable_length = 1
    max_replace_time = 5
    left_list = original_left_list.copy()
    right_list = original_right_list.copy()

    replace_time = random.randint(0, max_replace_time)
    for i in range(replace_time):
        left_list = replace_one_side(left_list, variable_pool, max_replace_variable_length,
                                     substring_length=substring_length)

    for i in range(replace_time):
        right_list = replace_one_side(right_list, variable_pool, max_replace_variable_length,
                                      substring_length=substring_length)

    return left_list, right_list


def replace_one_side(original_list, variable_pool, max_replace_variable_length, substring_length=5):
    # substring index
    random_start_index = random.randint(0, len(original_list))
    # random_substring_length = random.randint(0, len(original_list) - random_start_index)
    # substring_length=random_substring_length
    random_end_index = random_start_index + substring_length

    # choose the replacement variables not in existing variables
    current_variable_list = [t for t in original_list if t in variable_pool]
    available_variable_pool = [v for v in variable_pool if v not in current_variable_list]
    if available_variable_pool == []:
        return original_list

    replacement_variable_list = [random.choice(available_variable_pool) for _ in range(max_replace_variable_length)]

    original_list = replace_sublist(original_list, replacement_variable_list, random_start_index,
                                    random_end_index)
    return original_list


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
                _, _, _, eqs = generate_one_track_1(file_name, index, max_variables=24, max_terminals=24, max_length=20,
                                                    write_replacement_log=False)
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


def generate_random_string(length, pool):
    return ''.join(random.choice(pool) for _ in range(length))


def generate_random_substring(s):
    if len(s) > 1:
        length = random.randint(1, len(s) - 1)  # Randomly select a length between 1 and the length of the string
        start_index = random.randint(0, len(s) - length)  # Randomly select a starting index
        return s[start_index:start_index + length]  # Extract the substring
    else:
        return s


def generate_one_track_4_v3(file_name, index):
    log = True
    min_eq = 1
    max_eq = 100
    eq_number = random.randint(min_eq, max_eq)
    max_variables = 10
    max_terminals = 6
    one_side_max_length = 60
    fixed_one_side_length = random.randint(2, one_side_max_length)
    variable_pool, terminal_pool = get_variables_and_terminals(max_variables=max_variables,
                                                               max_terminals=max_terminals)

    # generate each eq string
    eq_string_list = []
    for i in range(eq_number):
        random_string = generate_random_string(fixed_one_side_length, terminal_pool)
        eq_string_list.append((random_string, random_string))


    # decide replacement substring
    substring_variable_map = {}
    for i, v in enumerate(variable_pool):
        # select one side from an eq string
        eq_sample = random.choice(eq_string_list)
        one_side_of_eq_sample = random.choice(eq_sample)
        # select a random substring from one_side_of_eq_sample
        substring_to_be_replaced = generate_random_substring(one_side_of_eq_sample)
        substring_variable_map[substring_to_be_replaced] = v
        replaced_eq_string_list = []
        for eq in eq_string_list:
            left_string = replace_one_side_by_substring_map_v1(eq[0], substring_to_be_replaced, substring_variable_map)
            right_string = replace_one_side_by_substring_map_v1(eq[1], substring_to_be_replaced, substring_variable_map)
            replaced_eq_string_list.append((left_string, right_string))

        eq_string_list = replaced_eq_string_list

    # formatting the result
    variable_list = []
    terminal_list = []
    eq_list = []
    for eq in eq_string_list:
        variable_list.extend([v for v in variable_pool if v in eq[0] + eq[1]])
        terminal_list.extend([t for t in terminal_pool if t in eq[0] + eq[1]])
        eq_list.append((eq[0], eq[1]))

    variable_list = remove_duplicates(variable_list)
    terminal_list = remove_duplicates(terminal_list)

    result = formatting_results_v2(variable_list, terminal_list, eq_list)

    # output track_info_file
    track_info_file = f"{os.path.dirname(os.path.dirname(file_name))}/track_info.txt"
    if not os.path.exists(track_info_file):
        track_info_str = (
            f"min_eq={min_eq}\nmax_eq={max_eq}\nmax_variables={max_variables}\nmax_terminals={max_terminals}\nmax_length={one_side_max_length}")
        # output track_info_str to file

        with open(track_info_file, 'w') as file:
            file.write(track_info_str)

    if log == True:
        # output substring_variable_map to file
        substring_variable_map_file_name = f"{strip_file_name_suffix(file_name)}.json"
        reversed_substring_variable_map = {v: k for k, v in substring_variable_map.items()}
        with open(substring_variable_map_file_name, 'w') as file:
            json.dump(reversed_substring_variable_map, file, indent=4)

        # debug info
        if len(variable_list) == 1:
            print("one variable")
        if len(terminal_list) == 1:
            print("one terminal")
        if len(variable_list) == 0:
            print("no variable")
        if len(terminal_list) == 0:
            print("no terminal")

    return result, variable_list, terminal_list, eq_list


def generate_one_track_4_v2(file_name, index):
    # new setting
    # min_eq = 2
    # max_eq = 50
    # max_variables = 10
    # max_terminals = 10
    # one_side_max_length = 50
    # terminal_pool = None
    # variable_pool = None
    # track_1_func=generate_one_track_1_v2 # letter pool = max_length

    # old setting
    # min_eq = 1
    # max_eq = 100
    # max_variables = 10
    # max_terminals = 6
    # one_side_max_length = 60
    # terminal_pool=None
    # variable_pool=None
    # track_1_func=generate_one_track_1 # letter pool = random.randint(1, max_length)

    # woorpje setting
    min_eq = 1
    max_eq = 100
    eq_number = random.randint(min_eq, max_eq)
    max_variables = 10
    max_terminals = 6
    one_side_max_length = 60
    fixed_one_side_length = random.randint(2, one_side_max_length)
    substring_length = random.randint(1, fixed_one_side_length)
    variable_pool, terminal_pool = get_variables_and_terminals(max_variables=max_variables,
                                                               max_terminals=max_terminals)

    # todo: for debug fix variable and terminal pool
    # variable_pool = list("BAC")
    # terminal_pool = list("dcab")
    # fixed_length = 8
    # eq_number = 117

    # generate each eq string
    eq_string_list = []
    for i in range(eq_number):
        eq_string_list.append(generate_random_string(fixed_one_side_length, terminal_pool))

    # decide replacement substring
    substring_variable_map = {}
    eq_sample_size = min(len(eq_string_list), len(variable_pool))
    eq_sample = random.sample(eq_string_list, eq_sample_size)
    for i, v in enumerate(variable_pool):
        # select a random substring from the eq string
        substring_to_be_replaced = generate_random_substring(eq_sample[i % eq_sample_size])
        substring_variable_map[substring_to_be_replaced] = v

    # todo:for debug fix variable and terminal pool
    # substring_variable_map={"dc":"B","aadc":"A","aadcb":"C"}

    track_1_func = generate_one_track_1_woorpje

    eq_list = []
    variable_list = []
    terminal_list = []
    for i in range(eq_number):
        result, variables, terminals, eq = track_1_func(file_name, index,
                                                        max_variables=max_variables,
                                                        max_terminals=max_terminals,
                                                        max_length=one_side_max_length,
                                                        terminal_pool=terminal_pool,
                                                        variable_pool=variable_pool,
                                                        fixed_length=fixed_one_side_length,
                                                        substring_length=substring_length,
                                                        substring_variable_map=substring_variable_map,
                                                        eq_string_list=eq_string_list, eq_index=i)
        left_list = eq[0][0]
        right_list = eq[0][1]
        temp_variable_list = [v for v in variables if v in left_list + right_list]
        temp_terminal_list = [t for t in terminals if t in left_list + right_list]
        variable_list.extend(temp_variable_list)
        terminal_list.extend(temp_terminal_list)
        variable_list = remove_duplicates(variable_list)
        terminal_list = remove_duplicates(terminal_list)
        eq_list.append((left_list, right_list))

    result = formatting_results_v2(variable_list, terminal_list, eq_list)

    track_info_file = f"{os.path.dirname(os.path.dirname(file_name))}/track_info.txt"
    if not os.path.exists(track_info_file):
        track_info_str = (
            f"min_eq={min_eq}\nmax_eq={max_eq}\nmax_variables={max_variables}\nmax_terminals={max_terminals}\nmax_length={one_side_max_length}")
        # output track_info_str to file

        with open(track_info_file, 'w') as file:
            file.write(track_info_str)

    # output substring_variable_map to file
    substring_variable_map_file_name = f"{strip_file_name_suffix(file_name)}.json"
    reversed_substring_variable_map = {v: k for k, v in substring_variable_map.items()}
    with open(substring_variable_map_file_name, 'w') as file:
        json.dump(reversed_substring_variable_map, file, indent=4)

    if len(variable_list) == 1:
        print("one variable")
    if len(terminal_list) == 1:
        print("one terminal")
    if len(variable_list) == 0:
        print("no variable")
    if len(terminal_list) == 0:
        print("no terminal")

    return result, variable_list, terminal_list, eq_list


def generate_one_track_4(file_name, index):
    # "01_track_multi_word_equations_generated_eval_1001_2000"
    min_eq = 1
    max_eq = 100
    max_variables = 10
    max_terminals = 6
    one_side_max_length = 60
    eq_number = random.randint(min_eq, max_eq)
    eq_list = []
    variable_list = []
    terminal_list = []
    for i in range(eq_number):
        result, variables, terminals, eqs=generate_one_random(max_variables=10, max_terminals=6, max_length=60)
        # result, variables, terminals, eqs = generate_one_track_3(file_name,index)
        # result, variables, terminals, eqs = generate_one_track_1(file_name, index, max_variables=max_variables,
        #                                                          max_terminals=max_terminals,
        #                                                          max_length=one_side_max_length,
        #                                                          write_replacement_log=False)
        left_str = eqs[0][0]
        right_str = eqs[0][1]
        variable_list.extend(variables)
        terminal_list.extend(terminals)
        variable_list = remove_duplicates(variable_list)
        terminal_list = remove_duplicates(terminal_list)
        eq_list.append((left_str, right_str))
    result = formatting_results(''.join(variable_list), ''.join(terminal_list), eq_list)

    if index == 1:
        track_info_str = (
            f"min_eq={min_eq}\n max_eq={max_eq} \n max_variables={max_variables}\n max_terminals={max_terminals}\n max_length={one_side_max_length}")
        # output track_info_str to file
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


def get_variables_and_terminals(max_variables=15, max_terminals=10):
    # Choose a number of variables and terminals
    num_variables = random.randint(1, max_variables)
    num_terminals = random.randint(1, max_terminals)

    # Generate variable and terminal sets
    variables = [string.ascii_uppercase[i] for i in range(num_variables)]  # start by A
    terminals = [string.ascii_lowercase[i] for i in range(num_terminals)]  # start by a
    # terminals = random.sample(string.ascii_lowercase, num_terminals) #not start by a
    return variables, terminals


if __name__ == '__main__':
    main()
