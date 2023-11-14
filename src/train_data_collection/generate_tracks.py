import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

import random
import string
from src.solver.Constants import project_folder,bench_folder
from copy import deepcopy
from src.solver.independent_utils import remove_duplicates,identify_available_capitals,strip_file_name_suffix
from src.train_data_collection.utils import dvivde_track_for_cluster
import glob
import shutil

def main():
    # generate tracks
    track_1_sat_folder = bench_folder+"/01_track_generated_SAT_eval"
    start_idx = 1001
    end_idx = 1200
    save_equations(start_idx, end_idx, track_1_sat_folder, "01_track_SAT",generate_one_track_1)

    # divide tracks
    dvivde_track_for_cluster(track_1_sat_folder,chunk_size=50)







def save_equations(start_index, end_index, folder, track_name,equation_generator):
    if not os.path.exists(folder):
        os.makedirs(folder)
    all_folder=folder+"/ALL"
    if not os.path.exists(all_folder):
        os.makedirs(all_folder)
    for i in range(start_index, end_index + 1):  # +1 because range is exclusive at the end
        print("---",str(i),"----")
        filename = os.path.join(all_folder, f"g_{track_name}_{i}.eq")
        equation = equation_generator(filename)
        with open(filename, 'w') as file:
            file.write(equation)



def generate_one_track_1(file_name,max_variables=15, max_terminals=10, max_length=300):
    _,terminals=get_variables_and_terminals(max_variables=max_variables, max_terminals=max_terminals)

    # Create a random string of the terminals
    random_string = ''.join(random.choice(terminals) for _ in range(random.randint(1,max_length)))
    equation_left = deepcopy(random_string)
    equation_right = deepcopy(random_string)

    replacement_log=""
    replaced_left,replaced_right,variables,replacement_log=replace_substring_with_new_variables(equation_left,equation_right,replacement_log)
    replacement_log = replacement_log+ "\n ----- results ----- \n"
    replacement_log = replacement_log + f"random_string {len(random_string)}: \n {random_string} \n"
    replacement_log=replacement_log+f"replaced_left {len(replaced_left)}: \n {replaced_left} \n"
    replacement_log = replacement_log + f"replaced_right {len(replaced_right)}: \n {replaced_right} \n"
    replacement_log=replacement_log + f"variables: {variables}"

    replacement_log_file=strip_file_name_suffix(file_name)+".replacement_log"
    with open(replacement_log_file, 'w') as file:
        file.write(replacement_log)


    # Format the result
    result = f"Variables {{{''.join(variables)}}}\n"
    result += f"Terminals {{{''.join(terminals)}}}\n"
    result += f"Equation: {replaced_left} = {replaced_right}\n"
    result += "SatGlucose(100)"

    return result



def replace_substring_with_new_variables(left,right,replacement_log):
    variables=[]
    left_length=len(left)
    right_length=len(right)
    max_replace_variable_length=5
    max_replace_time=5


    replacement_log=replacement_log+f"-lhs- \n"
    replace_time = random.randint(0, max_replace_time)
    replacement_log = replacement_log + f"replace_time {replace_time}\n"
    for i in range(replace_time):
        replacement_log=replacement_log+f"- {i} - \n"
        #substring index
        random_start_index=random.randint(0,left_length)
        random_substring_length=random.randint(0,left_length-random_start_index)
        random_end_index=random_start_index+random_substring_length
        #generate random variables
        available_variables=identify_available_capitals("".join(variables))
        if len(available_variables)<=1:
            break
        random_variables=generate_random_variables(available_variables,max_random_variables_length=len(available_variables)%max_replace_variable_length)
        #replace substring
        replacement_log = replacement_log + f"before: {left}  \n"
        replacement_log = replacement_log + f"replace: {left[random_start_index:random_end_index]} by {random_variables} \n"
        left = left[:random_start_index] + random_variables + left[random_end_index:]
        replacement_log = replacement_log + f"after: {left}  \n"


        #update length
        left_length=len(left)
        #update variables
        variables=remove_duplicates([x for x in left+right if x.isupper()])
        variable_str = "".join(variables)
        replacement_log = replacement_log + f"variables: {variable_str}  \n"

    replacement_log = replacement_log + f"\n"

    replacement_log=replacement_log+f"-rhs- \n"
    replace_time = random.randint(0, max_replace_time)
    replacement_log = replacement_log + f"replace_time {replace_time}\n"
    for i in range(replace_time):
        replacement_log=replacement_log+f"- {i} - \n"
        #substring index
        random_start_index=random.randint(0,right_length)
        random_substring_length=random.randint(0,right_length-random_start_index)
        random_end_index=random_start_index+random_substring_length
        #generate random variables
        available_variables=identify_available_capitals("".join(variables))
        if len(available_variables)<=1:
            break
        random_variables=generate_random_variables(available_variables,max_random_variables_length=len(available_variables)%max_replace_variable_length)

        #replace substring
        replacement_log = replacement_log + f"before: {right}  \n"
        replacement_log = replacement_log + f"replace: {right[random_start_index:random_end_index]} by {random_variables} \n"
        right = right[:random_start_index] + random_variables + right[random_end_index:]
        replacement_log = replacement_log + f"after: {right}  \n"
        #update length
        right_length=len(right)
        #update variables
        variables=remove_duplicates([x for x in left+right if x.isupper()])
        variable_str="".join(variables)
        replacement_log = replacement_log + f"variables: {variable_str}  \n"


    return left, right,variables,replacement_log


def generate_random_variables(available_variables,max_random_variables_length=5):
    if max_random_variables_length ==0:
        max_random_variables_length=1
    random_variables_length = random.randint(1, max_random_variables_length)
    random_variables = random.sample(available_variables, random_variables_length)
    random_variables = "".join(random_variables)
    return random_variables


def generate_one_random(file_name,max_variables=15, max_terminals=10, max_length=50):
    variables,terminals=get_variables_and_terminals(max_variables=max_variables, max_terminals=max_terminals)

    # Create a random string of the terminals
    random_left_string = ''.join(random.choice(terminals+variables) for _ in range(max_length))
    random_right_string = ''.join(random.choice(terminals + variables) for _ in range(max_length))

    # Format the result
    result = f"Variables {{{''.join(variables)}}}\n"
    result += f"Terminals {{{''.join(terminals)}}}\n"
    result += f"Equation: {random_left_string} = {random_right_string}\n"
    result += "SatGlucose(100)"

    return result


def generate_one_track_2(file_name,num_variables=1):
    # Generate variable and terminal sets
    variables = [string.ascii_uppercase[i] for i in range(num_variables)]
    terminals = ["a", "b"]

    # Create a equation with the partten "X_{n}aX_{n}bX_{n-1}bX_{n-2}\cdots bX_{1} = aX_{n}X_{n-1}X_{n-1}bX_{n-2}X_{n-2}b \cdots b X_{1}X_{1}baa"
    left_terms = []
    right_terms = []
    for i,v in enumerate(variables):
        if i==0:
            left_terms.extend([v,"a",v])
            right_terms.extend(["a",v])
        else:
            left_terms.extend(["b",v])
            right_terms.extend([v,v,"b"])
    right_terms.extend(["a", "a"])
    equation_left = "".join(left_terms)
    equation_right = "".join(right_terms)


    # Format the result
    result = f"Variables {{{''.join(variables)}}}\n"
    result += f"Terminals {{{''.join(terminals)}}}\n"
    result += f"Equation: {equation_left} = {equation_right}\n"
    result += "SatGlucose(100)"

    return result



def get_variables_and_terminals(max_variables=15, max_terminals=10):
    # Choose a number of variables and terminals
    num_variables = random.randint(1, max_variables)
    num_terminals = random.randint(1, max_terminals)

    # Generate variable and terminal sets
    variables = [string.ascii_uppercase[i] for i in range(num_variables)]
    terminals = random.sample(string.ascii_lowercase, num_terminals)
    return variables,terminals

if __name__ == '__main__':
    main()