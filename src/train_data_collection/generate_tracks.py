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

def main():
    track_1_mixed_folder = bench_folder+"/01_track_generated/SAT_200_for_eval"
    start_idx = 1001
    end_idx = 1200
    save_equations(start_idx, end_idx, track_1_mixed_folder, "01_track_SAT",generate_one_track_1)




def generate_one_random(max_variables=15, max_terminals=10, max_length=50):
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


def generate_one_track_2(num_variables=1):
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



def save_equations(start_index, end_index, folder, track_name,equation_generator):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(start_index, end_index + 1):  # +1 because range is exclusive at the end
        equation = equation_generator()
        filename = os.path.join(folder, f"g_{track_name}_{i}.eq")
        with open(filename, 'w') as file:
            file.write(equation)


def generate_equation(max_variables=15, max_terminals=10, max_length=300, unsat=False):

    variables,terminals=get_variables_and_terminals(max_variables=max_variables, max_terminals=max_terminals)

    # Create a random string of the terminals
    random_string = ''.join(random.choice(terminals) for _ in range(max_length))

    # Create a dictionary to store variable replacement values
    var_replacements = {}

    # Replace parts of the string with variables randomly
    for var in variables:
        start_index = random.randint(0, max_length - 1)
        end_index = random.randint(start_index, max_length)
        substring = random_string[start_index:end_index]

        # Replace and store the replacement value
        random_string = random_string.replace(substring, var, 1)
        var_replacements[var] = substring

    # Create the equation string
    equation_left = random_string
    equation_right = equation_left
    for var, value in var_replacements.items():
        equation_right = equation_right.replace(var, value, 1)

    if unsat:
        # Make a single modification to the right side to make the equation UNSAT
        random_index = random.randint(0, len(equation_right) - 1)
        random_character = random.choice(terminals)
        equation_right = equation_right[:random_index] + random_character + equation_right[random_index+1:]


    # Format the result
    result = f"Variables {{{''.join(variables)}}}\n"
    result += f"Terminals {{{''.join(terminals)}}}\n"
    result += f"Equation: {equation_left} = {equation_right}\n"
    result += "SatGlucose(100)"

    return result

def generate_one_track_1():
    return generate_equation(unsat=False)

def generate_one_track_1_unsat():
    return generate_equation(unsat=True)



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