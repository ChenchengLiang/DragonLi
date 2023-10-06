from .DataTypes import Variable, Terminal, Term, Assignment
from typing import Dict, List,Union


def print_results(result:Dict):
    print("-" * 10, "Problem", "-" * 10)
    original_equation,string_terminals, string_variables = assemble_parsed_content(result)
    print("Variables:", string_variables)
    print("Terminals:", string_terminals)
    print("Equation:", original_equation)

    print("-" * 10, "Solution", "-" * 10)

    satisfiability = result["result"]
    assignment = result["assignment"]

    if satisfiability is None:
        print("result: INTERNAL TIMEOUT")
    elif satisfiability == "max_variable_length_exceeded":
        print("result: MAX VARIABLE LENGTH EXCEEDED")
    else:
        solved_string_equation, _, _ = assemble_parsed_content(result, assignment)

        if satisfiability == True:
            print("result: SAT")
            assignment.pretty_print()
            print(solved_string_equation)

        if satisfiability == False:
            print("result: UNSAT")

    print(f'Algorithm runtime in seconds: {result["running_time"]}')




def assemble_parsed_content(parsed_content: Dict, assignment: Assignment = Assignment()):
    left_str = []
    right_str = []
    for t in parsed_content["left_terms"]:
        if type(t.value) == Variable:
            if assignment.is_empty():
                left_str.append(t.value.value)
            else:
                terminal_list = assignment.get_assignment(t.value)
                for tt in terminal_list:
                    left_str.append(tt.value)
        else:
            left_str.append(t.value.value)
    for t in parsed_content["right_terms"]:
        if type(t.value) == Variable:
            if assignment.is_empty():
                right_str.append(t.value.value)
            else:
                terminal_list = assignment.get_assignment(t.value)
                for tt in terminal_list:
                    right_str.append(tt.value)
        else:
            right_str.append(t.value.value)
    string_equation = "".join(left_str) + "=" + "".join(right_str)

    string_terminals = ",".join([t.value for t in parsed_content["terminals"] ])
    string_variables = ",".join([t.value for t in parsed_content["variables"] ])


    return string_equation, string_terminals, string_variables


def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened