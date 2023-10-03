from DataTypes import Variable, Terminal, Term, Assignment
from typing import Dict, List,Union


def print_results(result:Union[bool, Assignment,None], running_time:float, parsed_content:Dict):
    print("-" * 10, "Problem", "-" * 10)
    original_equation,string_terminals, string_variables = assemble_parsed_content(parsed_content)
    print("Variables:", string_variables)
    print("Terminals:", string_terminals)
    print("Equation:", original_equation)

    print("-" * 10, "Solution", "-" * 10)

    if result is None:
        print("TIMEOUT")
    elif result == "max_variable_length_exceeded":
        print("MAX VARIABLE LENGTH EXCEEDED")
    else:
        (satisfiability, assignment) = result
        solved_string_equation, _, _ = assemble_parsed_content(parsed_content, assignment)

        if satisfiability == True:
            print("SAT")
            assignment.pretty_print()
            print(solved_string_equation)

    print(f'Algorithm runtime in seconds: {running_time}')




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
