from DataTypes import Variable, Terminal, Term, Assignment
from typing import Dict, List


def print_results(satisfiability: bool, assignment: Assignment, parsed_content):
    print("-" * 10)
    original_equation,string_terminals, string_variables = assemble_parsed_content(parsed_content)
    solved_string_equation, _, _ = assemble_parsed_content(parsed_content, assignment)
    print("Variables:", string_variables)
    print("Terminals:", string_terminals)
    print("Equation:", original_equation)
    if satisfiability == True:
        print("SAT")
        assignment.pretty_print()
        print(solved_string_equation)

    else:
        print("UNSAT")


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

    string_terminals = "".join([t.value for t in parsed_content["terminals"] ])
    string_variables = "".join([t.value for t in parsed_content["variables"] ])


    return string_equation, string_terminals, string_variables


def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
