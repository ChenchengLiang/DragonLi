from .DataTypes import Variable, Terminal, Term, Assignment
from typing import Dict, List, Union, Iterable
from .Constants import INTERNAL_TIMEOUT
import sys
def print_results(result: Dict):
    if result["result"] == None:
        print("result: "+INTERNAL_TIMEOUT)
    else:
        print("-" * 10, "Problem", "-" * 10)
        print("recursion limit number", sys.getrecursionlimit())
        original_equation, string_terminals, string_variables = assemble_parsed_content(result)
        print("Variables:", string_variables)
        print("Terminals:", string_terminals)
        print("Equation:", original_equation)

        print("-" * 10, "Solution", "-" * 10)

        satisfiability = result["result"]
        assignment = result["assignment"]

        solved_string_equation, _, _ = assemble_parsed_content(result, assignment)

        if satisfiability == True:
            print("result: SAT")
            assignment.pretty_print()
            print(solved_string_equation)
        elif satisfiability == False:
            print("result: UNSAT")
        else:
            print("result:", satisfiability)

        if "total_explore_paths_call" in result:
            print(f'Total explore_paths call: {result["total_explore_paths_call"]}')

    print(f'Algorithm runtime in seconds: {result["running_time"]}')


def assemble_parsed_content(result: Dict, assignment: Assignment = Assignment()):
    left_str = []
    right_str = []
    for t in result["left_terms"]:
        if type(t.value) == Variable:
            if assignment.is_empty():
                left_str.append(t.value.value)
            else:
                terminal_list = assignment.get_assignment(t.value)
                for tt in terminal_list:
                    left_str.append(tt.value)
        else:
            left_str.append(t.value.value)
    for t in result["right_terms"]:
        if type(t.value) == Variable:
            if assignment.is_empty():
                right_str.append(t.value.value)
            else:
                terminal_list = assignment.get_assignment(t.value)
                for tt in terminal_list:
                    right_str.append(tt.value)
        else:
            right_str.append(t.value.value)

    left_terms_str = "".join(left_str) if len(left_str)!=0 else "\"\""
    right_terms_str = "".join(right_str) if len(right_str)!=0 else "\"\""

    string_equation = left_terms_str + " = " + right_terms_str

    string_terminals = ",".join([t.value for t in result["terminals"]])
    string_variables = ",".join([t.value for t in result["variables"]])

    return string_equation, string_terminals, string_variables


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
