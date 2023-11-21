from .DataTypes import Variable, Terminal, Term, Assignment,Equation,get_eq_graph_1,get_eq_graph_2,get_eq_graph_3,get_eq_graph_4,get_eq_graph_5
from typing import Dict, List
from .Constants import INTERNAL_TIMEOUT
import sys


graph_func_map = {None: get_eq_graph_1, "graph_1": get_eq_graph_1,
                      "graph_2": get_eq_graph_2, "graph_3":get_eq_graph_3, "graph_4":get_eq_graph_4,
                      "graph_5":get_eq_graph_5}




def print_results(result: Dict):
    if result["result"] == None:
        print("result: "+INTERNAL_TIMEOUT)
    else:
        print("-" * 10, "Problem", "-" * 10)
        print("recursion limit number", sys.getrecursionlimit())
        original_equation_list, string_terminals, string_variables = assemble_parsed_content(result)
        print("Variables:", string_variables)
        print("Terminals:", string_terminals)
        for e in original_equation_list:
            print("Equation:", e)


        print("-" * 10, "Solution", "-" * 10)

        satisfiability = result["result"]
        assignment = result["assignment"]

        solved_string_equation_list, _, _ = assemble_parsed_content(result, assignment)

        if satisfiability == "SAT":
            print("result: SAT")
            assignment.pretty_print()
            for se in solved_string_equation_list:
                print("Solved Equation:", se)
        elif satisfiability == "UNSAT":
            print("result: UNSAT")
        else:
            print("result:",satisfiability)

        if "total_explore_paths_call" in result:
            print(f'Total explore_paths call: {result["total_explore_paths_call"]}')

    print(f'Algorithm runtime in seconds: {result["running_time"]}')


def assemble_parsed_content(result: Dict, assignment: Assignment = Assignment()):
    string_equation_list=[]
    for eq in result["equation_list"]:
        string_equation=assemble_one_equation(eq.left_terms, eq.right_terms, assignment)
        string_equation_list.append(string_equation)

    string_terminals = get_terminal_string(result["terminals"])
    string_variables = get_variable_string(result["variables"])

    return string_equation_list, string_terminals, string_variables

def get_terminal_string(terminals: List[Terminal]):
    return ",".join([t.value for t in terminals])
def get_variable_string(variables: List[Variable]):
    return ",".join([t.value for t in variables])

def assemble_one_equation(left_terms:List[Term], right_terms:List[Term], assignment: Assignment = Assignment()):
    left_str = []
    right_str = []
    for t in left_terms:
        if type(t.value) == Variable:
            if assignment.is_empty():
                left_str.append(t.get_value_str)
            else:
                terminal_list = assignment.get_assignment(t.value)
                for tt in terminal_list:
                    left_str.append(tt.value)
        else:
            left_str.append(t.get_value_str)
    for t in right_terms:
        if type(t.value) == Variable:
            if assignment.is_empty():
                right_str.append(t.get_value_str)
            else:
                terminal_list = assignment.get_assignment(t.value)
                for tt in terminal_list:
                    right_str.append(tt.value)
        else:
            right_str.append(t.get_value_str)

    left_terms_str = "".join(left_str) if len(left_str) != 0 else "\"\""
    right_terms_str = "".join(right_str) if len(right_str) != 0 else "\"\""

    string_equation = left_terms_str + " = " + right_terms_str
    return string_equation


