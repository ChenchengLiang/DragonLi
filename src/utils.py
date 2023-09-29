from DataTypes import Variable, Terminal, Term, Assignment
from typing import Dict, List


def print_results(satisfiability: bool, assignment: Assignment, parsed_content):
    print("-" * 10)
    print("Equation:", assemble_parsed_content(parsed_content))
    if satisfiability == True:
        print("SAT")
        print("assignment:", assignment.assignments)
        string_equation = assemble_parsed_content(parsed_content, assignment)
        print(string_equation)
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
                left_str.append(assignment.get_assignment(t.value).value)
        else:
            left_str.append(t.value.value)
    for t in parsed_content["right_terms"]:
        if type(t.value) == Variable:
            if assignment.is_empty():
                right_str.append(t.value.value)
            else:
                right_str.append(assignment.get_assignment(t.value).value)
        else:
            right_str.append(t.value.value)
    string_equation = "".join(left_str) + "=" + "".join(right_str)

    return string_equation
