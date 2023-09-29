from typing import Dict, List, Set, Tuple
from DataTypes import Variable, Terminal, Term, Assignment
from itertools import product
from Constants import max_variable_length


class Solver:
    def __init__(self):
        pass

    def generate_combinations(self, terminals: List[Terminal], max_length: int) -> List[Tuple[Terminal]]:
        combinations = []
        for length in range(1, max_length + 1):
            for p in product(terminals, repeat=length):
                combinations.append(p)
        return combinations

    def solve(self, string_equation: Dict) -> (bool, Assignment):
        variables: Set[Variable] = string_equation["variables"]
        terminals: Set[Terminal] = string_equation["terminals"]
        left_terms: List[Term] = string_equation["left_terms"]
        right_terms: List[Term] = string_equation["right_terms"]
        variable_list = list(variables)
        terminal_list = list(terminals)

        possible_terminals = self.generate_combinations(terminals, max_variable_length)
        print("possible_terminals", len(possible_terminals))
        print(possible_terminals)
        # Generate all possible combinations of assignments
        assignments_list = list(product(possible_terminals, repeat=len(variables)))
        print("assignments_list:", len(assignments_list))

        # Create a list of dictionaries to represent each assignment
        assignment_dicts = []
        for assignment in assignments_list:
            assignment_dict = Assignment()
            for var, term in zip(variable_list, assignment):
                assignment_dict.set_assignment(var, list(term))
            assignment_dicts.append(assignment_dict)

        # Display the list of assignment dictionaries
        # print("-" * 10)
        # print("Assignment Dictionaries:", len(assignment_dicts))
        # for assignment_dict in assignment_dicts:
        #     print(assignment_dict.assignments)


        # Check each assignment dictionary to see if it satisfies the equation
        for assignment_dict in assignment_dicts:
            if self.check_equation(left_terms, right_terms, assignment_dict):
                return True, assignment_dict

        return False, Assignment()

    def check_equation(self, left_terms: List[Term], right_terms: List[Term], assignments: Assignment) -> bool:
        left_side = self.extract_values_from_terms(left_terms, assignments)
        right_side = self.extract_values_from_terms(right_terms, assignments)

        # todo: this need to be improved
        left_str = "".join(left_side).replace("<EMPTY>", "")
        right_str = "".join(right_side).replace("<EMPTY>", "")
        if left_str == right_str:
            return True
        else:
            return False

    def extract_values_from_terms(self, term_list, assignments):
        value_list = []
        for t in term_list:
            if type(t.value) == Variable:
                terminal_list = assignments.get_assignment(t.value)
                for tt in terminal_list:
                    value_list.append(tt.value)
            else:  # type(t.value) == Terminal
                value_list.append(t.value.value)
        return value_list
