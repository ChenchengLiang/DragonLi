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

    def solve(self, string_equation: Dict) -> (bool,Assignment):
        variables: Set[Variable] = string_equation["variables"]
        terminals: Set[Terminal] = string_equation["terminals"]
        left_terms: List[Term] = string_equation["left_terms"]
        right_terms: List[Term] = string_equation["right_terms"]

        # Generate all possible combinations of assignments
        assignments_list = list(product(terminals, repeat=len(variables)))

        # Create a list of dictionaries to represent each assignment
        assignment_dicts = []
        for assignment in assignments_list:
            assignment_dict = Assignment()
            for var, term in zip(variables, assignment):
                assignment_dict.set_assignment(var,term)
            assignment_dicts.append(assignment_dict)

        # Display the list of assignment dictionaries
        for assignment_dict in assignment_dicts:
            print(assignment_dict.assignments)

        # Check each assignment dictionary to see if it satisfies the equation
        for assignment_dict in assignment_dicts:
            if self.check_equation(left_terms, right_terms, assignment_dict):
                return True,assignment_dict


        return False,Assignment()


    def check_equation(self, left_terms: List[Term], right_terms: List[Term], assignments: Assignment) -> bool:
        left_side = []
        right_side = []
        for lt in left_terms:
            if type(lt.value) == Variable:
                left_side.append(assignments.get_assignment(lt.value).value)
            else:
                left_side.append(lt.value.value)

        for rt in right_terms:
            if type(rt.value) == Variable:
                right_side.append(assignments.get_assignment(rt.value).value)
            else:
                right_side.append(rt.value.value)

        #this need to be improved
        if "".join(left_side) == "".join(right_side):
            return True
        else:
            return False

