from itertools import product
from typing import List, Dict, Tuple, Generator
from src.solver.DataTypes import Assignment, Terminal
from.abstract_algorithm import AbstractAlgorithm

class EnumerateAssignmentsUsingGenerator(AbstractAlgorithm):
    def __init__(self, terminals, variables, left_terms, right_terms, parameters: Dict):
        super().__init__(terminals, variables, left_terms, right_terms)
        self.max_variable_length = parameters["max_variable_length"]

    def generate_possible_terminal_combinations(self, terminals: List[str], max_length: int) -> Generator[
        Tuple[str, ...], None, None]:
        for length in range(1, max_length + 1):
            for p in product(terminals, repeat=length):
                yield p

    def generate_assignments(self, variables, terminals, max_variable_length):
        possible_terminals = self.generate_possible_terminal_combinations(terminals, max_variable_length)

        # Generate all possible combinations of assignments
        assignments_generator = product(possible_terminals, repeat=len(variables))

        for assignment in assignments_generator:
            assignment_dict = Assignment()
            for var, term in zip(variables, assignment):
                assignment_dict.set_assignment(var, list(term))
            yield assignment_dict

    def run(self):
        assignment_generator = self.generate_assignments(self.variables, self.terminals, self.max_variable_length)

        # Check each assignment dictionary to see if it satisfies the equation
        for assignment in assignment_generator:
            if self.check_equation(self.left_terms, self.right_terms, assignment):
                return {"result": True, "assignment": assignment, "left_terms": self.left_terms,
                        "right_terms": self.right_terms,
                        "variables": self.variables, "terminals": self.terminals}

        return {"result": "max_variable_length_exceeded", "assignment": assignment,
                "left_terms": self.left_terms, "right_terms": self.right_terms, "variables": self.variables,
                "terminals": self.terminals}


class EnumerateAssignments(AbstractAlgorithm):
    def __init__(self, terminals, variables, left_terms, right_terms, parameters):
        super().__init__(terminals, variables, left_terms, right_terms)
        self.max_variable_length = parameters["max_variable_length"]

    def generate_possible_terminal_combinations(self, terminals: List[Terminal], max_length: int) -> List[
        Tuple[Terminal]]:
        combinations: List[Tuple[Terminal]] = []
        for length in range(1, max_length + 1):
            for p in product(terminals, repeat=length):
                combinations.append(p)  # type: ignore
        return combinations

    def run(self):
        possible_terminals = self.generate_possible_terminal_combinations(self.terminals, self.max_variable_length)
        print("possible_terminals", len(possible_terminals))
        print(possible_terminals)
        # Generate all possible combinations of assignments
        assignments_list = list(product(possible_terminals, repeat=len(self.variables)))
        print("assignments_list:", len(assignments_list))

        # Create a list of dictionaries to represent each assignment
        assignment_dicts = []
        for assignment in assignments_list:
            assignment_dict = Assignment()
            for var, term in zip(self.variables, assignment):
                assignment_dict.set_assignment(var, list(term))
            assignment_dicts.append(assignment_dict)

        # Display the list of assignment dictionaries
        print("-" * 10)
        print("Assignment Dictionaries:", len(assignment_dicts))
        for assignment_dict in assignment_dicts:
            print(assignment_dict.assignments)

        # Check each assignment dictionary to see if it satisfies the equation
        for assignment in assignment_dicts:
            if self.check_equation(self.left_terms, self.right_terms, assignment):
                return {"result": True, "assignment": assignment, "left_terms": self.left_terms,
                        "right_terms": self.right_terms,
                        "variables": self.variables, "terminals": self.terminals}

        return {"result": "max_variable_length_exceeded", "assignment": assignment,
                "left_terms": self.left_terms, "right_terms": self.right_terms, "variables": self.variables,
                "terminals": self.terminals}
