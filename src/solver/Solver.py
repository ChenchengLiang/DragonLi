from typing import Dict, List, Set, Tuple,Generator
from DataTypes import Variable, Terminal, Term, Assignment
from itertools import product
from Constants import max_variable_length
from Algorithms import AbstractAlgorithm, EnumerateAssignments,EnumerateAssignmentsUsingGenerator
import functools

class Solver:
    def __init__(self,algorithm:AbstractAlgorithm):
        self.algorithm=algorithm
        pass

    def generate_combinations(self, terminals: List[str], max_length: int) -> Generator[Tuple[str, ...], None, None]:
        for length in range(1, max_length + 1):
            for p in product(terminals, repeat=length):
                yield p

    def generate_assignments(self, variables, terminals, max_variable_length):
        possible_terminals = self.generate_combinations(terminals, max_variable_length)

        # Generate all possible combinations of assignments
        assignments_generator = product(possible_terminals, repeat=len(variables))

        for assignment in assignments_generator:
            assignment_dict = Assignment()  #
            for var, term in zip(variables, assignment):
                assignment_dict.set_assignment(var, list(term))
            yield assignment_dict

    def solve(self, string_equation: Dict) -> (bool, Assignment):
        variables: Set[Variable] = string_equation["variables"]
        terminals: Set[Terminal] = string_equation["terminals"]
        left_terms: List[Term] = string_equation["left_terms"]
        right_terms: List[Term] = string_equation["right_terms"]

        return self.algorithm(terminals, variables, left_terms, right_terms, max_variable_length).run()




