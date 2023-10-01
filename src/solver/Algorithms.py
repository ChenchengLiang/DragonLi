from abc import ABC, abstractmethod
from itertools import product
from DataTypes import Assignment, Term, Terminal, Variable
from typing import List, Dict, Tuple, Generator


class AbstractAlgorithm(ABC):
    def __init__(self, terminals, variables, left_terms, right_terms):
        self.terminals = terminals
        self.variables = variables
        self.left_terms = left_terms
        self.right_terms = right_terms

    @abstractmethod
    def run(self):
        pass
    
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



class ElimilateVariables(AbstractAlgorithm):
    def __init__(self, terminals, variables, left_terms, right_terms):
        super().__init__(terminals, variables, left_terms, right_terms)

    def run(self):
        #todo: implement this
        return False, Assignment()


class EnumerateAssignmentsUsingGenerator(AbstractAlgorithm):
    def __init__(self, terminals, variables, left_terms, right_terms, max_variable_length):
        super().__init__(terminals, variables, left_terms, right_terms)
        self.max_variable_length = max_variable_length

    def generate_possible_terminal_combinations(self, terminals: List[str], max_length: int) -> Generator[Tuple[str, ...], None, None]:
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
                return True, assignment

        return False, Assignment()


class EnumerateAssignments(AbstractAlgorithm):
    def __init__(self, terminals, variables, left_terms, right_terms, max_variable_length):
        super().__init__(terminals, variables, left_terms, right_terms)
        self.max_variable_length = max_variable_length

    def generate_possible_terminal_combinations(self, terminals: List[Terminal], max_length: int) -> List[Tuple[Terminal]]:
        combinations = []
        for length in range(1, max_length + 1):
            for p in product(terminals, repeat=length):
                combinations.append(p)
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
                return True, assignment

        return False, Assignment()
