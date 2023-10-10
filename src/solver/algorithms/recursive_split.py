import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import EMPTY_TERMINAL, BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED
from src.solver.DataTypes import Assignment, Term, Terminal, Variable
from src.solver.utils import flatten_list, assemble_parsed_content, remove_duplicates
from src.solver.visualize_util import visualize_path
from .abstract_algorithm import AbstractAlgorithm


class ElimilateVariablesRecursive(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], left_terms: List[Term],
                 right_terms: List[Term], parameters: Dict):
        super().__init__(terminals, variables, left_terms, right_terms)
        self.assignment = Assignment()
        self.parameters = parameters

    def run(self):
        result = self.explore_paths(self.left_terms, self.right_terms, self.variables)

        result_dict = {"result": result, "assignment": self.assignment, "left_terms": self.left_term_queue,
                       "right_terms": self.right_term_queue,
                       "variables": self.variables, "terminals": self.terminals}
        return result_dict

    def explore_paths(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term], variables: List[Variable]):
        branch_list=[]
        left_term = left_terms_queue[0]
        right_term = right_terms_queue[0]

        # terminate conditions
        # both side only have terminals
        if len(variables) == 0:
            result = "SAT" if self.check_equation(left_terms_queue, right_terms_queue) == True else "UNSAT"
            return result, variables
        # both side only have variables
        left_contains_no_terminal = not any(isinstance(term.value, Terminal) for term in left_terms_queue)
        right_contains_no_terminal = not any(isinstance(term.value, Terminal) for term in right_terms_queue)
        if left_contains_no_terminal and right_contains_no_terminal:
            return "SAT", variables

        # both side contains variables and terminals
        # both side empty
        if len(left_terms_queue) == 0 and len(right_terms_queue) == 0:
            return "SAT", variables
        # left side empty
        if len(left_terms_queue) == 0 and len(right_terms_queue) != 0:
            return "UNSAT", variables  # since one side has terminals
        # right side empty
        if len(left_terms_queue) != 0 and len(right_terms_queue) == 0:
            return "UNSAT", variables  # since one side has terminals

        # branch closed

        # split
        if left_term.value == right_term.value: # both side are the same
            left_terms_queue.popleft()
            right_terms_queue.popleft()
            updated_variables=self.update_variables(left_terms_queue, right_terms_queue)
            return self.explore_paths(left_terms_queue, right_terms_queue, updated_variables)
        else: # both side are different
            if type(left_term.value) == Variable and type(right_term.value) == Variable: # both side are differernt variables
                pass
            elif type(left_term.value) == Variable and type(right_term.value) == Terminal: # left side is variable, right side is terminal
                pass
            elif type(left_term.value) == Terminal and type(right_term.value) == Variable: # left side is terminal, right side is variable
                pass
            elif type(left_term.value) == Terminal and type(right_term.value) == Terminal: # both side are different terminals
                return "UNSAT", variables



    def update_variables(self,left_term_queue:Deque[Term],right_term_queue:Deque[Term])->List[Variable]:
        new_variables = []
        flattened_left_terms_list = flatten_list(left_term_queue)
        flattened_right_terms_list = flatten_list(right_term_queue)
        for t in flattened_left_terms_list+flattened_right_terms_list:
            if type(t.value) == Variable:
                new_variables.append(t.value)

        return remove_duplicates(new_variables)