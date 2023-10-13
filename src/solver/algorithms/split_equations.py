import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED, recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR
from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, EMPTY_TERMINAL
from src.solver.utils import flatten_list, assemble_parsed_content, remove_duplicates
from src.solver.visualize_util import visualize_path, visualize_path_html
from .abstract_algorithm import AbstractAlgorithm
import sys



class SplitEquations(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation],
                 parameters: Dict):
        super().__init__(terminals, variables, equation_list)
        self.assignment = Assignment()
        self.parameters = parameters
        self.nodes = []
        self.edges = []

        sys.setrecursionlimit(recursion_limit)
        print("recursion limit number", sys.getrecursionlimit())

    def run(self):

        self.propagate_facts(self.equation_list,self.assignment)

        satisfiability="SAT"
        result_dict = {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                       "variables": self.variables, "terminals": self.terminals}
        return result_dict

    def propagate_facts(self, equation_list: List[Equation], assignment: Assignment):
        facts = [e for e in equation_list if e.is_fact]
        not_facts = [e for e in equation_list if not e.is_fact]
        # transform facts to assignments
        for f in facts:
            assignment.set_assignment(f.variable_list[0],f.terminal_list)

        for f in facts:
            print("f:",f.eq_str)
        for not_f in not_facts:
            print("not_f:",not_f.eq_str)



    def visualize(self, file_path: str):
        pass