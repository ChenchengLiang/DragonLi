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
        pass