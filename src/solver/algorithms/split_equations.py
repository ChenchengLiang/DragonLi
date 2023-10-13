import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED, recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR, UNSAT, SAT, INTERNAL_TIMEOUT, UNKNOWN
from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, EMPTY_TERMINAL
from src.solver.utils import assemble_parsed_content
from ..independent_utils import remove_duplicates, flatten_list
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
        #check satisfiability for all equations
        satisfiability_list=[]
        for eq in self.equation_list:
            satisfiability = eq.check_satisfiability()
            # if there is a UNSAT equation, then return UNSAT
            if satisfiability == UNSAT:
                return {"result": UNSAT, "assignment": self.assignment, "equation_list": self.equation_list,
                       "variables": self.variables, "terminals": self.terminals}
            satisfiability_list.append(satisfiability)

        #if all elements are SAT, then return SAT
        if all([s == SAT for s in satisfiability_list]):
            return {"result": SAT, "assignment": self.assignment, "equation_list": self.equation_list,
                       "variables": self.variables, "terminals": self.terminals}


        self.propagate_facts(self.equation_list)

        #todo split equations



        satisfiability="SAT"
        result_dict = {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                       "variables": self.variables, "terminals": self.terminals}
        return result_dict

    def propagate_facts(self, equation_list: List[Equation]):
        '''
        Propagate facts in equation_list until no more facts can be propagated
        '''
        facts = []
        not_facts = []
        unknown_eq_list=[]
        for eq in equation_list:
            is_fact, assignment_list = eq.is_fact()
            if is_fact:
                facts.append(eq)
                # transform facts to assignments
                for (v,t_list) in assignment_list:
                    self.assignment.set_assignment(v, t_list)
            else:
                not_facts.append(eq)
            if eq.check_satisfiability() == "UNKNOWN":
                unknown_eq_list.append(eq)



        for f in facts:
            print("fact:",f.eq_str,f.check_satisfiability())
        for not_f in not_facts:
            print("not fact:",not_f.eq_str,not_f.check_satisfiability())
        for unknown_eq in unknown_eq_list:
            print("unknown_eq:",unknown_eq.eq_str,unknown_eq.check_satisfiability())

        print("equation_list_len:",len(equation_list))
        print("facts_len:",len(facts))
        print("not_facts_len:",len(not_facts))
        print("unknown_eq_list_len:",len(unknown_eq_list))

        #todo: propagate facts to unknown equations











    def visualize(self, file_path: str):
        pass