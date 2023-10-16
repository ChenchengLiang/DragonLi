import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED, recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR, UNSAT, SAT, INTERNAL_TIMEOUT, UNKNOWN
from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, EMPTY_TERMINAL, Formula
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

    def check_satisfiability_for_all_eqs(self, equation_list: List[Equation]) -> str:
        '''
        Check satisfiability for all equations in equation_list,
        if there is a UNSAT equation, then return UNSAT,
        if all equations are SAT, then return SAT,
        otherwise return UNKNOWN
        '''
        satisfiability_list = []
        for eq in equation_list:
            satisfiability = eq.check_satisfiability()
            # if there is a UNSAT equation, then return UNSAT
            if satisfiability == UNSAT:
                return UNSAT
            satisfiability_list.append(satisfiability)

        # if all elements are SAT, then return SAT
        if all([s == SAT for s in satisfiability_list]):
            return SAT
        return UNKNOWN

    def run(self):
        original_formula = Formula(self.equation_list)
        satisfiability, equation_list = self.propagate_facts(original_formula, unknown_number=original_formula.unknown_number)
        return {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                "variables": self.variables, "terminals": self.terminals}

        # if satisfiability != UNKNOWN:
        #     return {"result": SAT, "assignment": self.assignment, "equation_list": self.equation_list,
        #             "variables": self.variables, "terminals": self.terminals}

        # todo split equations

    def propagate_facts(self, original_formula:Formula, unknown_number):
        '''
        Propagate facts in equation_list until no more facts can be propagated
        '''

        print("propagate",str(original_formula.fact_number), "facts to ", str(unknown_number), "unknown equations")
        # transform facts to assignment
        temp_assigment = Assignment()
        if original_formula.fact_number != 0:
            for f, assignment_list in original_formula.facts:
                for (v, t_list) in assignment_list:
                    temp_assigment.set_assignment(v, t_list)

        else:
            pass

        # propagate facts to unknown equations
        propagated_eq_list = []
        if temp_assigment.is_empty():
            propagated_eq_list = original_formula.formula
        else:
            for eq in original_formula.formula:
                if eq.check_satisfiability() == UNKNOWN:
                    new_left_terms = []
                    for t in eq.left_terms:
                        if t.value in temp_assigment.assigned_variables:
                            for terminal in temp_assigment.get_assignment(t.value):
                                new_left_terms.append(Term(terminal))
                        else:
                            new_left_terms.append(t)
                    new_right_terms = []
                    for t in eq.right_terms:
                        if t.value in temp_assigment.assigned_variables:
                            for terminal in temp_assigment.get_assignment(t.value):
                                new_right_terms.append(Term(terminal))
                        else:
                            new_right_terms.append(t)

                    new_eq = Equation(new_left_terms, new_right_terms)
                    propagated_eq_list.append(new_eq)
                else:
                    propagated_eq_list.append(eq)

        # check propageted equations
        new_formula = Formula(propagated_eq_list)
        satisfiability_new_eq_list = new_formula.satisfiability
        if satisfiability_new_eq_list == UNKNOWN:
            if new_formula.unknown_number < unknown_number:  # if new unknown equations are found, continue to propagate
                return self.propagate_facts(new_formula, unknown_number=new_formula.unknown_number)
            else:  # if no new unknown equations are found, return unknown
                return satisfiability_new_eq_list, propagated_eq_list
        else:  # find solution
            return satisfiability_new_eq_list, propagated_eq_list

    def visualize(self, file_path: str):
        pass
