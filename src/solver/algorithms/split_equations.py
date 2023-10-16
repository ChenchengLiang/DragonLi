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


    def check_satisfiability_for_all_eqs(self, equation_list: List[Equation]):
        '''
        Check satisfiability for all equations in equation_list,
        if there is a UNSAT equation, then return UNSAT,
        if all equations are SAT, then return SAT,
        otherwise return UNKNOWN
        '''
        satisfiability_list=[]
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


        satisfiability=self.check_satisfiability_for_all_eqs(self.equation_list)
        if satisfiability!=UNKNOWN:
            return {"result": SAT, "assignment": self.assignment, "equation_list": self.equation_list,
                       "variables": self.variables, "terminals": self.terminals}


        satisfiability_list=self.propagate_facts(self.equation_list)

        #todo split equations



        satisfiability="SAT"
        result_dict = {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                       "variables": self.variables, "terminals": self.terminals}
        return result_dict

    def propagate_facts(self, equation_list: List[Equation]):
        '''
        Propagate facts in equation_list until no more facts can be propagated
        '''
        satisfiability_list=[]
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

        updated_unknown_eq_list=[]
        if self.assignment.is_empty():
            pass
        else:
            for unknown_eq in unknown_eq_list:
                new_left_terms=[]
                for t in unknown_eq.left_terms:
                    if t.value in self.assignment.assigned_variables:
                        for terminal in self.assignment.get_assignment(t.value):
                            new_left_terms.append(Term(terminal))
                    else:
                        new_left_terms.append(t)
                new_right_terms=[]
                for t in unknown_eq.right_terms:
                    if t.value in self.assignment.assigned_variables:
                        for terminal in self.assignment.get_assignment(t.value):
                            new_right_terms.append(Term(terminal))
                    else:
                        new_right_terms.append(t)

                new_eq=Equation(new_left_terms,new_right_terms)

                if new_eq.check_satisfiability() == SAT:
                    return SAT
                elif new_eq.check_satisfiability() == UNSAT:
                    return UNSAT
                else:
                    updated_unknown_eq_list.append(new_eq)


        return satisfiability_list












    def visualize(self, file_path: str):
        pass