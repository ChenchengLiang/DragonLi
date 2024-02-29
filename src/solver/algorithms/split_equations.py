import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED, recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR, UNSAT, SAT, INTERNAL_TIMEOUT, UNKNOWN, INITIAL_MAX_DEEP_BOUND_2
from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, EMPTY_TERMINAL, Formula
from src.solver.utils import assemble_parsed_content
from ..independent_utils import remove_duplicates, flatten_list
from src.solver.visualize_util import visualize_path, visualize_path_html
from .abstract_algorithm import AbstractAlgorithm
import sys
from src.solver.algorithms.split_equation_utils import _one_variable_one_terminal_branch_1, \
    _one_variable_one_terminal_branch_2, _two_variables_branch_1, _two_variables_branch_2, _two_variables_branch_3, \
    choose_an_unknown_eqiatons_random, choose_an_unknown_eqiatons_fixed, _update_formula_with_new_eq


class SplitEquations(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation],
                 parameters: Dict):
        super().__init__(terminals, variables, equation_list)
        self.assignment = Assignment()
        self.parameters = parameters
        self.nodes = []
        self.edges = []
        self.fresh_variable_counter = 0
        self.total_explore_paths_call = 0

        self.choose_unknown_eq_func_map = {"random": choose_an_unknown_eqiatons_random,
                                           "fixed": choose_an_unknown_eqiatons_fixed}
        self.choose_unknown_eq_func: Callable = self.choose_unknown_eq_func_map[
            self.parameters["choose_unknown_eq_method"]]
        self.branch_method_func_map = {"fixed": self._fixed_branch,
                                       "random": self._random_branch}
        self.branch_method_func: Callable = self.branch_method_func_map[self.parameters["branch_method"]]

        sys.setrecursionlimit(recursion_limit)
        print("recursion limit number", sys.getrecursionlimit())

    def run(self):
        original_formula = Formula(self.equation_list)

        satisfiability, new_formula = self.control_propagation_and_split(original_formula)

        return {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                "variables": self.variables, "terminals": self.terminals}

    def control_propagation_and_split(self, original_formula: Formula) -> Tuple[str, Formula]:
        current_formula = original_formula

        satisfiability, current_formula = self.propagate_facts(current_formula, current_formula.unknown_number)
        if satisfiability != UNKNOWN:
            return satisfiability, current_formula

        satisfiability,current_formula = self.split_equations(current_formula)


        return satisfiability, current_formula
        #
        # while True:
        #     satisfiability, current_formula = self.propagate_facts(current_formula, current_formula.unknown_number)
        #     if satisfiability != UNKNOWN:
        #         return satisfiability, current_formula
        #
        #     current_formula = self.split_equations(current_formula)

    def split_equations(self, original_formula: Formula) -> Tuple[str,Formula]:
        # choose a equation to split
        unknown_eq, current_formula = self.choose_unknown_eq_func(original_formula)

        print(f"----------------- explore path for {unknown_eq.eq_str} -----------------")
        # split the chosen equation
        self.total_explore_paths_call = 0
        (satisfiability, processed_formula) = self.explore_path(eq=unknown_eq, current_formula=current_formula,
                                                                current_depth=0)
        return satisfiability,processed_formula


    def explore_path(self, eq: Equation, current_formula: Formula, current_depth: int) -> Tuple[str, Formula]:
        self.total_explore_paths_call += 1
        #print(f"current_depth: {current_depth} total explored path: {self.total_explore_paths_call}, {eq.eq_str}")

        # todo add more terminate conditions for differernt backtrack strategies
        # if current_depth > INITIAL_MAX_DEEP_BOUND_2:
        #     return (UNKNOWN, current_formula)

        eq = self.simplify_equation(eq)  # pop the same prefix
        eq_res = eq.check_satisfiability()
        #print(f"** {'-' * current_depth} explore path for {eq.eq_str} at depth {current_depth}")
        #print(f"** {'-'*current_depth} equation {eq.eq_str} is {eq_res}")



        if eq_res == SAT:
            # todo check rest eqs
            #reconstructed formula don;t know the new eq is SAT
            new_formula_satisfiability,new_formula=_update_formula_with_new_eq(current_formula,eq,SAT)

            if new_formula_satisfiability == SAT:
                return (SAT, new_formula)
            elif new_formula_satisfiability == UNSAT:
                return (UNSAT, new_formula)
            elif new_formula_satisfiability == UNKNOWN:
                return self.control_propagation_and_split(new_formula)


        elif eq_res == UNSAT:
            new_formula_satisfiability, new_formula = _update_formula_with_new_eq(current_formula, eq,UNSAT)
            return (UNSAT, new_formula)
        elif eq_res == UNKNOWN:
            ################################ Split equation ################################


            # left_term != right_term
            left_term = eq.left_terms[0]
            right_term = eq.right_terms[0]

            # both side are different terminals #this has be done in eq_res = eq.check_satisfiability()
            # if type(left_term.value) == Terminal and type(right_term.value) == Terminal:
            #     new_formula_satisfiability, new_formula = _update_formula_with_new_eq(current_formula, eq,UNSAT)
            #     return (UNSAT, new_formula)
            # left side is variable, right side is terminal
            if type(left_term.value) == Variable and type(right_term.value) == Terminal:
                return self.branch_for_left_side_variable_right_side_terminal(eq, current_formula, current_depth)
            # left side is terminal, right side is variable
            elif type(left_term.value) == Terminal and type(right_term.value) == Variable:
                return self.branch_for_left_side_variable_right_side_terminal(Equation(eq.right_terms, eq.left_terms),
                                                                              current_formula, current_depth)
            # both side are differernt variables
            elif type(left_term.value) == Variable and type(right_term.value) == Variable:
                return self.branch_for_both_side_different_variables(eq, current_formula, current_depth)

    def branch_for_left_side_variable_right_side_terminal(self, eq: Equation, current_formula: Formula,
                                                          current_depth) -> Tuple[
        str, Formula]:
        branch_method_list = [_one_variable_one_terminal_branch_1,
                              _one_variable_one_terminal_branch_2]
        return self.branch_method_func(branch_method_list, eq, current_formula, current_depth)

    def branch_for_both_side_different_variables(self, eq: Equation, current_formula: Formula, current_depth) -> Tuple[
        str, Formula]:
        branch_method_list = [_two_variables_branch_1,
                              _two_variables_branch_2,
                              _two_variables_branch_3]
        return self.branch_method_func(branch_method_list, eq, current_formula, current_depth)

    def _random_branch(self, branch_method_list, eq: Equation, current_formula: Formula, current_depth) -> Tuple[
        str, Formula]:
        random.shuffle(branch_method_list)
        return self._fixed_branch(branch_method_list, eq, current_formula, current_depth)

    def _fixed_branch(self, branch_method_list, eq: Equation, current_formula: Formula, current_depth) -> Tuple[
        str, Formula]:
        unknow_flag = False
        for branch_method in branch_method_list:
            (branched_eq, branched_formula, branch_fresh_variable_counter) = branch_method(eq, current_formula,
                                                                                           self.fresh_variable_counter)
            self.fresh_variable_counter = branch_fresh_variable_counter
            satisfiability, result_formula = self.explore_path(branched_eq, branched_formula, current_depth + 1)
            if satisfiability == SAT:
                return (satisfiability, result_formula)
            elif satisfiability == UNKNOWN:
                unknow_flag = True

        if unknow_flag == True:
            return (UNKNOWN, current_formula)
        else:
            return (UNSAT, current_formula)

    def simplify_equation(self, eq: Equation) -> Equation:
        # pop the same prefix
        for index in range(min(len(eq.left_terms), len(eq.right_terms))):
            if eq.left_terms[0] == eq.right_terms[0]:
                eq.left_terms.pop(0)
                eq.right_terms.pop(0)
        return eq

    def propagate_facts(self, original_formula: Formula, unknown_number) -> Tuple[str, Formula]:
        '''
        Propagate facts in equation_list until no more facts can be propagated
        '''

        # todo check conflict between facts

        print("propagate", str(original_formula.fact_number), "facts to ", str(unknown_number), "unknown equations")
        # transform facts to assignment
        temp_assigment = Assignment()
        if original_formula.fact_number != 0:
            for (e, assignment_list) in original_formula.facts:
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
                return satisfiability_new_eq_list, new_formula
        else:  # find solution
            return satisfiability_new_eq_list, new_formula

    def visualize(self, file_path: str):
        pass
