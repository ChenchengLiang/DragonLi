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


class SplitEquations(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation],
                 parameters: Dict):
        super().__init__(terminals, variables, equation_list)
        self.assignment = Assignment()
        self.parameters = parameters
        self.nodes = []
        self.edges = []
        self.fresh_variable_counter = 0

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

        satisfiability, new_formula = self.control_propagation_and_split(original_formula)

        return {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                "variables": self.variables, "terminals": self.terminals}

    def control_propagation_and_split(self, original_formula: Formula) -> Tuple[str, Formula]:
        current_formula = original_formula
        while True:
            satisfiability, current_formula = self.propagate_facts(current_formula, current_formula.unknown_number)
            if satisfiability != UNKNOWN:
                return satisfiability, current_formula

            current_formula = self.split_equations(current_formula)

    def split_equations(self, original_formula: Formula) -> Formula:
        # choose a random equation to split
        unknown_eq_index = random.randint(0, len(original_formula.unknown_equations) - 1)
        unknown_eq: Equation = original_formula.unknown_equations.pop(unknown_eq_index)

        print(f"- explore path for {unknown_eq.eq_str} -")
        # todo split the chosen equation
        (satisfiability, processed_formula) = self.explore_path(eq=unknown_eq, current_formula=original_formula,
                                                                current_depth=0)
        return processed_formula

    def explore_path(self, eq: Equation, current_formula: Formula, current_depth: int) -> Tuple[str, Formula]:
        # todo add more terminate conditions for differernt backtrack strategies
        print(f"explore path for {eq.eq_str} at depth {current_depth}")
        # if current_depth > INITIAL_MAX_DEEP_BOUND_2:
        #     return (UNKNOWN, current_formula)

        eq_res = eq.check_satisfiability()

        if eq_res == SAT:
            # update the formula
            current_formula.sat_equations.append(eq)
            is_fact, fact_assignment = eq.is_fact()
            if is_fact:
                current_formula.facts.append((eq, fact_assignment))

            satisfiability, current_formula = self.propagate_facts(current_formula, current_formula.unknown_number)
            return (satisfiability, current_formula)
        if eq_res == UNSAT:
            return (UNSAT, current_formula)
        else:  # eq_res == UNKNOWN
            ################################ Split equation ################################
            eq = self.simplify_equation(eq)  # pop the same prefix

            # left_term != right_term
            left_term = eq.left_terms[0]
            right_term = eq.right_terms[0]

            # both side are different terminals
            if type(left_term.value) == Terminal and type(right_term.value) == Terminal:
                return (UNSAT, current_formula)
            # left side is variable, right side is terminal
            elif type(left_term.value) == Variable and type(right_term.value) == Terminal:
                return self.branch_for_left_side_variable_right_side_terminal(eq, current_formula, current_depth)
            # left side is terminal, right side is variable
            elif type(left_term.value) == Terminal and type(right_term.value) == Variable:
                return self.branch_for_left_side_terminal_right_side_variable(Equation(eq.right_terms, eq.left_terms),
                                                                              current_formula, current_depth)
            # both side are differernt variables
            elif type(left_term.value) == Variable and type(right_term.value) == Variable:
                return self.branch_for_both_side_different_variables(eq, current_formula, current_depth)

    def branch_for_left_side_variable_right_side_terminal(self, eq: Equation, current_formula: Formula,
                                                          current_depth) -> Tuple[
        str, Formula]:
        branch_method_list = [self._one_variable_one_terminal_branch_1,
                              self._one_variable_one_terminal_branch_2]
        return self._fixed_branch(branch_method_list, eq, current_formula, current_depth)

    def _one_variable_one_terminal_branch_1(self, eq: Equation, current_formula: Formula) -> Tuple[Equation, Formula]:
        '''
        Equation: V1 [Terms] = a [Terms]
        Assume V1 = ""
        Delete V1
        Obtain [Terms] [V1/""] = a [Terms] [V1/""]
        '''
        left_term: Term = eq.left_terms.pop(0)
        right_term: Term = eq.right_terms.pop(0)

        # define old and new term
        old_term: Term = left_term
        new_term: List[Term] = []

        # update equation
        new_left_term_list = self._update_term_list(old_term, new_term, eq.left_terms)
        new_right_term_list = [right_term] + self._update_term_list(old_term, new_term, eq.right_terms)
        new_eq = Equation(new_left_term_list, new_right_term_list)

        # update formula
        new_formula:Formula = self._update_formula(current_formula, old_term, new_term)

        return new_eq, new_formula


    def _one_variable_one_terminal_branch_2(self, eq: Equation, current_formula: Formula) -> Tuple[Equation, Formula]:
        '''
        Equation: V1 [Terms] = a [Terms]
        Assume V1 = aV1'
        Replace V1 with aV1'
        Obtain V1' [Terms] [V1/aV1'] = [Terms] [V1/aV1']
        '''
        left_term: Term = eq.left_terms.pop(0)
        right_term: Term = eq.right_terms.pop(0)

        # create fresh variable
        fresh_variable_term: Term = self._create_fresh_variables()
        # define old and new term
        old_term: Term = left_term
        new_term: List[Term] = [right_term, fresh_variable_term]

        # update equation
        new_left_term_list = [fresh_variable_term] + self._update_term_list(old_term, new_term, eq.left_terms)
        new_right_term_list = self._update_term_list(old_term, new_term, eq.right_terms)
        new_eq = Equation(new_left_term_list, new_right_term_list)

        # update formula
        new_formula:Formula = self._update_formula(current_formula, old_term, new_term)

        return new_eq, new_formula

    def _update_formula(self,f:Formula,old_term:Term,new_term:List[Term])->Formula:
        new_eq_list = []
        for eq_in_formula in f.formula:
            new_left = self._update_term_list(old_term, new_term, eq_in_formula.left_terms)
            new_right = self._update_term_list(old_term, new_term, eq_in_formula.right_terms)
            new_eq_list.append(Equation(new_left, new_right))
        return Formula(new_eq_list)

    def branch_for_both_side_different_variables(self, eq: Equation, current_formula: Formula, current_depth) -> Tuple[
        str, Formula]:
        branch_method_list = [self._two_variables_branch_1,
                              self._two_variables_branch_2,
                              self._two_variables_branch_3]
        return self._fixed_branch(branch_method_list, eq, current_formula, current_depth)

    def _fixed_branch(self, branch_method_list, eq: Equation, current_formula: Formula, current_depth) -> Tuple[
        str, Formula]:
        unknow_flag = False
        for branch_method in branch_method_list:
            (branched_eq, branched_formula) = branch_method(eq, current_formula)
            satisfiability, result_formula = self.explore_path(branched_eq, branched_formula, current_depth + 1)
            if satisfiability == SAT:
                return (satisfiability, result_formula)
            elif satisfiability == UNKNOWN:
                unknow_flag = True

        if unknow_flag == True:
            return (UNKNOWN, current_formula)
        else:
            return (UNSAT, current_formula)

    def _create_fresh_variables(self) -> Term:
        fresh_variable_term = Term(Variable(f"V{self.fresh_variable_counter}"))  # V1, V2, V3, ...
        self.fresh_variable_counter += 1
        return fresh_variable_term

    def _update_term_list(self, old_term: Term, new_term: List[Term], term_list: List[Term]) -> List[Term]:
        new_term_list = []
        for t in term_list:
            if t == old_term:
                for new_t in new_term:
                    new_term_list.append(new_t)
            else:
                new_term_list.append(t)
        return new_term_list

    def _two_variables_branch_1(self, eq: Equation, current_formula: Formula) -> Tuple[Equation, Formula]:
        '''
        Equation: V1 [Terms] = V2 [Terms]
        Assume |V1| > |V2|
        Replace V1 with V2V1'
        Obtain V1' [Terms] [V1/V2V1'] = [Terms] [V1/V2V1']
        '''
        left_term: Term = eq.left_terms.pop(0)
        right_term: Term = eq.right_terms.pop(0)

        # create fresh variable
        fresh_variable_term: Term = self._create_fresh_variables()
        # define old and new term
        new_term: List[Term] = [right_term, fresh_variable_term]
        old_term: Term = left_term

        # update equation
        new_left_term_list = [fresh_variable_term] + self._update_term_list(old_term, new_term, eq.left_terms)
        new_right_term_list = self._update_term_list(old_term, new_term, eq.right_terms)
        new_eq = Equation(new_left_term_list, new_right_term_list)

        # update formula
        new_formula:Formula = self._update_formula(current_formula, old_term, new_term)

        return new_eq, new_formula

    def _two_variables_branch_2(self, eq: Equation, current_formula: Formula) -> Tuple[Equation, Formula]:
        '''
        Equation: V1 [Terms] = V2 [Terms]
        Assume |V1| < |V2|
        Replace V2 with V1V2'
        Obtain [Terms] [V2/V1V2'] = V2' [Terms] [V2/V1V2']
        '''
        return self._two_variables_branch_1(Equation(eq.right_terms, eq.left_terms), current_formula)

    def _two_variables_branch_3(self, eq: Equation, current_formula: Formula) -> Tuple[Equation, Formula]:
        '''
        Equation: V1 [Terms] = V2 [Terms]
        Assume |V1| = |V2|
        Replace V1 with V2
        Obtain [Terms] [V1/V2] = [Terms] [V1/V2]
        '''
        left_term: Term = eq.left_terms.pop(0)
        right_term: Term = eq.right_terms.pop(0)

        # define old and new term
        old_term: Term = left_term
        new_term: List[Term] = [right_term]

        # update equation
        new_eq = Equation(self._update_term_list(old_term, new_term, eq.left_terms),
                          self._update_term_list(old_term, new_term, eq.right_terms))
        # update formula
        new_formula:Formula = self._update_formula(current_formula, old_term, new_term)

        return new_eq, new_formula

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
            for f in original_formula.facts:
                print(f)
                (e, assignment_list) = f
                for (v, t_list) in assignment_list:
                    temp_assigment.set_assignment(v, t_list)
            #
            # for f, assignment_list in original_formula.facts:
            #     for (v, t_list) in assignment_list:
            #         temp_assigment.set_assignment(v, t_list)
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
