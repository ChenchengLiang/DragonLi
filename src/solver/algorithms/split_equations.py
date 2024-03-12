import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED, recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR, UNSAT, SAT, INTERNAL_TIMEOUT, UNKNOWN, RESTART_INITIAL_MAX_DEEP, \
    RESTART_MAX_DEEP_STEP, compress_image
from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, EMPTY_TERMINAL, Formula
from src.solver.utils import assemble_parsed_content
from ..independent_utils import remove_duplicates, flatten_list, color_print
from src.solver.visualize_util import visualize_path, visualize_path_html, visualize_path_png
from .abstract_algorithm import AbstractAlgorithm
import sys
from src.solver.algorithms.split_equation_utils import _left_variable_right_terminal_branch_1, \
    _left_variable_right_terminal_branch_2, _two_variables_branch_1, _two_variables_branch_2, _two_variables_branch_3, _update_formula


class SplitEquations(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation],
                 parameters: Dict):
        super().__init__(terminals, variables, equation_list)
        self.assignment = Assignment()
        self.parameters = parameters
        self.nodes = []
        self.edges = []
        self.fresh_variable_counter = 0
        self.total_split_eq_call = 0
        self.restart_max_deep = RESTART_INITIAL_MAX_DEEP


        self.order_equations_func_map = {"fixed": self._order_equations_fixed,
                                         "random": self._order_equations_random}
        self.order_equations_func: Callable = self.order_equations_func_map[self.parameters["order_equations_method"]]

        self.branch_method_func_map = {"fixed": self._order_branches_fixed,
                                       "random": self._order_branches_random,
                                       "gnn": self._order_branches_gnn}
        self.order_branches_func: Callable = self.branch_method_func_map[self.parameters["branch_method"]]

        self.check_termination_condition_map = {"termination_condition_0": self.early_termination_condition_0,
                                                "termination_condition_1": self.early_termination_condition_1}
        self.check_termination_condition_func: Callable = self.check_termination_condition_map[
            self.parameters["termination_condition"]]

        sys.setrecursionlimit(recursion_limit)
        print("recursion limit number", sys.getrecursionlimit())

    def run(self):
        original_formula = Formula(self.equation_list)

        if self.parameters["termination_condition"] == "termination_condition_1":
            while True:
                initial_node: Tuple[int, Dict] = (
                    0, {"label": "start", "status": None, "output_to_file": False, "shape": "ellipse",
                        "back_track_count": 0})
                satisfiability, new_formula = self.split_eq(original_formula, current_depth=0,previous_node=initial_node,edge_label="start")
                if satisfiability != UNKNOWN:
                    break

                self.restart_max_deep += RESTART_MAX_DEEP_STEP
        else:
            initial_node: Tuple[int, Dict] = (
                0, {"label": "start", "status": None, "output_to_file": False, "shape": "ellipse",
                    "back_track_count": 0})
            self.nodes.append(initial_node)

            satisfiability, new_formula = self.split_eq(original_formula, current_depth=0, previous_node=initial_node,
                                                        edge_label="start")

        return {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                "variables": self.variables, "terminals": self.terminals}

    def split_eq(self, original_formula: Formula, current_depth: int, previous_node: Tuple[int, Dict],
                 edge_label: str) -> Tuple[str, Formula]:
        self.total_split_eq_call += 1

        print(f"----- total_split_eq_call:{self.total_split_eq_call}, current_depth:{current_depth} -----")
        #print(original_formula.eq_list_str)

        # early termination condition
        res = self.check_termination_condition_func(current_depth)
        if res != None:
            return (res, original_formula)


        satisfiability, current_formula = self.simplify_and_check_formula(original_formula)

        if satisfiability != UNKNOWN:
            return satisfiability, current_formula
        else:
            current_formula = self.order_equations_func(current_formula)
            current_eq, separated_formula = self.get_first_eq(current_formula)
            #print(f"current_eq:{current_eq.eq_str}, separated_formula:{separated_formula.eq_list_str}")

            current_node = self.record_node_and_edges(current_eq, separated_formula, previous_node, edge_label)

            children: List[Tuple[Equation, Formula, str]] = self.apply_rules(current_eq, separated_formula)
            children: List[Tuple[Equation, Formula, str]] = self.order_branches_func(children)

            unknown_flag = False
            for c_index, child in enumerate(children):
                (c_eq, c_formula, edge_label) = child
                satisfiability, res_formula = self.split_eq(c_formula, current_depth + 1, current_node,
                                                            edge_label)
                if satisfiability == SAT:
                    return (SAT, res_formula)
                elif satisfiability == UNKNOWN:
                    unknown_flag = True
            if unknown_flag == True:
                return (UNKNOWN, current_formula)
            else:
                return (UNSAT, current_formula)

            return satisfiability, processed_formula

    def record_node_and_edges(self, eq: Equation, f: Formula, previous_node: Tuple[int, Dict], edge_label: str) -> \
            Tuple[int, Dict]:
        current_node_number = self.total_split_eq_call
        label = f"{eq.eq_str},{f.eq_list_str}"
        current_node = (
            current_node_number,
            {"label": label, "status": None, "output_to_file": False, "shape": "ellipse", "back_track_count": 0})
        self.nodes.append(current_node)
        self.edges.append((previous_node[0], current_node_number, {'label': edge_label}))
        return current_node

    def simplify_and_check_formula(self, f: Formula) -> Tuple[str, Formula]:
        #f.print_eq_list()
        f.simplify_eq_list()

        satisfiability = f.check_satisfiability_2()


        return satisfiability, f

    def get_first_eq(self, f: Formula) -> Tuple[Equation, Formula]:
        return f.eq_list[0], Formula(f.eq_list[1:])

    def _order_equations_fixed(self, f: Formula) -> Formula:
        return f

    def _order_equations_random(self, f: Formula) -> Formula:
        random.shuffle(f.eq_list)
        return f


    def apply_rules(self, eq: Equation, f: Formula) -> List[Tuple[Equation, Formula]]:
        # handle non-split rules

        # both sides are empty
        if len(eq.term_list) == 0:
            children: List[Tuple[Equation, Formula, str]] = [(eq, f, " \" = \" ")]
        # left side is empty
        elif len(eq.left_terms) == 0 and len(eq.right_terms) > 0:
            children: List[Tuple[Equation, Formula, str]] = self.left_side_empty(eq, f)
        # right side is empty
        elif len(eq.left_terms) > 0 and len(eq.right_terms) == 0:  # right side is empty
            children: List[Tuple[Equation, Formula, str]] = self.left_side_empty(Equation(eq.right_terms,eq.left_terms), f)
        # both sides are not empty
        else:
            first_left_term = eq.left_terms[0]
            first_right_term = eq.right_terms[0]
            # \epsilon=\epsilon \wedge \phi case
            if eq.left_terms == eq.right_terms:
                children: List[Tuple[Equation, Formula, str]] = [(eq, f, " \" = \" ")]

            # match prefix terminal
            elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value == first_right_term.value:
                eq.simplify()
                children: List[Tuple[Equation, Formula, str]] = [
                    (eq, Formula([eq] + f.eq_list), " a u= a v \wedge \phi")]

            # mismatch prefix terminal
            elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value != first_right_term.value:
                eq.given_satisfiability = UNSAT
                children: List[Tuple[Equation, Formula, str]] = [
                    (eq, Formula([eq] + f.eq_list), " a u = b v \wedge \phi")]
            # split rules
            else:
                left_term = eq.left_terms[0]
                right_term = eq.right_terms[0]
                # left side is variable, right side is terminal
                if type(left_term.value) == Variable and type(right_term.value) == Terminal:
                    rule_list: List[Callable] = [_left_variable_right_terminal_branch_1, _left_variable_right_terminal_branch_2]
                    children: List[Tuple[Equation, Formula, str]] = self.get_split_children(eq, f, rule_list)

                # left side is terminal, right side is variable
                elif type(left_term.value) == Terminal and type(right_term.value) == Variable:
                    rule_list: List[Callable] = [_left_variable_right_terminal_branch_1, _left_variable_right_terminal_branch_2]
                    children: List[Tuple[Equation, Formula, str]] = self.get_split_children(
                        Equation(eq.right_terms, eq.left_terms), f,
                        rule_list)

                # both side are differernt variables
                elif type(left_term.value) == Variable and type(right_term.value) == Variable:
                    rule_list: List[Callable] = [_two_variables_branch_1, _two_variables_branch_2, _two_variables_branch_3]
                    children: List[Tuple[Equation, Formula, str]] = self.get_split_children(eq, f, rule_list)

                else:
                    children: List[Tuple[Equation, Formula, str]] = []
                    color_print(f"error: {eq.eq_str}", "red")

        return children

    def left_side_empty(self, eq: Equation, f: Formula) -> List[Tuple[Equation, Formula, str]]:
        '''
        Assume another side is empty.
        there are three conditions for one side: (1). terminals + variables (2). only terminals (3). only variables
        '''
        # (1) + (2): if there are any Terminal in the not_empty_side, then it is UNSAT
        not_empty_side=eq.right_terms
        if any(isinstance(term.value, Terminal) for term in not_empty_side):
            eq.given_satisfiability = UNSAT
            children: List[Tuple[Equation, Formula, str]] = [
                (eq, Formula([eq] + f.eq_list), " a u = \epsilon \wedge \phi")]
        # (3): if there are only Variables in the not_empty_side
        else:
            for variable_term in not_empty_side:
                f = _update_formula(f, variable_term, [])
            children: List[Tuple[Equation, Formula, str]] = [
                (eq, f, " XYZ = \epsilon \wedge \phi")]

        return children



    def get_split_children(self, eq: Equation, f: Formula, rule_list: Callable) -> List[Tuple[Equation, Formula, str]]:
        children: List[Tuple[Equation, Formula, str]] = []
        for rule in rule_list:
            new_eq, new_formula, fresh_variable_counter, label_str = rule(eq, f,
                                                                          self.fresh_variable_counter)
            self.fresh_variable_counter = fresh_variable_counter
            reconstructed_formula = Formula([new_eq] + new_formula.eq_list)
            child: Tuple[Equation, Formula, str] = (new_eq, reconstructed_formula, label_str)
            children.append(child)
        return children

    def _order_branches_fixed(self, children: List[Tuple[Equation, Formula]]) -> List[Tuple[Equation, Formula]]:
        return children

    def _order_branches_random(self, children: List[Tuple[Equation, Formula]]) -> List[Tuple[Equation, Formula]]:
        random.shuffle(children)
        return children

    def _order_branches_gnn(self, children: List[Tuple[Equation, Formula]]) -> List[Tuple[Equation, Formula]]:
        # todo implement gnn
        return children

    def early_termination_condition_0(self, current_depth: int):
        return None

    def early_termination_condition_1(self, current_depth: int):
        if current_depth > self.restart_max_deep:
            return UNKNOWN

    def visualize(self, file_path: str, graph_func: Callable):
        visualize_path_html(self.nodes, self.edges, file_path)
        visualize_path_png(self.nodes, self.edges, file_path, compress=compress_image)
