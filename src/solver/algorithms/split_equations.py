import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED, recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR, UNSAT, SAT, INTERNAL_TIMEOUT, UNKNOWN, RESTART_INITIAL_MAX_DEEP, \
    RESTART_MAX_DEEP_STEP, compress_image
from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, EMPTY_TERMINAL, Formula
from src.solver.utils import assemble_parsed_content
from ..independent_utils import remove_duplicates, flatten_list, color_print,log_control
from src.solver.visualize_util import visualize_path_html, visualize_path_png
from .abstract_algorithm import AbstractAlgorithm
import sys
from src.solver.algorithms.split_equation_utils import _category_formula_by_rules,apply_rules,simplify_and_check_formula


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
                                         "random": self._order_equations_random,
                                         "category": self._order_equations_category,
                                         "category_gnn": self._order_equations_category_gnn,
                                         "gnn": self._order_equations_gnn}
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

        self.log_enabled = True

    @log_control
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

        # early termination condition
        res = self.check_termination_condition_func(current_depth)
        if res != None:
            return (res, original_formula)


        satisfiability, current_formula = simplify_and_check_formula(original_formula)

        if satisfiability != UNKNOWN:
            return satisfiability, current_formula
        else:
            current_formula = self.order_equations_func(current_formula)
            current_eq, separated_formula = self.get_first_eq(current_formula)


            current_node = self.record_node_and_edges(current_eq, separated_formula, previous_node, edge_label)

            children,fresh_variable_counter= apply_rules(current_eq, separated_formula,self.fresh_variable_counter)
            self.fresh_variable_counter=fresh_variable_counter
            children: List[Tuple[Equation, Formula, str]] = self.order_branches_func(children)

            for c_index, child in enumerate(children):
                (c_eq, c_formula, edge_label) = child
                satisfiability, res_formula = self.split_eq(c_formula, current_depth + 1, current_node,
                                                            edge_label)
                if satisfiability == SAT:
                    return (SAT, res_formula)
                elif satisfiability == UNKNOWN:
                    return (UNKNOWN, res_formula)

            return (UNSAT, current_formula)


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

    def get_first_eq(self, f: Formula) -> Tuple[Equation, Formula]:
        return f.eq_list[0], Formula(f.eq_list[1:])


    def _order_equations_category_gnn(self, f: Formula) -> Formula:
        categoried_eq_list: List[Tuple[Equation, int]] = _category_formula_by_rules(f)

        # Check if the equation categories are only 5 and 6
        only_5_and_6:bool = all(n in [5, 6] for _, n in categoried_eq_list)

        if only_5_and_6==True:
            sorted_eq_list = self._order_equations_gnn(f).eq_list
        else:
            sorted_eq_list = sorted(categoried_eq_list, key=lambda x: x[1])

        return Formula([eq for eq, _ in sorted_eq_list])

    def _order_equations_gnn(self, f: Formula) -> Formula:
        # todo implement gnn
        return f


    def _order_equations_category(self, f: Formula) -> Formula:
        categoried_eq_list:List[Tuple[Equation, int]]=_category_formula_by_rules(f)
        sorted_eq_list = sorted(categoried_eq_list, key=lambda x: x[1])

        return Formula([eq for eq, _ in sorted_eq_list])

    def _order_equations_fixed(self, f: Formula) -> Formula:
        return f

    def _order_equations_random(self, f: Formula) -> Formula:
        random.shuffle(f.eq_list)
        return f



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
