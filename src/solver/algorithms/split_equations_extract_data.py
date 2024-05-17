import os.path
import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED, recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR, UNSAT, SAT, INTERNAL_TIMEOUT, UNKNOWN, RESTART_INITIAL_MAX_DEEP, \
    RESTART_MAX_DEEP_STEP, compress_image
from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, EMPTY_TERMINAL, Formula
from src.solver.utils import assemble_parsed_content
from ..independent_utils import remove_duplicates, flatten_list, color_print, log_control, strip_file_name_suffix, \
    dump_to_json_with_format
from src.solver.visualize_util import visualize_path_html, visualize_path_png
from .abstract_algorithm import AbstractAlgorithm
import sys
from src.solver.algorithms.split_equation_utils import _category_formula_by_rules, apply_rules, \
    simplify_and_check_formula, order_equations_fixed, order_equations_random, order_equations_category, \
    order_equations_category_random


class SplitEquationsExtractData(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation],
                 parameters: Dict):
        super().__init__(terminals, variables, equation_list)
        self.assignment = Assignment()
        self.parameters = parameters
        self.nodes = []
        self.edges = []
        self.fresh_variable_counter = 0
        self.total_split_eq_call = 0
        self.eq_node_number = 0
        self.total_node_number = 1
        self.restart_max_deep = RESTART_INITIAL_MAX_DEEP
        self.found_sat_path = 0
        self.found_unsat_path = 0
        self.found_path = 0
        self.total_output_branches = 0
        # control path number for extraction
        self.max_deep_for_extraction = 3
        self.max_found_sat_path_extraction = 1
        self.max_found_path_extraction = 20
        self.task = parameters["task"]
        self.file_name = strip_file_name_suffix(parameters["file_path"])
        self.train_data_count = 0

        self.order_equations_func_map = {"fixed": order_equations_fixed,
                                         "random": order_equations_random,
                                         "category": order_equations_category,
                                         "category_random": order_equations_category_random}
        self.order_equations_func: Callable = self.order_equations_func_map[self.parameters["order_equations_method"]]

        self.branch_method_func_map = {"fixed": self._order_branches_fixed,
                                       "random": self._order_branches_random}
        self.order_branches_func: Callable = self.branch_method_func_map[self.parameters["branch_method"]]

        self.check_termination_condition_map = {"termination_condition_0": self.early_termination_condition_0,
                                                # no limit
                                                "termination_condition_1": self.early_termination_condition_1,
                                                # restart
                                                "termination_condition_2": self.early_termination_condition_2,
                                                # max deepth
                                                "termination_condition_3": self.early_termination_condition_3,
                                                # found path
                                                "termination_condition_4": self.early_termination_condition_4}  # found sat path
        self.check_termination_condition_func: Callable = self.check_termination_condition_map[
            self.parameters["termination_condition"]]

        sys.setrecursionlimit(recursion_limit)
        print("recursion limit number", sys.getrecursionlimit())

        self.log_enabled = True
        self.png_edge_label = True

    @log_control
    def run(self):
        original_formula = Formula(list(self.equation_list))

        initial_node: Tuple[int, Dict] = (
            0, {"label": "start", "status": None, "output_to_file": False, "shape": "circle",
                "back_track_count": 0})
        self.nodes.append(initial_node)

        satisfiability, new_formula, child_node = self.split_eq(original_formula, current_depth=0,
                                                                previous_branch_node=initial_node,
                                                                edge_label="start")

        print(f"----- total_output_branches:{self.total_output_branches} -----")

        return {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                "variables": self.variables, "terminals": self.terminals}

    def split_eq(self, original_formula: Formula, current_depth: int, previous_branch_node: Tuple[int, Dict],
                 edge_label: str) -> Tuple[str, Formula, Tuple[int, Dict]]:
        self.total_split_eq_call += 1
        print(f"----- total_split_eq_call:{self.total_split_eq_call}, current_depth:{current_depth} -----")

        current_node = self.record_node_and_edges(original_formula, previous_branch_node, edge_label)

        ####################### early termination condition #######################
        res = self.check_termination_condition_func(current_depth)
        if res != None:  # None denote skip termination check
            current_node[1]["status"] = res
            current_node[1]["back_track_count"] = 1
            self.found_path += 1
            return (res, original_formula, current_node)  # terminate current branch

        ####################### search #######################
        satisfiability, current_formula = simplify_and_check_formula(original_formula)

        if satisfiability != UNKNOWN:
            current_node[1]["status"] = satisfiability
            current_node[1]["back_track_count"] = 1
            self.found_path += 1
            if satisfiability == SAT:
                self.found_sat_path += 1
            if satisfiability == UNSAT:
                self.found_unsat_path += 1

            return (satisfiability, current_formula, current_node)
        else:
            # systematic search training data by using "order_equations_method": "fixed"
            current_formula: Formula = self.order_equations_func(current_formula)

            split_back_track_count = 1
            branch_eq_satisfiability_list: List[Tuple[Equation, str]] = []
            for index, eq in enumerate(list(current_formula.eq_list)):
                current_eq, separated_formula = self.get_eq_by_index(Formula(list(current_formula.eq_list)), index)
                current_eq_node = self.record_eq_node_and_edges(current_eq, previous_node=current_node,
                                                                edge_label=f"eq:{index}: {current_eq.eq_str}")

                children, fresh_variable_counter = apply_rules(current_eq, separated_formula,
                                                               self.fresh_variable_counter)
                self.fresh_variable_counter = fresh_variable_counter
                children: List[Tuple[Equation, Formula, str]] = self.order_branches_func(children)

                eq_back_track_count = 0
                split_branch_satisfiability_list: List[Tuple[Equation, str, int]] = []

                for c_index, child in enumerate(children):
                    (c_eq, c_formula, edge_label) = child
                    satisfiability, res_formula, child_node = self.split_eq(c_formula, current_depth + 1,
                                                                            previous_branch_node=current_eq_node,
                                                                            edge_label=edge_label)
                    split_back_track_count += child_node[1]["back_track_count"]

                    eq_back_track_count += child_node[1]["back_track_count"]

                    split_branch_satisfiability_list.append(satisfiability)

                current_eq_node[1]["back_track_count"] = eq_back_track_count

                if any(eq_satisfiability == SAT for eq_satisfiability in split_branch_satisfiability_list):
                    current_eq_node[1]["status"] = SAT
                    branch_eq_satisfiability_list.append((current_eq, SAT, current_eq_node[1]["back_track_count"]))
                elif any(eq_satisfiability == UNKNOWN for eq_satisfiability in split_branch_satisfiability_list):
                    current_eq_node[1]["status"] = UNKNOWN
                    branch_eq_satisfiability_list.append((current_eq, UNKNOWN, current_eq_node[1]["back_track_count"]))
                else:
                    current_eq_node[1]["status"] = UNSAT
                    branch_eq_satisfiability_list.append((current_eq, UNSAT, current_eq_node[1]["back_track_count"]))

            current_node[1]["back_track_count"] = split_back_track_count
            if any(eq_satisfiability == SAT for _, eq_satisfiability, _ in branch_eq_satisfiability_list):
                current_node[1]["status"] = SAT
            elif any(eq_satisfiability == UNKNOWN for _, eq_satisfiability, _ in branch_eq_satisfiability_list):
                current_node[1]["status"] = UNKNOWN
            else:
                current_node[1]["status"] = UNSAT

            # output labeled eqs according to order_equations_method
            if len(branch_eq_satisfiability_list) > 1:
                if "category" in self.parameters["order_equations_method"]:  # category
                    categoried_eq_list: List[Tuple[Equation, int]] = _category_formula_by_rules(current_formula)
                    # Check if the equation categories are only 5 and 6
                    only_5_and_6: bool = all(n in [5, 6] for _, n in categoried_eq_list)
                    if only_5_and_6 == True:
                        current_node[1]["output_to_file"]=True
                        _,label_list=self.extract_dynamic_embedding_train_data(branch_eq_satisfiability_list, current_node[0])
                        # print("total eqs", len(current_formula.eq_list))
                        # for eq, label in zip(current_formula.eq_list, label_list):
                        #     print(eq.eq_str, label)
                else:  # fix or random
                    current_node[1]["output_to_file"] = True
                    _,label_list=self.extract_dynamic_embedding_train_data(branch_eq_satisfiability_list, current_node[0])
                    # print("total eqs",len(current_formula.eq_list))
                    # for eq,label in zip(current_formula.eq_list,label_list):
                    #     print(eq.eq_str,label)

            return (current_node[1]["status"], current_formula, current_node)

    def extract_dynamic_embedding_train_data(self, branch_eq_satisfiability_list, node_id):
        self.total_output_branches += 1

        label_list = [0] * len(branch_eq_satisfiability_list)
        satisfiability_list = []
        back_track_count_list = []
        middle_branch_eq_file_name_list = []
        one_train_data_name = f"{self.file_name}@{node_id}"
        for index, (eq, satisfiability, branch_number) in enumerate(branch_eq_satisfiability_list):
            satisfiability_list.append(satisfiability)
            back_track_count_list.append(branch_number)
            one_eq_file_name = f"{self.file_name}@{node_id}:{index}"

            eq.output_eq_file_rank(one_eq_file_name, satisfiability)
            middle_branch_eq_file_name_list.append(os.path.basename(one_eq_file_name) + ".eq")


        # output one-hot encoding labels
        if satisfiability_list.count(SAT) == 1:
            label_list[satisfiability_list.index(SAT)] = 1
        elif satisfiability_list.count(SAT) > 1:
            sat_back_track_counts = [(index, back_track_count_list[index]) for index, value in
                                     enumerate(satisfiability_list) if value == SAT]
            min_back_track_count_index = min(sat_back_track_counts, key=lambda x: x[1])[0]
            label_list[min_back_track_count_index] = 1
        elif satisfiability_list.count(SAT) == 0:
            # mix unsat and unknown
            if satisfiability_list.count(UNSAT) >= 1 and satisfiability_list.count(UNKNOWN) >= 1:
                if satisfiability_list.count(UNKNOWN) == 1:
                    label_list[satisfiability_list.index(UNKNOWN)] = 1
                else: #satisfiability_list.count(UNKNOWN)>1
                    unknown_back_track_counts = [(index, back_track_count_list[index]) for index, value in
                                                 enumerate(satisfiability_list) if value == UNKNOWN]
                    min_back_track_count_index = min(unknown_back_track_counts, key=lambda x: x[1])[0]
                    label_list[min_back_track_count_index] = 1
            else: #only unsat or unknown
                min_back_track_count_index = back_track_count_list.index(min(back_track_count_list))
                label_list[min_back_track_count_index]=1

        assert sum(label_list) == 1

        # write label_list to file
        label_dict = {"satisfiability_list": satisfiability_list, "back_track_count_list": back_track_count_list,
                      "label_list": label_list, "middle_branch_eq_file_name_list": middle_branch_eq_file_name_list}
        dump_to_json_with_format(label_dict, one_train_data_name + ".label.json")

        self.train_data_count += 1
        return satisfiability_list,label_list

    def get_eq_by_index(self, f: Formula, index: int) -> Tuple[Equation, Formula]:
        poped_eq = f.eq_list.pop(index)
        return poped_eq, Formula(f.eq_list)

    def get_first_eq(self, f: Formula) -> Tuple[Equation, Formula]:
        return f.eq_list[0], Formula(f.eq_list[1:])

    def _order_branches_fixed(self, children: List[Tuple[Equation, Formula]]) -> List[Tuple[Equation, Formula]]:
        return children

    def _order_branches_random(self, children: List[Tuple[Equation, Formula]]) -> List[Tuple[Equation, Formula]]:
        random.shuffle(children)
        return children

    def early_termination_condition_0(self, current_depth: int):
        return None

    def early_termination_condition_1(self, current_depth: int):
        if current_depth > self.restart_max_deep:
            return UNKNOWN

    def early_termination_condition_2(self, current_depth: int):
        if current_depth > self.max_deep_for_extraction:
            return UNKNOWN

    def early_termination_condition_3(self, current_depth: int):
        if self.found_path >= self.max_found_path_extraction:
            return UNKNOWN

    def early_termination_condition_4(self, current_depth: int):
        if self.found_sat_path >= self.max_found_sat_path_extraction:
            return UNKNOWN

    def visualize(self, file_path: str, graph_func: Callable):
        visualize_path_html(self.nodes, self.edges, file_path)
        visualize_path_png(self.nodes, self.edges, file_path, compress=compress_image, edge_label=self.png_edge_label)
