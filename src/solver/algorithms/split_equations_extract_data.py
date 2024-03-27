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
    simplify_and_check_formula


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
        self.eq_node_number=0
        self.total_node_number=1
        self.restart_max_deep = RESTART_INITIAL_MAX_DEEP
        #control path number for extraction
        self.max_deep_for_extraction=3
        self.max_found_sat_path_extraction=5
        self.found_sat_path=0
        self.max_found_path_extraction=20
        self.found_path=0
        self.task= parameters["task"]
        self.file_name = strip_file_name_suffix(parameters["file_path"])
        self.train_data_count=0

        self.order_equations_func_map = {"fixed": self._order_equations_fixed,
                                         "random": self._order_equations_random,
                                         "category": self._order_equations_category}
        self.order_equations_func: Callable = self.order_equations_func_map[self.parameters["order_equations_method"]]

        self.branch_method_func_map = {"fixed": self._order_branches_fixed,
                                       "random": self._order_branches_random}
        self.order_branches_func: Callable = self.branch_method_func_map[self.parameters["branch_method"]]

        self.check_termination_condition_map = {"termination_condition_0": self.early_termination_condition_0, #no limit
                                                "termination_condition_1": self.early_termination_condition_1, #restart
                                                "termination_condition_2": self.early_termination_condition_2, #max deepth
                                                "termination_condition_3": self.early_termination_condition_3, #found path
                                                "termination_condition_4": self.early_termination_condition_4} #found sat path
        self.check_termination_condition_func: Callable = self.check_termination_condition_map[
            self.parameters["termination_condition"]]

        sys.setrecursionlimit(recursion_limit)
        print("recursion limit number", sys.getrecursionlimit())

        self.log_enabled = True
        self.png_edge_label= True
    @log_control
    def run(self):
        original_formula = Formula(list(self.equation_list))

        initial_node: Tuple[int, Dict] = (
            0, {"label": "start", "status": None, "output_to_file": False, "shape": "circle",
                "back_track_count": 0})
        self.nodes.append(initial_node)

        satisfiability, new_formula,child_node = self.split_eq(original_formula, current_depth=0, previous_branch_node=initial_node,
                                                    edge_label="start")

        return {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                "variables": self.variables, "terminals": self.terminals}

    def split_eq(self, original_formula: Formula, current_depth: int, previous_branch_node: Tuple[int, Dict],
                 edge_label: str) -> Tuple[str, Formula,Tuple[int, Dict]]:
        self.total_split_eq_call += 1
        print(f"----- total_split_eq_call:{self.total_split_eq_call}, current_depth:{current_depth} -----")

        current_node = self.record_node_and_edges(original_formula, previous_branch_node, edge_label)

        ####################### early termination condition #######################
        res = self.check_termination_condition_func(current_depth)
        if res != None:
            current_node[1]["status"] = res
            current_node[1]["back_track_count"] = 1
            self.found_path+=1
            return (res, original_formula,current_node)

        ####################### search #######################
        satisfiability, current_formula = simplify_and_check_formula(original_formula)

        if satisfiability != UNKNOWN:
            current_node[1]["status"] = satisfiability
            current_node[1]["back_track_count"] = 1
            self.found_path+=1
            return (satisfiability, current_formula,current_node)
        else:
            #systematic search training data
            back_track_count=0
            branch_eq_satisfiability_list: List[Tuple[Equation, str]] = []
            for index, eq in enumerate(list(current_formula.eq_list)):
                current_eq, separated_formula = self.get_eq_by_index(Formula(list(current_formula.eq_list)), index)
                current_eq_node=self.record_eq_node_and_edges(current_eq, previous_node=current_node, edge_label=f"eq:{index}")

                children, fresh_variable_counter = apply_rules(current_eq, separated_formula, self.fresh_variable_counter)
                self.fresh_variable_counter = fresh_variable_counter
                children: List[Tuple[Equation, Formula, str]] = self.order_branches_func(children)

                eq_back_track_count=0
                split_branch_satisfiability_list:List[Tuple[Equation,str,int]] = []
                for c_index, child in enumerate(children):
                    (c_eq, c_formula, edge_label) = child
                    satisfiability, res_formula,child_node = self.split_eq(c_formula, current_depth + 1, previous_branch_node=current_eq_node,
                                                                edge_label=edge_label)
                    back_track_count+=child_node[1]["back_track_count"]
                    eq_back_track_count+=child_node[1]["back_track_count"]
                    split_branch_satisfiability_list.append(satisfiability)


                current_eq_node[1]["back_track_count"] = eq_back_track_count
                if any(eq_satisfiability == SAT for eq_satisfiability in split_branch_satisfiability_list):
                    current_eq_node[1]["status"] = SAT
                    branch_eq_satisfiability_list.append((current_eq, SAT,current_eq_node[1]["back_track_count"] ))
                elif any(eq_satisfiability == UNKNOWN for eq_satisfiability in split_branch_satisfiability_list):
                    current_eq_node[1]["status"] = UNKNOWN
                    branch_eq_satisfiability_list.append((current_eq, UNKNOWN,current_eq_node[1]["back_track_count"] ))
                else:
                    current_eq_node[1]["status"] = UNSAT
                    branch_eq_satisfiability_list.append((current_eq, UNSAT,current_eq_node[1]["back_track_count"] ))

            current_node[1]["back_track_count"] = back_track_count
            if all(eq_satisfiability == SAT for _, eq_satisfiability,_ in branch_eq_satisfiability_list):
                current_node[1]["status"] = SAT
            elif any(eq_satisfiability == UNSAT for _, eq_satisfiability,_ in branch_eq_satisfiability_list):
                current_node[1]["status"] = UNSAT
            else:
                current_node[1]["status"] = UNKNOWN

            #output labeled eqs
            if len(branch_eq_satisfiability_list)>1:
                self.extract_dynamic_embedding_train_data(branch_eq_satisfiability_list,current_node[0])


            return (current_node[1]["status"], current_formula,current_node)

    def extract_dynamic_embedding_train_data(self, branch_eq_satisfiability_list,node_id):
        min_count=min([count for _,_,count in branch_eq_satisfiability_list])
        label_list=[0]*len(branch_eq_satisfiability_list)
        satisfiability_list=[]
        back_track_count_list=[]
        middle_branch_eq_file_name_list=[]
        one_train_data_name = f"{self.file_name}@{node_id}"
        for index,(eq,satisfiability,branch_number) in enumerate(branch_eq_satisfiability_list):
            satisfiability_list.append(satisfiability)
            back_track_count_list.append(branch_number)
            one_eq_file_name = f"{self.file_name}@{node_id}:{index}"
            eq.output_eq_file(one_eq_file_name, satisfiability)
            middle_branch_eq_file_name_list.append(os.path.basename(one_eq_file_name))
            if sum(label_list)<1 and branch_number==min_count:
                label_list[index] = 1

        # write label_list to file
        label_dict = {"satisfiability_list": satisfiability_list, "back_track_count_list": back_track_count_list,
                      "label_list": label_list, "middle_branch_eq_file_name_list": middle_branch_eq_file_name_list}
        dump_to_json_with_format(label_dict, one_train_data_name+".label.json")

        self.train_data_count+=1


    def record_eq_node_and_edges(self, eq: Equation, previous_node: Tuple[int, Dict], edge_label: str) -> Tuple[int, Dict]:
        current_node_number = self.total_node_number
        label = f"{eq.eq_str}"
        current_node = (
            current_node_number,
            {"label": label, "status": None, "output_to_file": False, "shape": "box", "back_track_count": 0})
        self.nodes.append(current_node)
        self.edges.append((previous_node[0], current_node_number, {'label': edge_label}))
        self.eq_node_number+=1
        self.total_node_number+=1
        return current_node

    def record_node_and_edges(self, f: Formula, previous_node: Tuple[int, Dict], edge_label: str) -> \
            Tuple[int, Dict]:
        current_node_number = self.total_node_number
        label = f"{f.eq_list_str}"
        current_node = (
            current_node_number,
            {"label": label, "status": None, "output_to_file": False, "shape": "ellipse", "back_track_count": 0})
        self.nodes.append(current_node)
        self.edges.append((previous_node[0], current_node_number, {'label': edge_label}))
        self.total_node_number+=1
        return current_node

    def get_eq_by_index(self, f: Formula, index: int) -> Tuple[Equation, Formula]:
        poped_eq=f.eq_list.pop(index)
        return poped_eq, Formula(f.eq_list)

    def get_first_eq(self, f: Formula) -> Tuple[Equation, Formula]:
        return f.eq_list[0], Formula(f.eq_list[1:])

    def _order_equations_category(self, f: Formula) -> Formula:
        categoried_eq_list: List[Tuple[Equation, int]] = _category_formula_by_rules(f)
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
        visualize_path_png(self.nodes, self.edges, file_path, compress=compress_image,edge_label=self.png_edge_label)

