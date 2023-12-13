import copy
import os.path
import random
import sys
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Optional, Callable

import torch

from src.solver.Constants import recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR, SAT, UNSAT, UNKNOWN, project_folder, INITIAL_MAX_DEEP, MAX_DEEP_STEP, \
    MAX_SPLIT_CALL, OUTPUT_LEAF_NODE_PERCENTAGE, GNN_BRANCH_RATIO, MAX_ONE_SIDE_LENGTH, MAX_DEEP, compress_image, \
    INITIAL_MAX_DEEP_BOUND_2, MAX_DEEP_STEP_BOUND_2
from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, get_eq_graph_1, SeparateSymbol
from src.solver.algorithms.abstract_algorithm import AbstractAlgorithm
from src.solver.algorithms.utils import graph_to_gnn_format,concatenate_eqs,merge_graphs
from src.solver.independent_utils import remove_duplicates, flatten_list, strip_file_name_suffix, \
    dump_to_json_with_format, identify_available_capitals,get_memory_usage
from src.solver.models.Dataset import get_one_dgl_graph
from src.solver.models.utils import load_model
from src.solver.visualize_util import visualize_path_html, visualize_path_png

sys.path.append(
    project_folder + "/src/solver/models")


class ElimilateVariablesRecursive(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation],
                 parameters: Dict):

        super().__init__(terminals, variables, equation_list)

        self.assignment = Assignment()
        self.parameters = parameters
        self.file_name = strip_file_name_suffix(parameters["file_path"])
        self.total_explore_paths_call = 0
        self.total_split_call = 0
        self.current_deep = 0
        self.explored_deep = 0
        self.gnn_branch_memory_limitation = 1.0
        self.max_deep = INITIAL_MAX_DEEP
        self.nodes = []
        self.edges = []
        self.branch_method_func_map = {"extract_branching_data_task_1": self._extract_branching_data_task_1,
                                       "extract_branching_data_task_2": self._extract_branching_data_task_2,
                                       "extract_branching_data_task_3": self._extract_branching_data_task_3,
                                       "fixed": self._use_fixed_branching,
                                       "random": self._use_random_branching,
                                       "gnn": self._use_gnn_branching,
                                       "gnn:random": self._use_gnn_with_random_branching,
                                       "gnn:fixed": self._use_gnn_with_fixed_branching}
        self._branch_method = parameters["branch_method"]
        self._branch_method_func = self.branch_method_func_map[parameters["branch_method"]]
        self.record_and_close_branch: Callable = self._record_and_close_branch_with_file if parameters[
                                                                                                "branch_method"] == "extract_branching_data_task_1" else self._record_and_close_branch_without_file
        self.graph_func = get_eq_graph_1
        self.termination_condition = parameters["termination_condition"]
        self.termination_condition_map = {
            "execute_termination_condition_0": self._execute_branching_data_termination_condition_0,
            "execute_termination_condition_1": self._execute_branching_data_termination_condition_1,
            "execute_termination_condition_2": self._execute_branching_data_termination_condition_2, }
        self._execute_branching_data_termination_condition = self.termination_condition_map[self.termination_condition]
        sys.setrecursionlimit(recursion_limit)
        # print("recursion limit number", sys.getrecursionlimit())

        if "extract_branching_data" in parameters["branch_method"]:
            self.extract_algorithm = parameters["extract_algorithm"]
            print("extract_algorithm:", self.extract_algorithm)
        if "gnn" in parameters["branch_method"]:
            # load the model from mlflow
            # experiment_id = "856005721390468951"
            # run_id = "feb2e17e68bb4310bb3c539c672fd166"
            # self.gnn_model = load_model_from_mlflow(experiment_id, run_id)
            self.graph_func = parameters["graph_func"]
            if parameters["task"] == "task_1":
                self.gnn_model = load_model(parameters["gnn_model_path"])
                self.branch_prediction_func = self._task_1_and_2_branch_prediction
                self.draw_graph_func = self._draw_graph_task_1
            elif parameters["task"] == "task_2":
                self.gnn_model = load_model(parameters["gnn_model_path"])
                self.branch_prediction_func = self._task_1_and_2_branch_prediction
                self.draw_graph_func = self._draw_graph_task_2
            elif parameters["task"] == "task_3":
                self.gnn_model_2 = load_model(parameters["gnn_model_path"].replace("_0_", "_2_"))
                self.gnn_model_3 = load_model(parameters["gnn_model_path"].replace("_0_", "_3_"))
                self.branch_prediction_func = self._task_3_branch_prediction
        self.output_train_data = False if self.file_name == "" else True

    def run(self):
        print("branch_method:", self.parameters["branch_method"])
        #first_equation = self.equation_list[0]
        concatenated_eqs = concatenate_eqs(self.equation_list)
        print("concatenated_eqs:")
        print(len(concatenated_eqs.eq_left_str),concatenated_eqs.eq_left_str)
        print(len(concatenated_eqs.eq_right_str),concatenated_eqs.eq_right_str)
        target_eqs = concatenated_eqs

        try:
            if self.termination_condition == "execute_termination_condition_2":
                self.max_deep = INITIAL_MAX_DEEP_BOUND_2
                while True:
                    self.nodes = []
                    self.edges = []
                    node_info = (0, {"label": "start", "status": None, "output_to_file": False, "shape": "ellipse",
                                     "back_track_count": 0})
                    self.nodes.append(node_info)
                    satisfiability, variables, back_track_count = self.explore_paths(target_eqs,
                                                                                     {"node_number": node_info[0],
                                                                                      "label": node_info[1]["label"]})
                    if satisfiability == SAT or satisfiability == UNSAT:
                        break
                    if self.max_deep >= MAX_DEEP:
                        break
                    self.max_deep += MAX_DEEP_STEP_BOUND_2
                    print("max_deep extended", self.max_deep)
            else:
                node_info = (0, {"label": "start", "status": None, "output_to_file": False, "shape": "ellipse",
                                 "back_track_count": 0})
                self.nodes.append(node_info)
                satisfiability, variables, back_track_count = self.explore_paths(target_eqs,
                                                                                 {"node_number": node_info[0],
                                                                                  "label": node_info[1]["label"]})

        except RecursionError as e:
            if "maximum recursion depth exceeded" in str(e):
                satisfiability = RECURSION_DEPTH_EXCEEDED
                # print(RECURSION_DEPTH_EXCEEDED)
            else:
                satisfiability = RECURSION_ERROR
                # print(RECURSION_ERROR)

        result_dict = {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                       "variables": self.variables, "terminals": self.terminals,
                       "total_explore_paths_call": self.total_explore_paths_call, "explored_deep": self.explored_deep}
        return result_dict



    def explore_paths(self, current_eq: Equation,
                      previous_dict: Dict) -> Tuple[str, List[Variable], List[int]]:
        self.total_explore_paths_call += 1
        self.current_deep += 1
        if self.explored_deep < self.current_deep:
            self.explored_deep = self.current_deep
        #print(f"explore_paths call: {self.total_explore_paths_call}")
        # print(previous_dict)
        # print(len(current_eq.eq_left_str),current_eq.eq_left_str)
        # print(len(current_eq.eq_right_str),current_eq.eq_right_str)
        # print(f"current_deep: {self.current_deep}, max deep: {self.max_deep}")



        ################################ Record nodes and edges ################################

        current_node_number = self.total_explore_paths_call
        node_info = (
            current_node_number,
            {"label": current_eq.eq_str, "status": None, "output_to_file": False, "shape": "ellipse"})
        self.nodes.append(node_info)
        self.edges.append((previous_dict["node_number"], current_node_number, {'label': previous_dict["label"]}))

        ################################ Check terminate conditions ################################

        ## both side contains variables and terminals
        ###both side empty
        if len(current_eq.left_terms) == 0 and len(current_eq.right_terms) == 0:
            return self.record_and_close_branch(SAT, current_eq.variable_list, node_info, current_eq)
        ### left side empty
        if len(current_eq.left_terms) == 0 and len(current_eq.right_terms) != 0:
            if len(current_eq.termimal_list_without_empty_terminal) != 0 and current_eq.variable_number != 0:  # terminals+variables
                return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)
            elif len(
                    current_eq.termimal_list_without_empty_terminal) == 0 and current_eq.variable_number != 0:  # variables
                return self.record_and_close_branch(SAT, current_eq.variable_list, node_info, current_eq)
            else:  # terminals
                return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)
        ### right side empty
        if len(current_eq.left_terms) != 0 and len(current_eq.right_terms) == 0:
            if len(current_eq.termimal_list_without_empty_terminal) != 0 and current_eq.variable_number != 0:  # terminals+variables
                return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)
            elif len(
                    current_eq.termimal_list_without_empty_terminal) == 0 and current_eq.variable_number != 0:  # variables
                return self.record_and_close_branch(SAT, current_eq.variable_list, node_info, current_eq)
            else:  # terminals
                return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)

        ## both side only have terminals
        if current_eq.variable_number == 0:
            # satisfiability = SAT if self.check_equation(current_eq.left_terms,current_eq.right_terms) == True else UNSAT
            return self.record_and_close_branch(current_eq.check_all_terminal_case(), current_eq.variable_list,
                                                node_info, current_eq)

        ## both side only have variables
        if len(current_eq.termimal_list_without_empty_terminal) == 0:
            return self.record_and_close_branch(SAT, current_eq.variable_list, node_info, current_eq)

        ## special cases
        ### special case: one side only have terminals and another side have longer terminals.
        ### special case: variables surrounded by identical terminals
        ### special case: variables surrounded by different terminals
        ### special case: starting or ending with variables

        ### special case 1: mismatched leading or tailing terminals
        left_leading_terminals, first_left_non_terminal_term = self.get_leading_terminals(current_eq.left_terms)
        right_leading_terminals, first_right_non_terminal_term = self.get_leading_terminals(current_eq.right_terms)
        if len(left_leading_terminals) > 0 and len(right_leading_terminals) > 0 and len(left_leading_terminals) == len(
                right_leading_terminals) and first_left_non_terminal_term != None and first_right_non_terminal_term != None and left_leading_terminals != right_leading_terminals and first_left_non_terminal_term == first_right_non_terminal_term:
            return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)

        left_tailing_terminals, first_left_non_terminal_term = self.get_leading_terminals(
            list(reversed(current_eq.left_terms)))
        right_tailing_terminals, first_right_non_terminal_term = self.get_leading_terminals(
            list(reversed(current_eq.right_terms)))
        if len(left_tailing_terminals) > 0 and len(right_tailing_terminals) > 0 and len(left_tailing_terminals) == len(
                right_tailing_terminals) and first_left_non_terminal_term != None and first_right_non_terminal_term != None and left_tailing_terminals != right_tailing_terminals and first_left_non_terminal_term == first_right_non_terminal_term:
            return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)

        ### special case 2: one side only have one variable, e,g. M = terminals+variables SAT, M = terminals SAT, M = variables SAT, M="" SAT
        if current_eq.number_of_special_symbols == 0:
            if (len(current_eq.left_terms) == 1 and current_eq.left_terms[0].value_type == Variable):
                if current_eq.left_terms[
                    0] in current_eq.right_terms and current_eq.terminal_numbers_without_empty_terminal != 0:  # M = terminals+variables and M in right hand side
                    return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)
                else:
                    return self.record_and_close_branch(SAT, current_eq.variable_list, node_info, current_eq)
            if (len(current_eq.right_terms) == 1 and current_eq.right_terms[0].value_type == Variable):
                if current_eq.right_terms[
                    0] in current_eq.left_terms and current_eq.terminal_numbers_without_empty_terminal != 0:
                    return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)
                else:
                    return self.record_and_close_branch(SAT, current_eq.variable_list, node_info, current_eq)

        ################################ Split equation ################################
        left_term = current_eq.left_terms[0]
        right_term = current_eq.right_terms[0]
        # both side are the same
        if left_term.value == right_term.value:
            return self.both_side_same_terms(current_eq, current_eq.variable_list, current_node_number, node_info)

        # both side are different
        else:
            ## both side are differernt variables
            if type(left_term.value) == Variable and type(right_term.value) == Variable:
                return self.both_side_different_variables(current_eq,
                                                          current_node_number, node_info)

            ## left side is variable, right side is terminal
            elif type(left_term.value) == Variable and type(right_term.value) == Terminal:
                return self.left_side_variable_right_side_terminal(current_eq,
                                                                   current_node_number, node_info)

            ## left side is terminal, right side is variable
            elif type(left_term.value) == Terminal and type(right_term.value) == Variable:
                return self.left_side_variable_right_side_terminal(
                    Equation(current_eq.right_terms, current_eq.left_terms),
                    current_node_number, node_info)

            ## both side are different terminals
            elif type(left_term.value) == Terminal and type(right_term.value) == Terminal:
                return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)

            ## one side is # and another side is terminal
            elif type(left_term.value) == SeparateSymbol and type(right_term.value) == Terminal:
                return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)
            elif type(right_term.value)==SeparateSymbol and type(left_term.value)==Terminal:
                return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)
            ## oneside is # and another side is variable
            elif type(left_term.value)==Variable and type(right_term.value)==SeparateSymbol:
                return self.left_side_vairable_right_side_special_symbol(current_eq,current_node_number,node_info)

            elif type(left_term.value) == SeparateSymbol and type(right_term.value) == Variable:
                return self.left_side_vairable_right_side_special_symbol(Equation(current_eq.right_terms,current_eq.left_terms),current_node_number,node_info)



    def both_side_same_terms(self, eq: Equation, variables: List[Variable],
                             current_node_number, node_info):
        left_poped_list_str, right_poped_list_str, l, r = self.pop_both_same_terms(eq.left_terms, eq.right_terms,
                                                                                   eq.variable_list)
        new_eq = Equation(list(l), list(r))
        # updated_variables = self.update_variables(eq.left_terms, eq.right_terms)
        branch_satisfiability, branch_variables, back_track_count_list = self.explore_paths(new_eq,
                                                                                            {
                                                                                                "node_number": current_node_number,
                                                                                                "label": left_poped_list_str + "=" + right_poped_list_str})
        back_track_count_list = [x + 1 for x in back_track_count_list]
        return self.record_and_close_branch(branch_satisfiability, branch_variables, node_info, eq,
                                            back_track_count=back_track_count_list)

    def both_side_different_variables(self, eq: Equation,
                                      current_node_number, node_info):
        # Define the methods for each branch
        branch_methods = [
            self.two_variables_split_branch_1,
            self.two_variables_split_branch_2,
            self.two_variables_split_branch_3  # assume two variables are the same
        ]
        self.total_split_call += 1
        return self._branch_method_func(eq, current_node_number,
                                        node_info, branch_methods)

    def left_side_variable_right_side_terminal(self, eq: Equation,
                                               current_node_number, node_info):
        # Define the methods for each branch
        branch_methods = [
            self.one_variable_one_terminal_split_branch_1,  # assume the variable is empty
            self.one_variable_one_terminal_split_branch_2
        ]
        self.total_split_call += 1
        return self._branch_method_func(eq, current_node_number,
                                        node_info, branch_methods)
    def left_side_vairable_right_side_special_symbol(self, eq: Equation,current_node_number, node_info):

        #
        # branch_methods=[self.one_variable_one_terminal_split_branch_1]
        # self.total_split_call += 1
        # return self._branch_method_func(eq, current_node_number,
        #                                 node_info, branch_methods)


        l, r, updated_variables, edge_label=self.one_variable_one_terminal_split_branch_1(eq.left_terms, eq.right_terms, eq.variable_list)
        new_eq = Equation(list(l), list(r))
        # updated_variables = self.update_variables(eq.left_terms, eq.right_terms)
        branch_satisfiability, branch_variables, back_track_count_list = self.explore_paths(new_eq,
                                                                                            {"node_number": current_node_number,
                                                                                                "label": edge_label})
        back_track_count_list = [x + 1 for x in back_track_count_list]
        return self.record_and_close_branch(branch_satisfiability, branch_variables, node_info, eq,
                                            back_track_count=back_track_count_list)




    def _use_gnn_with_random_branching(self, eq: Equation, current_node_number, node_info, branch_methods):
        if random.random() < GNN_BRANCH_RATIO:  # random.random() generate a float between 0 to 1
            return self._use_gnn_branching(eq, current_node_number, node_info, branch_methods)
        else:
            return self._use_random_branching(eq, current_node_number, node_info, branch_methods)

    def _use_gnn_with_fixed_branching(self, eq: Equation, current_node_number, node_info, branch_methods):
        if random.random() < GNN_BRANCH_RATIO:  # random.random() generate a float between 0 to 1
            return self._use_gnn_branching(eq, current_node_number, node_info, branch_methods)
        else:
            return self._use_fixed_branching(eq, current_node_number, node_info, branch_methods)

    def _task_3_branch_prediction(self, eq: Equation, branch_methods):
        split_graph_list = []
        split_eq_list = []
        edge_label_list = []
        for i, method in enumerate(branch_methods):
            # branch
            l, r, _, edge_label = method(eq.left_terms, eq.right_terms, eq.variable_list)
            split_eq_nodes, split_eq_edges = self.graph_func(l, r)
            graph_dict = graph_to_gnn_format(split_eq_nodes, split_eq_edges)
            # Load data
            dgl_graph, _ = get_one_dgl_graph(graph_dict)
            split_graph_list.append(dgl_graph)
            # record
            split_eq: Equation = Equation(l, r)
            split_eq_list.append(split_eq)
            edge_label_list.append(edge_label)
        # predict
        with torch.no_grad():
            # todo this can be improved by passing functions
            if len(branch_methods) == 2:
                pred_list = self.gnn_model_2(split_graph_list).squeeze()  # separate model returns a float number
                # [1 if label == [1,0] else 0 for label in self.labels]
                if pred_list > 0.5:
                    pred_list = [1, 0]
                else:
                    pred_list = [0, 1]
                # pred_list=[1,0]#this make it use fixed branching

            elif len(branch_methods) == 3:
                pred_list = self.gnn_model_3(split_graph_list).squeeze()
                # pred_list=[1,0.5,0]#this make it use fixed branching


        # sort
        prediction_list = []
        for pred, split_eq, edge_label in zip(pred_list, split_eq_list, edge_label_list):
            prediction_list.append([pred, (split_eq, edge_label)])

        sorted_prediction_list = sorted(prediction_list, key=lambda x: x[0], reverse=True)
        # print([x[0] for x in sorted_prediction_list])
        return sorted_prediction_list

    def _task_1_and_2_branch_prediction(self, eq: Equation, branch_methods):
        prediction_list = []
        for method in branch_methods:
            # branch
            l, r, _, edge_label = method(eq.left_terms, eq.right_terms, eq.variable_list)
            split_eq = Equation(l, r)
            # draw graph
            nodes, edges = self.draw_graph_func(split_eq, eq)
            graph_dict = graph_to_gnn_format(nodes, edges)
            # Load data
            dgl_graph, _ = get_one_dgl_graph(graph_dict)
            # predict
            with torch.no_grad():
                pred = self.gnn_model(dgl_graph).squeeze()  # pred is a float between 0 and 1
                # pred = self.gnn_model(bached_graph,bached_graph.ndata["feat"].float())  # pred is a float between 0 and 1

            prediction_list.append([pred, (split_eq, edge_label)])

        sorted_prediction_list = sorted(prediction_list, key=lambda x: x[0], reverse=True)
        # print([x[0] for x in sorted_prediction_list])
        return sorted_prediction_list

    def _draw_graph_task_1(self, split_eq, eq):
        nodes, edges = self.graph_func(split_eq.left_terms, split_eq.right_terms)
        return nodes, edges

    def _draw_graph_task_2(self, split_eq, eq):
        split_nodes, split_edges = self.graph_func(split_eq.left_terms, split_eq.right_terms)
        eq_nodes, eq_edges = self.graph_func(eq.left_terms, eq.right_terms)
        merged_nodes, merged_edges = merge_graphs(eq_nodes, eq_edges, split_nodes, split_edges)
        return merged_nodes, merged_edges

    def _use_gnn_branching(self, eq: Equation, current_node_number, node_info, branch_methods):
        ################################ stop branching condition ################################
        result = self._execute_branching_data_termination_condition(eq, node_info)
        if result == None:
            pass
        else:
            return result


        # if self.total_split_call%50 ==0:
        #     memory_text,gb=get_memory_usage()
        #     if gb>self.gnn_branch_memory_limitation:
        #         self.gnn_branch_memory_limitation+=0.2
        #         #print(eq.eq_str)
        #         return self.record_and_close_branch(UNKNOWN, eq.variable_list, node_info, eq)
        #
        # print(f"- {self.total_split_call} gnn branch -")

        ################################ prediction ################################
        sorted_prediction_list = self.branch_prediction_func(eq, branch_methods)

        # Perform depth-first search based on the sorted prediction list
        satisfiability_list = []
        for i, data in enumerate(sorted_prediction_list):
            split_eq, edge_label = data[1]
            satisfiability, branch_variables, back_track_count_list = self.explore_paths(split_eq,
                                                                                         {
                                                                                             "node_number": current_node_number,
                                                                                             "label": edge_label})
            satisfiability_list.append(satisfiability)

            # Handle branch outcome
            back_track_count_list = [x + 1 for x in back_track_count_list]
            result = self._handle_one_split_branch_outcome(i, branch_methods, satisfiability, branch_variables,
                                                           node_info, split_eq, back_track_count=back_track_count_list,
                                                           satisfiability_list=satisfiability_list)

            if result == None:
                pass
            else:
                return result

    def _use_random_branching(self, eq: Equation, current_node_number, node_info,
                              branch_methods):
        # print(self.total_split_call,"random branch")
        random.shuffle(branch_methods)
        return self._use_fixed_branching(eq, current_node_number, node_info,
                                         branch_methods)

    def _use_fixed_branching(self, eq: Equation, current_node_number, node_info,
                             branch_methods):
        ################################ stop branching condition ################################
        result = self._execute_branching_data_termination_condition(eq, node_info)
        if result == None:
            pass
        else:
            return result

        # print(self.total_split_call,"fixed branch")
        # # Print the current memory usage
        # print(f"Memory usage: {get_memory_usage()}")

        satisfiability_list = []
        for i, branch in enumerate(branch_methods):
            l, r, _, edge_label = branch(eq.left_terms, eq.right_terms, eq.variable_list)
            split_eq = Equation(l, r)

            satisfiability, branch_variables, back_track_count_list = self.explore_paths(split_eq,
                                                                                         {
                                                                                             "node_number": current_node_number,
                                                                                             "label": edge_label})
            # print(satisfiability, edge_label)
            # print(eq.eq_left_str)
            # print(eq.eq_right_str)

            satisfiability_list.append(satisfiability)

            # Handle branch outcome
            back_track_count_list = [x + 1 for x in back_track_count_list]
            result = self._handle_one_split_branch_outcome(i, branch_methods, satisfiability, branch_variables,
                                                           node_info, split_eq, back_track_count=back_track_count_list,
                                                           satisfiability_list=satisfiability_list)
            if result == None:
                pass
            else:
                return result

    def _handle_one_split_branch_outcome(self, i, branch_methods, satisfiability, branch_variables, node_info, eq,
                                         back_track_count: List[int], satisfiability_list=None):
        if i < len(branch_methods) - 1:  # not last branch
            if satisfiability == SAT:
                return self.record_and_close_branch(SAT, branch_variables, node_info, eq,
                                                    back_track_count=back_track_count)
            elif satisfiability == UNSAT:
                node_info[1]["status"] = UNSAT
                return None
            elif satisfiability == UNKNOWN:
                node_info[1]["status"] = UNKNOWN
                return None

        else:  # last branch

            satisfiability = self._get_satisfiability_from_satisfiability_list(satisfiability_list)

            return self.record_and_close_branch(satisfiability, branch_variables, node_info, eq,
                                                back_track_count=back_track_count)

    def _extract_branching_data_task_3(self, eq: Equation, current_node_number, node_info, branch_methods):
        return self._extract_branching_data_task_2(eq, current_node_number, node_info, branch_methods)

    def _extract_branching_data_task_2(self, eq: Equation, current_node_number, node_info, branch_methods):
        #print(self.total_split_call, "fixed branch")
        # Print the current memory usage
        # print(f"Memory usage: {get_memory_usage()}")

        ################################ stop branching condition ################################
        result = self._extract_branching_data_termination_condition(eq, node_info)
        if result == None:
            pass
        else:
            return result

        ################################ branching ################################
        satisfiability_list, back_track_count_list, branch_eq_list = self._extract_branching_data_branching(eq,
                                                                                                            branch_methods,
                                                                                                            current_node_number)

        ################################ output train data ################################
        back_track_count_list = [x + 1 for x in back_track_count_list]
        current_eq_satisfiability = self._get_satisfiability_from_satisfiability_list(satisfiability_list)

        # draw two eq graphs
        # output current node to eq file
        middle_eq_file_name = f"{self.file_name}@{node_info[0]}"
        self._output_train_data(middle_eq_file_name, eq, current_eq_satisfiability, node_info, "diamond")

        if self.output_train_data == True:
            # output splited nodes to eq files
            middle_branch_eq_file_name_list = []
            for branch_index, (branch_eq, satisfiability) in enumerate(zip(branch_eq_list, satisfiability_list)):
                middle_branch_eq_file_name = f"{self.file_name}@{node_info[0]}:{branch_index}"
                branch_eq.output_eq_file(middle_branch_eq_file_name, satisfiability)
                middle_branch_eq_file_name_list.append(os.path.basename(middle_branch_eq_file_name + ".eq"))

            # output pairs' labels to file
            if len(back_track_count_list) == 2:
                # two branches 2 (positions) x 3 (conditions) = 6 situations
                ## 2 SAT  label [1,0]
                ## 1 SAT 1 others [1,0]
                ## 2 UNSAT [1,0]
                ## 1 UNSAT 1 UNKNOWN [0,1]
                ## 2 UNKNOWN []
                label_list = [0, 0]
                # if the satisfiabilities are the same the shortest back_track_count has label 1 and others are 0
                if satisfiability_list.count(SAT) == 2 or satisfiability_list.count(
                        UNSAT) == 2 or satisfiability_list.count(UNKNOWN) == 2:  # 2 SAT or 2 UNSAT or 2 UNKNOWN
                    min_value = min(back_track_count_list)
                    min_value_indeces = [i for i, x in enumerate(back_track_count_list) if x == min_value]
                    for min_value_index in min_value_indeces:
                        label_list[min_value_index] = 1
                elif satisfiability_list.count(SAT) == 1:  # 1 SAT 1 others
                    label_list[satisfiability_list.index(SAT)] = 1
                elif satisfiability_list.count(UNSAT) == 1 and satisfiability_list.count(
                        UNKNOWN) == 1:  # 1 UNSAT 1 UNKNOWN
                    label_list[satisfiability_list.index(UNKNOWN)] = 1



            elif len(back_track_count_list) == 3:
                # three branches 3 (positions) x 3 (conditions) = 9 situations
                ## 3 SAT
                ## 3 UNSAT
                ## 3 UNKNOWN
                ## 2 SAT 1 others [UNSAT or UNKNOWN]
                ## 1 SAT 2 others [UNSAT or UNKNOWN]
                ## 2 UNKNOWN 1 UNSAT
                ## 2 UNSAT 1 UNKNWON

                label_list = [0, 0, 0]
                if satisfiability_list.count(SAT) == 3 or satisfiability_list.count(
                        UNSAT) == 3 or satisfiability_list.count(UNKNOWN) == 3:
                    min_value = min(back_track_count_list)
                    min_value_indeces = [i for i, x in enumerate(back_track_count_list) if x == min_value]
                    for min_value_index in min_value_indeces:
                        label_list[min_value_index] = 1
                elif satisfiability_list.count(SAT) == 2:
                    sat_indices = [index for index, value in enumerate(satisfiability_list) if value == SAT]
                    others_indices = [index for index, value in enumerate(satisfiability_list) if value != SAT]
                    min_value = min([back_track_count_list[i] for i in sat_indices])
                    min_value_indeces = [i for i, x in enumerate(back_track_count_list) if
                                         x == min_value and i not in others_indices]
                    for min_value_index in min_value_indeces:
                        label_list[min_value_index] = 1
                elif satisfiability_list.count(SAT) == 1:
                    label_list[satisfiability_list.index(SAT)] = 1
                elif satisfiability_list.count(UNSAT) == 1 and satisfiability_list.count(UNKNOWN) == 2:
                    unknown_indices = [index for index, value in enumerate(satisfiability_list) if value == UNKNOWN]
                    others_indices = [index for index, value in enumerate(satisfiability_list) if value != UNKNOWN]
                    min_value = min([back_track_count_list[i] for i in unknown_indices])
                    min_value_indeces = [i for i, x in enumerate(back_track_count_list) if
                                         x == min_value and i not in others_indices]
                    for min_value_index in min_value_indeces:
                        label_list[min_value_index] = 1

                elif satisfiability_list.count(UNSAT) == 2 and satisfiability_list.count(UNKNOWN) == 1:
                    label_list[satisfiability_list.index(UNKNOWN)] = 1
            else:
                label_list = [0, 0, 0]

            # write label_list to file
            label_json_file_name = middle_eq_file_name + ".label.json"
            label_dict = {"satisfiability_list": satisfiability_list, "back_track_count_list": back_track_count_list,
                          "label_list": label_list, "middle_branch_eq_file_name_list": middle_branch_eq_file_name_list}
            dump_to_json_with_format(label_dict, label_json_file_name)

        # return result
        return self.record_and_close_branch(current_eq_satisfiability, eq.variable_list, node_info, eq,
                                            back_track_count=back_track_count_list)

    def _extract_branching_data_task_1(self, eq: Equation, current_node_number, node_info, branch_methods):
        #print(self.total_split_call,"fixed branch")
        # Print the current memory usage
        # print(f"Memory usage: {get_memory_usage()}")

        ################################ stop branching condition ################################
        result = self._extract_branching_data_termination_condition(eq, node_info)
        if result == None:
            pass
        else:
            return result

        ################################ branching ################################
        satisfiability_list, back_track_count_list, _ = self._extract_branching_data_branching(eq, branch_methods,
                                                                                               current_node_number)

        ################################ output train data ################################
        back_track_count_list = [x + 1 for x in back_track_count_list]
        current_eq_satisfiability = self._get_satisfiability_from_satisfiability_list(satisfiability_list)

        return self.record_and_close_branch_and_output_eq(current_eq_satisfiability, eq.variable_list, node_info, eq,
                                                          back_track_count=back_track_count_list)

    def _extract_branching_data_branching(self, eq: Equation, branch_methods, current_node_number):
        ################################ branching ################################
        satisfiability_list = []
        back_track_count_list = []
        branch_eq_list = []
        if self.extract_algorithm == "random":
            random.shuffle(branch_methods)

        for i, branch in enumerate(branch_methods):
            l, r, v, edge_label = branch(eq.left_terms, eq.right_terms, eq.variable_list)
            branch_eq = Equation(l, r)

            satisfiability, branch_variables, back_track_count = self.explore_paths(branch_eq,
                                                                                    {"node_number": current_node_number,
                                                                                     "label": edge_label})

            satisfiability_list.append(satisfiability)
            back_track_count_list.append(sum(back_track_count))
            branch_eq_list.append(branch_eq)

        return satisfiability_list, back_track_count_list, branch_eq_list

    def _execute_branching_data_termination_condition_0(self, eq: Equation, node_info):
        return None

    def _execute_branching_data_termination_condition_1(self, eq: Equation, node_info):
        if self.current_deep > self.max_deep:
            # print("max deep reached",self.current_deep)
            self.max_deep += MAX_DEEP_STEP
            return self.record_and_close_branch(UNKNOWN, eq.variable_list, node_info, eq)

        return None

    def _execute_branching_data_termination_condition_2(self, eq: Equation, node_info):
        if self.current_deep >= self.max_deep:
            # print("max deep reached",self.current_deep)
            return self.record_and_close_branch(UNKNOWN, eq.variable_list, node_info, eq)

        return None

    def _extract_branching_data_termination_condition(self, eq: Equation, node_info):
        if eq.left_hand_side_length > MAX_ONE_SIDE_LENGTH or eq.right_hand_side_length > MAX_ONE_SIDE_LENGTH:
            return self.record_and_close_branch(UNKNOWN, eq.variable_list, node_info, eq)

        if self.total_split_call > MAX_SPLIT_CALL:
            return self.record_and_close_branch(UNKNOWN, eq.variable_list, node_info, eq)

        return None

    def _get_satisfiability_from_satisfiability_list(self, satisfiability_list: List[str]) -> str:
        # if there is an element in satisfiability_list is SAT, return SAT
        if SAT in satisfiability_list:
            current_eq_satisfiability = SAT
        elif UNKNOWN in satisfiability_list:
            current_eq_satisfiability = UNKNOWN
        else:
            current_eq_satisfiability = UNSAT
        return current_eq_satisfiability

    def record_and_close_branch_and_output_eq(self, satisfiability: str, variables, node_info: Tuple[int, Dict],
                                              eq: Equation,
                                              back_track_count):  # non-leaf node
        if satisfiability != UNKNOWN:
            middle_eq_file_name = self.file_name + "_" + str(node_info[0])
            self._output_train_data(middle_eq_file_name, eq, satisfiability, node_info, "diamond")

        return self._record_and_close_branch_without_file(satisfiability, variables, node_info, eq,
                                                          back_track_count=back_track_count)

    def _record_and_close_branch_with_file(self, satisfiability: str, variables: List[Variable],
                                           node_info: Tuple[int, Dict], eq: Equation,
                                           back_track_count: List[int] = [0]) -> Tuple[
        str, List[Variable], List[int]]:  # leaf node
        if satisfiability != UNKNOWN:
            if random.random() < OUTPUT_LEAF_NODE_PERCENTAGE:  # random.random() generates a float between 0.0 and 1.0
                middle_eq_file_name = self.file_name + "_" + str(node_info[0])
                self._output_train_data(middle_eq_file_name, eq, satisfiability, node_info, "box")
            else:
                pass
        return self._record_and_close_branch_without_file(satisfiability, variables, node_info, eq,
                                                          back_track_count=back_track_count)

    def _record_and_close_branch_without_file(self, satisfiability: str, variables: List[Variable],
                                              node_info: Tuple[int, Dict], eq: Equation,
                                              back_track_count: List[int] = [0]) -> Tuple[
        str, List[Variable], List[int]]:  # leaf node
        node_info[1]["status"] = satisfiability
        node_info[1]["back_track_count"] = back_track_count
        self.current_deep -= 1
        return satisfiability, variables, back_track_count

    def _output_train_data(self, file_name, eq, satisfiability, node_info, shape):
        node_info[1]["output_to_file"] = True
        node_info[1]["shape"] = shape
        if self.output_train_data == True:
            eq.output_eq_file(file_name, satisfiability)

    def pop_both_same_terms(self, left_terms: List[Term], right_terms: List[Term], variables):
        '''
        This will change left_terms_queue and right_terms_queue
        '''
        # local variables
        local_left_terms_queue = deque(copy.deepcopy(left_terms))
        local_right_terms_queue = deque(copy.deepcopy(right_terms))

        equal_term_counter = 0
        for l, r in zip(local_left_terms_queue, local_right_terms_queue):
            if l.value == r.value:
                equal_term_counter += 1
            else:
                break

        left_poped_list = []
        right_poped_list = []
        for i in range(equal_term_counter):
            left_poped_list.append(local_left_terms_queue.popleft())
            right_poped_list.append(local_right_terms_queue.popleft())

        left_poped_list_str = "".join([term.get_value_str for term in left_poped_list])
        right_poped_list_str = "".join([term.get_value_str for term in right_poped_list])

        return left_poped_list_str, right_poped_list_str, local_left_terms_queue, local_right_terms_queue

    def two_variables_split_branch_1(self, left_terms: List[Term], right_terms: List[Term], variables):
        '''
        Equation: V1 [Terms] = V2 [Terms]
        Assume |V1| > |V2|
        Replace V1 with V2V1'
        Obtain V1' [Terms] [V1/V2V1'] = [Terms] [V1/V2V1']
        '''
        # local variables
        local_left_terms_queue = deque(copy.deepcopy(left_terms))
        local_right_terms_queue = deque(copy.deepcopy(right_terms))

        # pop both sides to [Terms] = [Terms]
        left_term = local_left_terms_queue.popleft()
        right_term = local_right_terms_queue.popleft()

        # create fresh variable V1'
        fresh_variable_term = self._create_fresh_variables(variables.copy())
        # replace V1 with V2 V1' to obtain [Terms] [V1/V2V1'] = [Terms] [V1/V2V1']
        self.replace_a_term(old_term=left_term, new_term=[[right_term, fresh_variable_term]],
                            terms_queue=local_left_terms_queue)
        self.replace_a_term(old_term=left_term, new_term=[[right_term, fresh_variable_term]],
                            terms_queue=local_right_terms_queue)

        # construct V1' [Terms] [V1/V2V1'] = [Terms] [V1/V2V1']
        local_left_terms_queue.appendleft(fresh_variable_term)

        # update variables
        updated_variables = self.update_variables(local_left_terms_queue, local_right_terms_queue)

        edge_label = f"{left_term.get_value_str}>{right_term.get_value_str}: {left_term.get_value_str}={right_term.get_value_str}{fresh_variable_term.get_value_str}"

        return local_left_terms_queue, local_right_terms_queue, updated_variables, edge_label

    def two_variables_split_branch_2(self, left_terms: List[Term], right_terms: List[Term], variables):
        '''
        Equation: V1 [Terms] = V2 [Terms]
        Assume |V1| < |V2|
        Replace V2 with V1V2'
        Obtain [Terms] [V2/V1V2'] = V2' [Terms] [V2/V1V2']
        '''
        local_left_terms_queue, local_right_terms_queue, updated_variables, edge_label = self.two_variables_split_branch_1(
            right_terms, left_terms, variables)
        return local_left_terms_queue, local_right_terms_queue, updated_variables, edge_label

    def two_variables_split_branch_3(self, left_terms: List[Term], right_terms: List[Term], variables):
        '''
        Equation: V1 [Terms] = V2 [Terms]
        Assume |V1| = |V2|
        Replace V1 with V2
        Obtain [Terms] [V1/V2] = [Terms] [V1/V2]
        '''
        # local variables
        local_left_terms_queue = deque(copy.deepcopy(left_terms))
        local_right_terms_queue = deque(copy.deepcopy(right_terms))

        # pop both sides to [Terms] = [Terms]
        left_term = local_left_terms_queue.popleft()
        right_term = local_right_terms_queue.popleft()

        # replace V1 with V2 to obtain [Terms] [V1/V2] = [Terms] [V1/V2]
        self.replace_a_term(old_term=left_term, new_term=right_term, terms_queue=local_left_terms_queue)
        self.replace_a_term(old_term=left_term, new_term=right_term, terms_queue=local_right_terms_queue)

        # update variables
        updated_variables = self.update_variables(local_left_terms_queue, local_right_terms_queue)

        edge_label = f"{left_term.get_value_str}={right_term.get_value_str}: {left_term.get_value_str}={right_term.get_value_str}"

        return local_left_terms_queue, local_right_terms_queue, updated_variables, edge_label

    def one_variable_one_terminal_split_branch_1(self, left_terms: List[Term], right_terms: List[Term], variables:List[Variable])->Tuple[Deque[Term],Deque[Term],List[Variable],str]:
        '''
        Equation: V1 [Terms] = a [Terms]
        Assume V1 = ""
        Delete V1
        Obtain [Terms] [V1/""] = a [Terms] [V1/""]
        '''
        # print("*","one_variable_one_terminal_split_branch_1","*")
        # local variables
        local_left_terms_queue = deque(copy.deepcopy(left_terms))
        local_right_terms_queue = deque(copy.deepcopy(right_terms))

        # pop left side to [Terms] = a [Terms]
        left_term = local_left_terms_queue.popleft()

        # delete V1 from both sides to obtain [Terms] [V1/""] = a [Terms] [V1/""]
        new_left_terms_queue = deque(item for item in local_left_terms_queue if item != left_term)
        new_right_terms_queue = deque(item for item in local_right_terms_queue if item != left_term)

        # update variables
        updated_variables = self.update_variables(new_left_terms_queue, new_right_terms_queue)

        edge_label = left_term.get_value_str + "=\"\""

        return new_left_terms_queue, new_right_terms_queue, updated_variables, edge_label

    def one_variable_one_terminal_split_branch_2(self, left_terms: List[Term], right_terms: List[Term], variables):
        '''
        Equation: V1 [Terms] = a [Terms]
        Assume V1 = aV1'
        Replace V1 with aV1'
        Obtain V1' [Terms] [V1/aV1'] = [Terms] [V1/aV1']
        '''
        # print("*","one_variable_one_terminal_split_branch_2","*")
        # local variables
        local_left_terms_queue = deque(copy.deepcopy(left_terms))
        local_right_terms_queue = deque(copy.deepcopy(right_terms))

        # pop both sides to [Terms] = [Terms]
        left_term = local_left_terms_queue.popleft()
        right_term = local_right_terms_queue.popleft()

        # create fresh variable V1'
        fresh_variable_term = self._create_fresh_variables(variables.copy())

        # replace V1 with aV1' to obtain [Terms] [V1/aV1'] = [Terms] [V1/aV1']
        self.replace_a_term(old_term=left_term, new_term=[[right_term, fresh_variable_term]],
                            terms_queue=local_left_terms_queue)
        self.replace_a_term(old_term=left_term, new_term=[[right_term, fresh_variable_term]],
                            terms_queue=local_right_terms_queue)

        # construct V1' [Terms] [V1/aV1'] = [Terms] [V1/aV1']
        local_left_terms_queue.appendleft(fresh_variable_term)

        # update variables
        updated_variables = self.update_variables(local_left_terms_queue, local_right_terms_queue)

        edge_label = f"{left_term.get_value_str}={right_term.get_value_str}{fresh_variable_term.get_value_str}"

        return local_left_terms_queue, local_right_terms_queue, updated_variables, edge_label

    def _create_fresh_variables(self, variables: List[Variable]) -> Term:
        # fresh_variable_term = Term(Variable(left_term.value.value + "'"))  # V1'
        available_caps = identify_available_capitals("".join([v.value for v in variables]))
        fresh_variable_term = Term(Variable(available_caps.pop()))  # a capital rather than V1
        return fresh_variable_term

    def replace_a_term(self, old_term: Term, new_term: Union[List[List[Term]], Term], terms_queue: Deque[Term]):
        for i, t in enumerate(terms_queue):
            if t.value == old_term.value:
                terms_queue[i] = new_term
        term_list = flatten_list(terms_queue)
        terms_queue.clear()
        terms_queue.extend(term_list)

    def update_variables(self, left_term_queue: Deque[Term], right_term_queue: Deque[Term]) -> List[Variable]:
        new_variables = []
        # flattened_left_terms_list = flatten_list(left_term_queue)
        # flattened_right_terms_list = flatten_list(right_term_queue)
        for t in list(left_term_queue) + list(
                right_term_queue):  # flattened_left_terms_list+flattened_right_terms_list:
            if type(t.value) == Variable:
                new_variables.append(t.value)

        return remove_duplicates(new_variables)

    def get_leading_terminals(self, term_list: List[Term]) -> Tuple[List[Term], Optional[Term]]:
        leading_terminal_list = []
        first_non_terminal_term = None
        for t in term_list:
            if t.value_type == Variable or t.value_type == SeparateSymbol:
                first_non_terminal_term = t
                break
            else:
                leading_terminal_list.append(t)
        return leading_terminal_list, first_non_terminal_term

    def visualize(self, file_path: str, graph_func: Callable):
        visualize_path_html(self.nodes, self.edges, file_path)
        visualize_path_png(self.nodes, self.edges, file_path, compress=compress_image)
        concatenated_eqs = concatenate_eqs(self.equation_list)
        concatenated_eqs.visualize_graph(file_path, graph_func)
