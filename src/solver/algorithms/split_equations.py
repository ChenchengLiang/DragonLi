import gc
import random

from typing import List, Dict, Tuple, Callable
from sys import setrecursionlimit, getrecursionlimit
from dgl import batch
from torch import no_grad,stack,mean,concat,cat

from src.solver.Constants import recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR, UNSAT, SAT, UNKNOWN, RESTART_INITIAL_MAX_DEEP, \
    RESTART_MAX_DEEP_STEP, compress_image, HYBRID_ORDER_EQUATION_RATE, RANDOM_SEED
from src.solver.DataTypes import Assignment, Terminal, Variable, Equation, Formula
from src.solver.algorithms.split_equation_utils import _category_formula_by_rules, \
    simplify_and_check_formula, order_equations_random, order_equations_category_random, run_summary, \
    _get_global_info, order_branches_fixed, order_branches_random, \
    order_branches_hybrid_fixed_random, order_equations_static_func_map, _get_unsatcore, apply_rules_prefix, \
    apply_rules_suffix
from src.solver.models.utils import load_model
from src.solver.visualize_util import visualize_path_html, visualize_path_png, draw_graph
from . import graph_to_gnn_format
from .abstract_algorithm import AbstractAlgorithm
from .utils import softmax
from ..independent_utils import log_control, strip_file_name_suffix, hash_graph_with_glob_info, time_it, empty_function
from ..models.Dataset import get_one_dgl_graph


class SplitEquations(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation],
                 parameters: Dict):
        super().__init__(terminals, variables, equation_list)
        self.assignment = Assignment()
        self.parameters = parameters
        self.nodes = []
        self.edges = []
        self.visualize_gnn_input = False
        self.visualize_gnn_input_func = empty_function if self.visualize_gnn_input == False else draw_graph


        self.file_name = strip_file_name_suffix(parameters["file_path"])
        self.unsat_core = _get_unsatcore(self.file_name, parameters, equation_list)

        self.prefix_rules = True
        self.apply_rules = apply_rules_prefix if self.prefix_rules == True else apply_rules_suffix
        self.prefix_suffix_change_frequency = 100
        self.prefix_rules_count = 0
        self.suffix_rules_count = 0
        self.decide_rules_map = {"probability": self._decide_rules_by_probability,
                                 "frequency": self._decide_rules_by_frequency,
                                 "quadratic_pattern": self._decide_rules_by_quadratic_pattern,
                                 "prefix": self._decide_rules_prefix, "suffix": self._decide_rules_suffix}
        self.decide_rules_func = self.decide_rules_map["prefix"]

        self.post_process_ordered_formula_func = self.post_process_ordered_formula_func_map["None"]

        self.fresh_variable_counter = 0
        self.total_gnn_call = 0
        self.total_category_call = 0
        self.total_split_eq_call = 0
        self.total_rank_call = 0
        self.total_node_number = 1
        self.eq_node_number = 0
        self.termination_condition_max_depth = 20000
        self.restart_max_deep = RESTART_INITIAL_MAX_DEEP
        self.each_n_iterations = 10000
        self.first_n_itarations = 1
        self.dynamic_condition_check_point_frequency = 1000
        self.current_eq_size = 1000000
        self.changed_eq_size = 0

        self.rank_task_gnn_func_map = {0: self._order_equations_gnn_rank_task_0,
                                       1: self._order_equations_gnn_rank_task_1,
                                       2: self._order_equations_gnn_rank_task_2,
                                       None: None}

        self._order_equations_gnn = self.rank_task_gnn_func_map[parameters["rank_task"]]
        self.order_equations_func_map = {
            # gnn based
            "category_gnn": self._order_equations_category_gnn,  # first category then gnn
            "category_gnn_each_n_iterations": self._order_equations_category_gnn_each_n_iterations,
            "category_gnn_first_n_iterations": self._order_equations_category_gnn_first_n_iterations,
            "category_gnn_formula_size": self._order_equations_category_gnn_formula_size,
            "hybrid_category_gnn_random": self._order_equations_hybrid_category_gnn_random,
            "hybrid_category_gnn_random_each_n_iterations": self._order_equations_hybrid_category_gnn_random_each_n_iterations,
            "hybrid_category_gnn_random_first_n_iterations": self._order_equations_hybrid_category_gnn_random_first_n_iterations,
            "hybrid_category_gnn_random_formula_size": self._order_equations_hybrid_category_gnn_random_formula_size,
            "gnn": self._order_equations_gnn,
            "gnn_each_n_iterations": self._order_equations_gnn_each_n_iterations,
            "gnn_first_n_iterations": self._order_equations_gnn_first_n_iterations,
            "gnn_formula_size": self._order_equations_gnn_formula_size,
            "hybrid_gnn_random": self._order_equations_hybrid_gnn_random,
            "hybrid_gnn_random_each_n_iterations": self._order_equations_hybrid_gnn_random_each_n_iterations,
            "hybrid_gnn_random_first_n_iterations": self._order_equations_hybrid_gnn_random_first_n_iterations,
            "hybrid_gnn_random_formula_size": self._order_equations_hybrid_gnn_random_formula_size,
        }
        self.order_equations_func_map.update(order_equations_static_func_map)
        self.order_equations_func: Callable = self.order_equations_func_map[self.parameters["order_equations_method"]]
        # load model if call gnn
        if "gnn" in self.parameters["order_equations_method"]:
            gnn_model_path = parameters["gnn_model_path"].replace("_0_", f"_{parameters['label_size']}_")
            self.gnn_rank_model = load_model(gnn_model_path)  # this is a GraphClassifier class
            # self.gnn_rank_model = load_model_torch_script(gnn_model_path)
            # self.gnn_rank_model = load_model_onnx(gnn_model_path)
            self.gnn_rank_model.is_test = True

            self.graph_func = parameters["graph_func"]

            self.empty_graph_dict = {"nodes": [0], "node_types": [5], "edges": [[0, 0]],
                                     "edge_types": [1],
                                     "label": 0, "satisfiability": SAT}
            self.empty_dgl_graph, _ = get_one_dgl_graph(self.empty_graph_dict)

        self.branch_method_func_map = {"fixed": order_branches_fixed,
                                       "random": order_branches_random,
                                       "hybrid_fixed_random": order_branches_hybrid_fixed_random,
                                       "gnn": self._order_branches_gnn}
        self.order_branches_func: Callable = self.branch_method_func_map[self.parameters["branch_method"]]

        self.check_termination_condition_map = {"termination_condition_0": self.early_termination_condition_0,
                                                "termination_condition_1": self.early_termination_condition_1}
        self.check_termination_condition_func: Callable = self.check_termination_condition_map[
            self.parameters["termination_condition"]]

        self.log_enabled = True
        self.png_edge_label = True

        ####### hash tables###
        self.predicted_data_hash_table = {}
        self.predicted_data_hash_table_hit = 0
        self.dgl_hash_table = {}
        self.dgl_hash_table_hit = 0

        print("----- Settings -----")
        setrecursionlimit(recursion_limit)
        print("recursion limit number", getrecursionlimit())

        print("order_equations_method:", self.parameters["order_equations_method"])

    @log_control
    def run(self):
        random.seed(RANDOM_SEED)
        original_formula = Formula(self.equation_list)

        if self.parameters["termination_condition"] == "termination_condition_1":
            while True:
                initial_node: Tuple[int, Dict] = (
                    0, {"label": "start", "status": None, "output_to_file": False, "shape": "ellipse",
                        "back_track_count": 0, "gnn_call": False, "depth": 0,"is_category_call":False})
                try:
                    satisfiability, new_formula = self.split_eq(original_formula, current_depth=0,
                                                                previous_node=initial_node, edge_label="start")
                except RecursionError as e:
                    if "maximum recursion depth exceeded" in str(e):
                        satisfiability = RECURSION_DEPTH_EXCEEDED
                        # print(RECURSION_DEPTH_EXCEEDED)
                    else:
                        satisfiability = RECURSION_ERROR
                        # print(RECURSION_ERROR)

                if satisfiability != UNKNOWN:
                    break

                self.restart_max_deep += RESTART_MAX_DEEP_STEP
        else:
            initial_node: Tuple[int, Dict] = (
                0, {"label": "start", "status": None, "output_to_file": False, "shape": "ellipse",
                    "back_track_count": 0, "gnn_call": False, "depth": 0,"is_category_call":False})
            self.nodes.append(initial_node)
            try:
                satisfiability, new_formula = self.split_eq(original_formula, current_depth=0,
                                                            previous_node=initial_node,
                                                            edge_label="start")
            except RecursionError as e:
                if "maximum recursion depth exceeded" in str(e):
                    satisfiability = RECURSION_DEPTH_EXCEEDED
                    # print(RECURSION_DEPTH_EXCEEDED)
                else:
                    satisfiability = RECURSION_ERROR
                    # print(RECURSION_ERROR)
        # notice that the name "Total explore_paths call" is for summary script to parse
        summary_dict = {"Total explore_paths call": self.total_split_eq_call, "total_rank_call": self.total_rank_call,
                        "total_gnn_call": self.total_gnn_call, "total_category_call": self.total_category_call,
                        "predicted_data_hash_table_hit": self.predicted_data_hash_table_hit,
                        "dgl_hash_table_hit": self.dgl_hash_table_hit, "prefix_rules_count": self.prefix_rules_count,
                        "suffix_rules_count": self.suffix_rules_count}
        run_summary(summary_dict)

        return {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                "variables": self.variables, "terminals": self.terminals}

    def split_eq(self, input_formula: Formula, current_depth: int, previous_node: Tuple[int, Dict],
                 edge_label: str) -> Tuple[str, Formula]:
        self.total_split_eq_call += 1

        current_node = self.record_node_and_edges(input_formula, previous_node, edge_label, current_depth)

        if self.total_split_eq_call % 10000 == 0:
            print(f"----- total_split_eq_call:{self.total_split_eq_call}, current_depth:{current_depth} -----")

        # early termination condition
        res = self.check_termination_condition_func(current_depth)
        if res != None:
            current_node[1]["status"] = res
            current_node[1]["back_track_count"] = 1
            return (res, input_formula)

        satisfiability, current_formula = simplify_and_check_formula(input_formula)

        if satisfiability != UNKNOWN:
            current_node[1]["status"] = satisfiability
            current_node[1]["back_track_count"] = 1
            return satisfiability, current_formula
        else:
            current_formula = self.order_equations_func_wrapper(current_formula, current_node)

            current_eq, separated_formula = self.get_first_eq(current_formula)

            current_eq_node = self.record_eq_node_and_edges(current_eq, previous_node=current_node,
                                                            edge_label=f"eq:{0}: {current_eq.eq_str}")

            self.decide_rules_func(current_eq)
            self._count_rule_type()

            children, fresh_variable_counter = self.apply_rules(current_eq, separated_formula,
                                                                self.fresh_variable_counter)
            self.fresh_variable_counter = fresh_variable_counter
            children: List[Tuple[Equation, Formula, str]] = self.order_branches_func(children)

            unknown_flag = 0
            for c_index, child in enumerate(children):
                (c_eq, c_formula, edge_label) = child
                satisfiability, res_formula = self.split_eq(c_formula, current_depth + 1, current_eq_node,
                                                            edge_label)
                if satisfiability == SAT:
                    current_node[1]["status"] = SAT
                    current_eq_node[1]["status"] = SAT
                    return (SAT, res_formula)
                elif satisfiability == UNKNOWN:
                    unknown_flag = 1

            if unknown_flag == 1:
                current_node[1]["status"] = UNKNOWN
                current_eq_node[1]["status"] = UNKNOWN
                return (UNKNOWN, current_formula)
            else:
                current_node[1]["status"] = UNSAT
                current_eq_node[1]["status"] = UNSAT
                return (UNSAT, current_formula)

    def _decide_rules_by_quadratic_pattern(self, current_eq: Equation):
        # when the term to be replace is quadratic, shift
        if current_eq.left_hand_side_length != 0 and current_eq.right_hand_side_length != 0:
            if self.prefix_rules == True:
                first_left_term = current_eq.left_terms[0]
                first_right_term = current_eq.right_terms[0]
                first_left_term_occurrence_in_left_terms = current_eq.left_terms.count(first_left_term)
                first_left_term_occurrence_in_right_terms = current_eq.right_terms.count(first_left_term)
                first_right_term_occurrence_in_left_terms = current_eq.left_terms.count(first_right_term)
                first_right_term_occurrence_in_right_terms = current_eq.right_terms.count(first_right_term)

                if first_left_term.value_type == Variable and first_right_term.value_type == Terminal and first_left_term_occurrence_in_left_terms <= first_left_term_occurrence_in_right_terms:
                    self.prefix_rules = False
                    self.apply_rules = apply_rules_suffix

                elif first_left_term.value_type == Terminal and first_right_term.value_type == Variable and first_right_term_occurrence_in_right_terms <= first_right_term_occurrence_in_left_terms:
                    self.prefix_rules = False
                    self.apply_rules = apply_rules_suffix

                elif first_left_term.value_type == Variable and first_right_term.value_type == Variable and first_left_term != first_right_term and (
                        first_left_term_occurrence_in_left_terms <= first_left_term_occurrence_in_right_terms or first_right_term_occurrence_in_right_terms <= first_right_term_occurrence_in_left_terms):
                    self.prefix_rules = False
                    self.apply_rules = apply_rules_suffix

                else:
                    pass

            else:
                last_left_term = current_eq.left_terms[-1]
                last_right_term = current_eq.right_terms[-1]
                last_left_term_occurrence_in_left_terms = current_eq.left_terms.count(last_left_term)
                last_left_term_occurrence_in_right_terms = current_eq.right_terms.count(last_left_term)
                last_right_term_occurrence_in_left_terms = current_eq.left_terms.count(last_right_term)
                last_right_term_occurrence_in_right_terms = current_eq.right_terms.count(last_right_term)

                if last_left_term.value_type == Variable and last_right_term.value_type == Terminal and last_left_term_occurrence_in_left_terms <= last_left_term_occurrence_in_right_terms:
                    self.prefix_rules = True
                    self.apply_rules = apply_rules_prefix

                elif last_left_term.value_type == Terminal and last_right_term.value_type == Variable and last_right_term_occurrence_in_right_terms <= last_right_term_occurrence_in_left_terms:
                    self.prefix_rules = True
                    self.apply_rules = apply_rules_prefix

                elif last_left_term.value_type == Variable and last_right_term.value_type == Variable and last_left_term != last_right_term and (
                        last_left_term_occurrence_in_left_terms <= last_left_term_occurrence_in_right_terms or last_right_term_occurrence_in_right_terms <= last_right_term_occurrence_in_left_terms):
                    self.prefix_rules = True
                    self.apply_rules = apply_rules_prefix

                else:
                    pass

    def _decide_rules_by_frequency(self, current_eq):
        if self.total_split_eq_call % self.prefix_suffix_change_frequency == 0:
            self.prefix_rules = not self.prefix_rules
            self.apply_rules = apply_rules_prefix if self.prefix_rules == True else apply_rules_suffix

        # self._count_rule_type()

    def _decide_rules_by_probability(self, current_eq):
        probability = random.random()
        self.apply_rules = apply_rules_prefix if probability < 0.5 else apply_rules_suffix

        # self._count_rule_type()

    def _decide_rules_prefix(self, current_eq):
        self.apply_rules = apply_rules_prefix

    def _decide_rules_suffix(self, current_eq):
        self.apply_rules = apply_rules_suffix

    def _count_rule_type(self):
        if self.apply_rules == apply_rules_prefix:
            self.prefix_rules_count += 1
        else:
            self.suffix_rules_count += 1

    def _order_equations_hybrid_category_gnn_random(self, f: Formula, category_call=0) -> (Formula, int):
        probability = random.random()
        if probability < HYBRID_ORDER_EQUATION_RATE:
            return self._order_equations_category_gnn(f, category_call)
        else:
            return order_equations_category_random(f, category_call)

    def _order_equations_hybrid_category_gnn_random_first_n_iterations(self, f: Formula, category_call=0) -> (
            Formula, int):
        if self.total_gnn_call < self.first_n_itarations:
            return self._order_equations_category_gnn(f, category_call)
        else:
            return order_equations_category_random(f, category_call)

    def _order_equations_hybrid_category_gnn_random_each_n_iterations(self, f: Formula, category_call=0) -> (
            Formula, int):
        if self.total_rank_call % self.each_n_iterations == 0:
            return self._order_equations_category_gnn(f, category_call)
        else:
            return order_equations_category_random(f, category_call)

    def _order_equations_hybrid_category_gnn_random_formula_size(self, f: Formula, category_call=0) -> (
            Formula, int):
        condition = self._compute_condition_for_changed_eq_size(f)
        if condition:
            return self._order_equations_category_gnn(f, category_call)
        else:
            return order_equations_category_random(f, category_call)

    def _order_equations_category_gnn(self, f: Formula, category_call=0) -> (Formula, int):
        return self._order_equations_category_gnn_with_conditions(f, True, category_call)

    def _order_equations_category_gnn_formula_size(self, f: Formula, category_call=0) -> (Formula, int):
        condition = self._compute_condition_for_changed_eq_size(f)
        return self._order_equations_category_gnn_with_conditions(f, condition, category_call)

    def _order_equations_category_gnn_first_n_iterations(self, f: Formula, category_call=0) -> (Formula, int):
        condition = self.total_gnn_call < self.first_n_itarations
        return self._order_equations_category_gnn_with_conditions(f, condition, category_call)

    def _order_equations_category_gnn_each_n_iterations(self, f: Formula, category_call=0) -> (Formula, int):
        condition = self.total_rank_call % self.each_n_iterations == 0
        return self._order_equations_category_gnn_with_conditions(f, condition, category_call)

    def _order_equations_category_gnn_with_conditions(self, f: Formula, condition: bool, category_call=0) -> (
            Formula, int):
        categoried_eq_list: List[Tuple[Equation, int]] = _category_formula_by_rules(f)

        # Check if the equation categories are only 5 and 6
        only_5_and_6: bool = all(n in [5, 6] for _, n in categoried_eq_list)

        if only_5_and_6 == True and len(categoried_eq_list) > 1:
            if condition:
                ordered_formula, category_call = self._order_equations_gnn(f, category_call)
                sorted_eq_list = ordered_formula.eq_list
            else:
                category_call += 1
                sorted_eq_list = [eq for eq, _ in sorted(categoried_eq_list, key=lambda x: x[1])]
        else:
            category_call += 1
            sorted_eq_list = [eq for eq, _ in sorted(categoried_eq_list, key=lambda x: x[1])]

        return Formula(sorted_eq_list), category_call

    def _order_equations_hybrid_gnn_random_first_n_iterations(self, f: Formula, category_call=0) -> (Formula, int):
        condition = self.total_gnn_call < self.first_n_itarations
        return self._order_equations_gnn_with_condition(f, condition, order_equations_random, category_call)

    def _order_equations_hybrid_gnn_random_each_n_iterations(self, f: Formula, category_call=0) -> (Formula, int):
        condition = self.total_rank_call % self.each_n_iterations == 0
        return self._order_equations_gnn_with_condition(f, condition, order_equations_random, category_call)

    def _order_equations_hybrid_gnn_random_formula_size(self, f: Formula, category_call=0) -> (Formula, int):
        probability = random.random()
        condition = probability < HYBRID_ORDER_EQUATION_RATE and self._compute_condition_for_changed_eq_size(f)
        return self._order_equations_gnn_with_condition(f, condition, order_equations_random, category_call)

    def _order_equations_hybrid_gnn_random(self, f: Formula, category_call=0) -> (Formula, int):
        probability = random.random()
        condition = probability < HYBRID_ORDER_EQUATION_RATE
        return self._order_equations_gnn_with_condition(f, condition, order_equations_random, category_call)

    def _order_equations_gnn_first_n_iterations(self, f: Formula, category_call=0) -> (Formula, int):
        condition = self.total_gnn_call < self.first_n_itarations
        return self._order_equations_gnn_with_condition(f, condition, self.empty_else_func, category_call)

    def _order_equations_gnn_each_n_iterations(self, f: Formula, category_call=0) -> (Formula, int):
        condition = self.total_rank_call % self.each_n_iterations == 0
        return self._order_equations_gnn_with_condition(f, condition, self.empty_else_func, category_call)

    def _order_equations_gnn_formula_size(self, f: Formula, category_call=0) -> (Formula, int):
        # if each self.dynamic_condition_check_point_frequency call that no reduction of eq length, then call gnn
        condition = self._compute_condition_for_changed_eq_size(f)
        return self._order_equations_gnn_with_condition(f, condition, self.empty_else_func, category_call)

    def _order_equations_gnn_with_condition(self, f: Formula, condition: bool, else_func, category_call=0) -> (
            Formula, int):
        if condition:
            return self._order_equations_gnn(f, category_call)
        else:
            return else_func(f, category_call)

    def empty_else_func(self, f: Formula, category_call=0) -> (Formula, int):
        return f, category_call

    def _compute_condition_for_changed_eq_size(self, f: Formula):
        condition = False
        if self.total_split_eq_call % self.dynamic_condition_check_point_frequency == 0:
            self.changed_eq_size = self.current_eq_size - f.formula_size  # positive integer mean the eq length is reduced
            self.current_eq_size = f.formula_size
            if self.changed_eq_size <= 0:
                condition = True
        return condition

    @time_it
    def _get_G_list_dgl(self, f: Formula):
        gc.disable()
        global_info = _get_global_info(f.eq_list)
        G_list_dgl = []

        # Local references to the hash table and counter for efficiency
        dgl_hash_table = self.dgl_hash_table
        dgl_hash_table_hit = self.dgl_hash_table_hit

        for index, eq in enumerate(f.eq_list):

            split_eq_nodes, split_eq_edges = self.graph_func(eq.left_terms, eq.right_terms, global_info)

            # _construct_graph_for_prediction_start = time.time()
            # _construct_graph_for_prediction(eq.left_terms, eq.right_terms, global_info)
            # _construct_graph_for_prediction_end = time.time() - _construct_graph_for_prediction_start
            # if _construct_graph_for_prediction_end > 0.1:
            #     print("_construct_graph_for_prediction time", _construct_graph_for_prediction_end, "eq length", eq.term_length)
            #     _construct_graph_for_prediction_start_again = time.time()
            #     _construct_graph_for_prediction(eq.left_terms, eq.right_terms, global_info)
            #     print("_construct_graph_for_prediction time again", time.time() - _construct_graph_for_prediction_start_again)

            # hash eq+global info to dgl
            hashed_eq, _ = hash_graph_with_glob_info(split_eq_nodes, split_eq_edges)
            if hashed_eq in dgl_hash_table:
                dgl_graph = dgl_hash_table[hashed_eq]
                dgl_hash_table_hit += 1
            else:
                graph_dict = graph_to_gnn_format(split_eq_nodes, split_eq_edges)
                dgl_graph, _ = get_one_dgl_graph(graph_dict)
                dgl_hash_table[hashed_eq] = dgl_graph

            G_list_dgl.append(dgl_graph)

            self.visualize_gnn_input_func(nodes=split_eq_nodes, edges=split_eq_edges,filename=self.file_name + f"_rank_call_{self.total_rank_call}_{index}")

        # Update the hit count back to the global variable
        self.dgl_hash_table_hit = dgl_hash_table_hit
        gc.enable()
        return G_list_dgl


    def _get_rank_list(self, G_list_dgl):
        with no_grad():

            # use different module of gnn_rank_model
            # embedding output [n,1,128]
            G_list_embeddings = self.gnn_rank_model.shared_gnn.embedding(batch(G_list_dgl))

            # concat target output [n,1,256]
            mean_tensor = mean(G_list_embeddings, dim=0)  # [1,128]
            input_eq_embeddings_list = []
            for g in G_list_embeddings:
                # input_eq_embeddings_list.append(concat([g, mean_tensor], dim=1))
                input_eq_embeddings_list.append(concat([g, mean_tensor]))  # For multi filters
            input_eq_embeddings_list = stack(input_eq_embeddings_list)

            # classifier
            classifier_output = self.gnn_rank_model.classifier(input_eq_embeddings_list)  # [n,2]

            # transform [x,y] to one score
            rank_list = []
            for rank in classifier_output.tolist():  # rank [2]
                rank_softmax = softmax(rank)
                rank_list.append(rank_softmax[0])

        return rank_list

    # @time_it
    # def _get_rank_list(self, G_list_dgl):
    #     from torch import softmax
    #     with no_grad():
    #         # Compute embeddings for the batch of graphs
    #         G_list_embeddings = self.gnn_rank_model.shared_gnn.embedding(batch(G_list_dgl))  # [n, 1, 128]
    #
    #         # Compute the mean tensor across embeddings
    #         mean_tensor = mean(G_list_embeddings, dim=0, keepdim=True)  # [1, 128]
    #
    #         input_eq_embeddings_list = []
    #         for g in G_list_embeddings:
    #             # input_eq_embeddings_list.append(concat([g, mean_tensor], dim=1))
    #             input_eq_embeddings_list.append(concat([g, mean_tensor]))  # For multi filters
    #         input_eq_embeddings_list = stack(input_eq_embeddings_list)
    #
    #         # Classifier output
    #         classifier_output = self.gnn_rank_model.classifier(input_eq_embeddings_list)  # [n, 2]
    #
    #         # Apply softmax and extract the first column as the score
    #         rank_list = softmax(classifier_output, dim=-1)[:, :, 0].squeeze().tolist()
    #
    #     return rank_list

    def _sort_eq_list_by_prediction(self, rank_list, eq_list):
        prediction_list = []
        for pred, split_eq in zip(rank_list, eq_list):
            prediction_list.append([pred, split_eq])

        sorted_prediction_list = sorted(prediction_list, key=lambda x: x[0], reverse=True)

        # print("-" * 10)
        # print("before sort")
        # for rank,eq in zip(rank_list,eq_list):
        #     print(rank,eq.eq_str)
        # print("after sort")
        # for x in sorted_prediction_list:
        #     print(x[0],x[1].eq_str)

        formula_with_sorted_eq_list = Formula([x[1] for x in sorted_prediction_list])
        return formula_with_sorted_eq_list

    def _order_equations_gnn_rank_task_0(self, f: Formula, category_call=0) -> (Formula, int):
        self.total_gnn_call += 1
        self.gnn_call_flag = True

        # form input graphs
        G_list_dgl = self._get_G_list_dgl(f)

        # predict
        with no_grad():
            classifier_output = self.gnn_rank_model(batch(G_list_dgl)).squeeze()

        rank_list: List[float] = []
        for one_output in classifier_output:
            softmax_one_output = softmax(one_output.tolist())
            rank_list.append(softmax_one_output[0])

        # sort
        formula_with_sorted_eq_list = self._sort_eq_list_by_prediction(rank_list, f.eq_list)

        return formula_with_sorted_eq_list, category_call


    def _order_equations_gnn_rank_task_1(self, f: Formula, category_call=0) -> (Formula, int):
        self.total_gnn_call += 1
        self.gnn_call_flag = True

        # form input graphs
        G_list_dgl = self._get_G_list_dgl(f)

        # predict
        rank_list = self._get_rank_list(G_list_dgl)

        # sort
        formula_with_sorted_eq_list = self._sort_eq_list_by_prediction(rank_list, f.eq_list)

        return formula_with_sorted_eq_list, category_call

    def _order_equations_gnn_rank_task_2(self, f: Formula, category_call=0) -> (Formula, int):
        self.total_gnn_call += 1
        self.gnn_call_flag = True

        # form input graphs
        G_list_dgl = self._get_G_list_dgl(f)

        # predict
        # predict_time_start = time.time()
        if len(G_list_dgl) < self.parameters["label_size"]:  # pad list
            while len(G_list_dgl) < self.parameters["label_size"]:
                G_list_dgl.append(self.empty_dgl_graph)
        elif len(G_list_dgl) > self.parameters["label_size"]:  # trim list
            G_list_dgl = G_list_dgl[:self.parameters["label_size"]]
        else:
            pass

        with no_grad():
            classifier_output = self.gnn_rank_model(G_list_dgl).squeeze()
        rank_list = classifier_output.tolist()
        # print("predict time:", time.time() - predict_time_start)

        # sort
        formula_with_sorted_eq_list = self._sort_eq_list_by_prediction(rank_list, f.eq_list)

        return formula_with_sorted_eq_list, category_call

    def _order_branches_gnn(self, children: List[Tuple[Equation, Formula]]) -> List[Tuple[Equation, Formula]]:
        # todo implement gnn
        return children

    def early_termination_condition_0(self, current_depth: int):
        if current_depth > self.termination_condition_max_depth:
            return UNKNOWN

    def early_termination_condition_1(self, current_depth: int):
        if current_depth > self.restart_max_deep or current_depth > self.termination_condition_max_depth:
            return UNKNOWN

    def visualize(self, file_path: str, graph_func: Callable):
        visualize_path_html(self.nodes, self.edges, file_path)
        visualize_path_png(self.nodes, self.edges, file_path, compress=compress_image, edge_label=self.png_edge_label)
