import random
import time
from typing import List, Dict, Tuple, Callable

import dgl
from dgl.dataloading import GraphDataLoader

from src.solver.Constants import recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR, UNSAT, SAT, UNKNOWN, RESTART_INITIAL_MAX_DEEP, \
    RESTART_MAX_DEEP_STEP, compress_image, HYBRID_ORDER_EQUATION_RATE, RANDOM_SEED
from src.solver.DataTypes import Assignment, Terminal, Variable, Equation, Formula, _construct_graph_for_prediction
from . import graph_to_gnn_format
from .utils import sigmoid,softmax
from ..independent_utils import log_control, strip_file_name_suffix, color_print, time_it, hash_one_data, \
    hash_graph_with_glob_info, hash_one_dgl_data
from src.solver.visualize_util import visualize_path_html, visualize_path_png, draw_graph
from .abstract_algorithm import AbstractAlgorithm
import sys
from src.solver.algorithms.split_equation_utils import _category_formula_by_rules, \
    apply_rules, simplify_and_check_formula, order_equations_fixed, order_equations_random, order_equations_category, \
    order_equations_category_random, run_summary, _get_global_info, order_equations_hybrid_fixed_random, \
    order_equations_hybrid_category_fixed_random, order_branches_fixed, order_branches_random, \
    order_branches_hybrid_fixed_random
from src.solver.models.utils import load_model, load_model_torch_script, load_model_onnx
from ..models.Dataset import get_one_dgl_graph
import torch
import gc

from ..models.train_util import custom_collate_fn


class SplitEquations(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation],
                 parameters: Dict):
        super().__init__(terminals, variables, equation_list)
        self.assignment = Assignment()
        self.parameters = parameters
        self.nodes = []
        self.edges = []
        self.visualize_gnn_input = False

        self.file_name = strip_file_name_suffix(parameters["file_path"])
        self.fresh_variable_counter = 0
        self.total_gnn_call = 0
        self.total_category_call = 0
        self.total_split_eq_call = 0
        self.total_rank_call = 0
        self.total_node_number = 1
        self.eq_node_number = 0
        self.termination_condition_max_depth = 20000
        self.restart_max_deep = RESTART_INITIAL_MAX_DEEP
        self.each_n_iterations = 10
        self.first_n_itarations = 1

        self.order_equations_func_map = {"fixed": order_equations_fixed,
                                         "random": order_equations_random,
                                         "hybrid_fixed_random": order_equations_hybrid_fixed_random,
                                         "category": order_equations_category,
                                         "category_random": order_equations_category_random,
                                         "hybrid_category_fixed_random": order_equations_hybrid_category_fixed_random,
                                         "category_gnn": self._order_equations_category_gnn,  # first category then gnn
                                         "category_gnn_each_n_iterations": self._order_equations_category_gnn_each_n_iterations,
                                         "category_gnn_first_n_iterations":self._order_equations_category_gnn_first_n_iterations,
                                         "hybrid_category_gnn_random": self._order_equations_hybrid_category_gnn_random,
                                         "hybrid_category_gnn_random_each_n_iterations": self._order_equations_hybrid_category_gnn_random_each_n_iterations,
                                         "hybrid_category_gnn_random_first_n_iterations": self._order_equations_hybrid_category_gnn_random_first_n_iterations,
                                         #"gnn": self._order_equations_gnn,
                                         "gnn":self._order_equations_gnn_rank_task_0,
                                         "gnn_each_n_iterations": self._order_equations_gnn_each_n_iterations,
                                         "gnn_first_n_iterations": self._order_equations_gnn_first_n_iterations,
                                         "hybrid_gnn_random": self._order_equations_hybrid_gnn_random,
                                         "hybrid_gnn_random_each_n_iterations": self._order_equations_hybrid_gnn_random_each_n_iterations,
                                         "hybrid_gnn_random_first_n_iterations": self._order_equations_hybrid_gnn_random_first_n_iterations
                                         }
        self.order_equations_func: Callable = self.order_equations_func_map[self.parameters["order_equations_method"]]
        # load model if call gnn
        if "gnn" in self.parameters["order_equations_method"]:
            gnn_model_path = parameters["gnn_model_path"].replace("_0_", "_2_")
            self.gnn_rank_model = load_model(gnn_model_path)  # this is a GraphClassifier class
            # self.gnn_rank_model = load_model_torch_script(gnn_model_path)
            # self.gnn_rank_model = load_model_onnx(gnn_model_path)
            self.gnn_rank_model.is_test = True

            self.graph_func = parameters["graph_func"]

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
        sys.setrecursionlimit(recursion_limit)
        print("recursion limit number", sys.getrecursionlimit())

        print("order_equations_method:", self.parameters["order_equations_method"])

    @log_control
    def run(self):
        random.seed(RANDOM_SEED)
        original_formula = Formula(self.equation_list)

        if self.parameters["termination_condition"] == "termination_condition_1":
            while True:
                initial_node: Tuple[int, Dict] = (
                    0, {"label": "start", "status": None, "output_to_file": False, "shape": "ellipse",
                        "back_track_count": 0, "gnn_call": False})
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
                    "back_track_count": 0, "gnn_call": False})
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
                        "dgl_hash_table_hit": self.dgl_hash_table_hit}
        run_summary(summary_dict)

        return {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                "variables": self.variables, "terminals": self.terminals}

    def split_eq(self, input_formula: Formula, current_depth: int, previous_node: Tuple[int, Dict],
                 edge_label: str) -> Tuple[str, Formula]:
        self.total_split_eq_call += 1

        current_node = self.record_node_and_edges(input_formula, previous_node, edge_label)

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

            children, fresh_variable_counter = apply_rules(current_eq, separated_formula, self.fresh_variable_counter)
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

    def get_first_eq(self, f: Formula) -> Tuple[Equation, Formula]:
        return f.eq_list[0], Formula(f.eq_list[1:])

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

    def _order_equations_category_gnn(self, f: Formula, category_call=0) -> (Formula, int):
        categoried_eq_list: List[Tuple[Equation, int]] = _category_formula_by_rules(f)

        # Check if the equation categories are only 5 and 6
        only_5_and_6: bool = all(n in [5, 6] for _, n in categoried_eq_list)

        if only_5_and_6 == True and len(categoried_eq_list) > 1:
            ordered_formula, category_call = self._order_equations_gnn(f, category_call)
            sorted_eq_list = ordered_formula.eq_list
        else:
            category_call += 1
            sorted_eq_list = [eq for eq, _ in sorted(categoried_eq_list, key=lambda x: x[1])]

        return Formula(sorted_eq_list), category_call

    def _order_equations_category_gnn_first_n_iterations(self, f: Formula, category_call=0) -> (Formula, int):
        categoried_eq_list: List[Tuple[Equation, int]] = _category_formula_by_rules(f)

        # Check if the equation categories are only 5 and 6
        only_5_and_6: bool = all(n in [5, 6] for _, n in categoried_eq_list)

        if only_5_and_6 == True and len(categoried_eq_list) > 1:
            if self.total_gnn_call < self.first_n_itarations:
                ordered_formula, category_call = self._order_equations_gnn(f, category_call)
                sorted_eq_list = ordered_formula.eq_list
            else:
                category_call += 1
                sorted_eq_list = [eq for eq, _ in sorted(categoried_eq_list, key=lambda x: x[1])]
        else:
            category_call += 1
            sorted_eq_list = [eq for eq, _ in sorted(categoried_eq_list, key=lambda x: x[1])]

        return Formula(sorted_eq_list), category_call

    def _order_equations_category_gnn_each_n_iterations(self, f: Formula, category_call=0) -> (Formula, int):
        categoried_eq_list: List[Tuple[Equation, int]] = _category_formula_by_rules(f)

        # Check if the equation categories are only 5 and 6
        only_5_and_6: bool = all(n in [5, 6] for _, n in categoried_eq_list)

        if only_5_and_6 == True and len(categoried_eq_list) > 1:
            if self.total_rank_call % self.each_n_iterations == 0:
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

        if self.total_gnn_call < self.first_n_itarations:
            return self._order_equations_gnn(f, category_call)
        else:
            return order_equations_random(f, category_call)
    def _order_equations_hybrid_gnn_random_each_n_iterations(self, f: Formula, category_call=0) -> (Formula, int):

        if self.total_rank_call % self.each_n_iterations == 0:
            return self._order_equations_gnn(f, category_call)
        else:
            return order_equations_random(f, category_call)

    def _order_equations_hybrid_gnn_random(self, f: Formula, category_call=0) -> (Formula, int):
        probability = random.random()
        if probability < HYBRID_ORDER_EQUATION_RATE:
            return self._order_equations_gnn(f, category_call)
        else:
            return order_equations_random(f, category_call)

    def _order_equations_gnn_first_n_iterations(self, f: Formula, category_call=0) -> (Formula, int):
        if self.total_gnn_call < self.first_n_itarations:
            return self._order_equations_gnn(f, category_call)
        else:
            return f, category_call
    def _order_equations_gnn_each_n_iterations(self, f: Formula, category_call=0) -> (Formula, int):
        if self.total_rank_call % self.each_n_iterations == 0:
            return self._order_equations_gnn(f, category_call)
        else:
            return f, category_call



    def _get_G_list_dgl(self, f: Formula):
        gc.disable()
        global_info = _get_global_info(f.eq_list)
        G_list_dgl = []

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
            if hashed_eq in self.dgl_hash_table:
                dgl_graph = self.dgl_hash_table[hashed_eq]
                self.dgl_hash_table_hit += 1
            else:
                graph_dict = graph_to_gnn_format(split_eq_nodes, split_eq_edges)
                dgl_graph, _ = get_one_dgl_graph(graph_dict)
                self.dgl_hash_table[hashed_eq] = dgl_graph

            G_list_dgl.append(dgl_graph)
            if self.visualize_gnn_input == True:
                draw_graph(nodes=split_eq_nodes, edges=split_eq_edges,
                           filename=self.file_name + f"_rank_call_{self.total_rank_call}_{index}")

        gc.enable()
        return G_list_dgl


    def _get_rank_list(self, G_list_dgl):
        with torch.no_grad():

            # use different module of gnn_rank_model
            # embedding output [n,1,128]
            G_list_embeddings = self.gnn_rank_model.shared_gnn.embedding(dgl.batch(G_list_dgl))


            # concat target output [n,1,256]
            mean_tensor = torch.mean(G_list_embeddings, dim=0)  # [1,128]
            input_eq_embeddings_list = []
            for g in G_list_embeddings:
                input_eq_embeddings_list.append(torch.concat([g, mean_tensor], dim=1))
            input_eq_embeddings_list = torch.stack(input_eq_embeddings_list)


            # classifier
            classifier_output = self.gnn_rank_model.classifier(input_eq_embeddings_list)  # [n,1,2]

            # transform [x,y] to one score
            classifier_output_list = [item[0] for item in classifier_output.tolist()] #[n,2]
            rank_list = []
            for rank in classifier_output_list: #rank [2]
                rank_softmax=softmax(rank)
                rank_list.append(rank_softmax[0])

        return rank_list


    def _sort_eq_list_by_prediction(self, rank_list, eq_list):
        prediction_list = []
        for pred, split_eq in zip(rank_list, eq_list):
            prediction_list.append([pred, split_eq])

        sorted_prediction_list = sorted(prediction_list, key=lambda x: x[0], reverse=True)

        # print("rank_list", rank_list)
        # for x in sorted_prediction_list:
        #     print(x[0],x[1].eq_str)

        formula_with_sorted_eq_list = Formula([x[1] for x in sorted_prediction_list])
        return formula_with_sorted_eq_list

    def _order_equations_gnn_rank_task_0(self, f: Formula, category_call=0) -> (Formula, int):
        self.total_gnn_call += 1
        self.gnn_call_flag = True

        # form input graphs
        G_list_dgl= self._get_G_list_dgl(f)

        start = time.time()
        # predict
        with torch.no_grad():
            classifier_output = self.gnn_rank_model(dgl.batch(G_list_dgl)).squeeze()

        rank_list=[]
        for one_output in classifier_output:
            softmax_one_output=softmax(one_output.tolist())
            rank_list.append(softmax_one_output[0])



        end = time.time() - start
        print("predict time", end)

        # sort
        formula_with_sorted_eq_list=self._sort_eq_list_by_prediction(rank_list, f.eq_list)

        return formula_with_sorted_eq_list, category_call

    def _order_equations_gnn(self, f: Formula, category_call=0) -> (Formula, int):
        self.total_gnn_call += 1
        self.gnn_call_flag = True

        # form input graphs
        G_list_dgl = self._get_G_list_dgl(f)

        # predict
        rank_list = self._get_rank_list(G_list_dgl)

        # sort
        formula_with_sorted_eq_list=self._sort_eq_list_by_prediction(rank_list, f.eq_list)

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
