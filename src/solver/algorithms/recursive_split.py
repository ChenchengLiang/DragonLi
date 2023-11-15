import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED, recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR, SAT, UNSAT, UNKNOWN, project_folder, MAX_DEEP, MAX_SPLIT_CALL, \
    OUTPUT_LEAF_NODE_PERCENTAGE, GNN_BRANCH_RATIO,MAX_EQ_LENGTH
from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, EMPTY_TERMINAL
from src.solver.utils import assemble_parsed_content
from src.solver.independent_utils import remove_duplicates, flatten_list, strip_file_name_suffix, \
    dump_to_json_with_format, identify_available_capitals
from src.solver.visualize_util import visualize_path, visualize_path_html, visualize_path_png
from src.solver.algorithms.abstract_algorithm import AbstractAlgorithm
from src.solver.models.utils import load_model, load_model_from_mlflow
from src.solver.models.Dataset import WordEquationDataset
from dgl.dataloading import GraphDataLoader
import sys
import copy

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
        self.nodes = []
        self.edges = []
        self.branch_method_func_map = {"extract_branching_data_task_1": self._extract_branching_data_task_1,
                                       "extract_branching_data_task_2": self._extract_branching_data_task_2,
                                       "fixed": self._use_fixed_branching, "random": self._use_random_branching,
                                       "gnn": self._use_gnn_branching,
                                       "gnn:random": self._use_gnn_with_random_branching,
                                       "gnn:fixed": self._use_gnn_with_fixed_branching}
        self._branch_method = parameters["branch_method"]
        self._branch_method_func = self.branch_method_func_map[parameters["branch_method"]]
        self.record_and_close_branch = self._record_and_close_branch_with_file if "extract_branching_data" in parameters["branch_method"] else self._record_and_close_branch_without_file
        sys.setrecursionlimit(recursion_limit)
        # print("recursion limit number", sys.getrecursionlimit())

        if "gnn" in parameters["branch_method"]:
            # Load the model
            self.gnn_model = load_model(parameters["gnn_model_path"])
            # load the model from mlflow
            # experiment_id = "856005721390468951"
            # run_id = "feb2e17e68bb4310bb3c539c672fd166"
            # self.gnn_model = load_model_from_mlflow(experiment_id, run_id)
            self.graph_func = parameters["graph_func"]
        self.output_train_data = False if self.file_name == "" else True

    def run(self):
        print("branch_method:", self.parameters["branch_method"])
        first_equation = self.equation_list[0]

        try:
            node_info = (0, {"label": "start", "status": None, "output_to_file": False, "shape": "ellipse"})
            self.nodes.append(node_info)
            satisfiability, variables = self.explore_paths(first_equation, {"node_number": node_info[0],
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
                       "total_explore_paths_call": self.total_explore_paths_call}
        return result_dict

    def explore_paths(self, current_eq: Equation,
                      previous_dict) -> Tuple[str, List[Variable]]:
        self.total_explore_paths_call += 1

        ################################ Record nodes and edges ################################

        current_node_number = self.total_explore_paths_call
        node_info = (
        current_node_number, {"label": current_eq.eq_str, "status": None, "output_to_file": False, "shape": "ellipse"})
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
            satisfiability = SAT if self.check_equation(current_eq.left_terms,
                                                        current_eq.right_terms) == True else UNSAT
            return self.record_and_close_branch(satisfiability, current_eq.variable_list, node_info, current_eq)

        ## both side only have variables
        if len(current_eq.termimal_list_without_empty_terminal) == 0:
            return self.record_and_close_branch(SAT, current_eq.variable_list, node_info, current_eq)

        ## special cases
        ### special case: one side only have terminals and another side have longer terminals.
        ### special case: variables surrounded by identical terminals
        ### special case: variables surrounded by different terminals
        ### special case: starting or ending with variables

        ### special case 1: mismatched leading or tailing terminals
        left_leading_terminals, left_first_variable = self.get_leading_terminals(current_eq.left_terms)
        right_leading_terminals, right_first_variable = self.get_leading_terminals(current_eq.right_terms)
        if len(left_leading_terminals) > 0 and len(right_leading_terminals) > 0 and len(left_leading_terminals) == len(
                right_leading_terminals) and left_first_variable != None and right_first_variable != None and left_leading_terminals != right_leading_terminals and left_first_variable == right_first_variable:
            return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)
        left_tailing_terminals, left_first_variable = self.get_leading_terminals(reversed(current_eq.left_terms))
        right_tailing_terminals, right_first_variable = self.get_leading_terminals(reversed(current_eq.right_terms))
        if len(left_tailing_terminals) > 0 and len(right_tailing_terminals) > 0 and len(left_tailing_terminals) == len(
                right_tailing_terminals) and left_first_variable != None and right_first_variable != None and left_tailing_terminals != right_tailing_terminals and left_first_variable == right_first_variable:
            return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)

        ### special case 2: one side only have one variable, e,g. M = terminals+variables SAT, M = terminals SAT, M = variables SAT, M="" SAT
        if (len(current_eq.left_terms) == 1 and current_eq.left_terms[0].value_type == Variable):
            if current_eq.left_terms[
                0] in current_eq.right_terms and current_eq.terminal_numers_without_empty_terminal != 0:  # M = terminals+variables and M in right hand side
                return self.record_and_close_branch(UNSAT, current_eq.variable_list, node_info, current_eq)
            else:
                return self.record_and_close_branch(SAT, current_eq.variable_list, node_info, current_eq)
        if (len(current_eq.right_terms) == 1 and current_eq.right_terms[0].value_type == Variable):
            if current_eq.right_terms[
                0] in current_eq.left_terms and current_eq.terminal_numers_without_empty_terminal != 0:
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

    def both_side_same_terms(self, eq: Equation, variables,
                             current_node_number, node_info):
        left_poped_list_str, right_poped_list_str, l, r = self.pop_both_same_terms(eq.left_terms, eq.right_terms,
                                                                                   eq.variable_list)
        new_eq = Equation(l, r)
        # updated_variables = self.update_variables(eq.left_terms, eq.right_terms)
        branch_satisfiability, branch_variables = self.explore_paths(new_eq,
                                                                     {"node_number": current_node_number,
                                                                      "label": left_poped_list_str + "=" + right_poped_list_str})
        return self.record_and_close_branch(branch_satisfiability, branch_variables, node_info, eq)

    def both_side_different_variables(self, eq: Equation,
                                      current_node_number, node_info):
        # Define the methods for each branch
        branch_methods = [
            self.two_variables_split_branch_1,
            self.two_variables_split_branch_2,
            self.two_variables_split_branch_3
        ]
        self.total_split_call += 1
        return self._branch_method_func(eq, current_node_number,
                                        node_info, branch_methods)

    def left_side_variable_right_side_terminal(self, eq: Equation,
                                               current_node_number, node_info):
        # Define the methods for each branch
        branch_methods = [
            self.one_variable_one_terminal_split_branch_1,
            self.one_variable_one_terminal_split_branch_2
        ]
        self.total_split_call += 1
        return self._branch_method_func(eq, current_node_number,
                                        node_info, branch_methods)

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

    def _use_gnn_branching(self, eq: Equation, current_node_number, node_info, branch_methods):
        # print(self.total_split_call,"gnn branch")
        # Compute branches and prepare data structures
        branches = []
        graph_list = []

        for method in branch_methods:
            l, r, v, edge_label = method(eq.left_terms, eq.right_terms, eq.variable_list)
            new_eq = Equation(l, r)
            nodes, edges = self.graph_func(new_eq.left_terms, new_eq.right_terms)
            graph_dict = new_eq.graph_to_gnn_format(nodes, edges)
            tuple_data = (new_eq, v, edge_label)

            branches.append(tuple_data)
            graph_list.append(graph_dict)

        # Load data
        evaluation_dataset = WordEquationDataset(graph_folder="", data_fold="eval", graphs_from_memory=graph_list)
        evaluation_dataloader = GraphDataLoader(evaluation_dataset, batch_size=1, drop_last=False)

        # Call gnn to predict
        prediction_list = []

        for (batched_graph, labels), eq_tuple in zip(evaluation_dataloader, branches):
            pred = self.gnn_model(batched_graph, batched_graph.ndata["feat"].float())  # pred is a float between 0 and 1
            prediction_list.append([pred, eq_tuple])

        sorted_prediction_list = sorted(prediction_list, key=lambda x: x[0], reverse=True)

        # Perform depth-first search based on the sorted prediction list
        for i, data in enumerate(sorted_prediction_list):
            eq, v, edge_label = data[1]
            satisfiability, variables = self.explore_paths(eq,
                                                           {"node_number": current_node_number, "label": edge_label})

            # Handle branch outcome
            if i < len(sorted_prediction_list) - 1:  # not the last branch
                if satisfiability == SAT:
                    return self.record_and_close_branch(SAT, variables, node_info, eq)
                elif satisfiability == UNSAT:
                    node_info[1]["status"] = UNSAT
            else:  # last branch
                return self.record_and_close_branch(satisfiability, variables, node_info, eq)

    def _use_random_branching(self, eq: Equation, current_node_number, node_info,
                              branch_methods):
        # print(self.total_split_call,"random branch")
        random.shuffle(branch_methods)
        return self._use_fixed_branching(eq, current_node_number, node_info,
                                         branch_methods)

    def _use_fixed_branching(self, eq: Equation, current_node_number, node_info,
                             branch_methods):
        # print(self.total_split_call,"fixed branch")
        for i, branch in enumerate(branch_methods):
            l, r, v, edge_label = branch(eq.left_terms, eq.right_terms, eq.variable_list)

            satisfiability, branch_variables = self.explore_paths(Equation(l, r),
                                                                  {"node_number": current_node_number,
                                                                   "label": edge_label})

            # Handle branch outcome
            if i < len(branch_methods) - 1:  # not last branch
                if satisfiability == SAT:
                    return self.record_and_close_branch(SAT, branch_variables, node_info, eq)
                elif satisfiability == UNSAT:
                    node_info[1]["status"] = UNSAT
            else:  # last branch
                return self.record_and_close_branch(satisfiability, branch_variables, node_info, eq)

    def _extract_branching_data_task_2(self, eq: Equation, current_node_number, node_info, branch_methods):
        ################################ branching ################################
        satisfiability_list = []
        branching_eq_list = []
        for i, branch in enumerate(branch_methods):
            l, r, v, edge_label = branch(eq.left_terms, eq.right_terms, eq.variable_list)
            branch_eq = Equation(l, r)
            branching_eq_list.append(branch_eq)
            satisfiability, branch_variables = self.explore_paths(branch_eq, {"node_number": current_node_number,
                                                                              "label": edge_label})
            satisfiability_list.append(satisfiability)

    def _extract_branching_data_task_1(self, eq: Equation, current_node_number, node_info, branch_methods):

        ################################ stop branching condition ################################
        if eq.term_length>MAX_EQ_LENGTH:
            return self.record_and_close_branch(UNKNOWN, eq.variable_list, node_info, eq)

        if self.total_split_call > MAX_SPLIT_CALL:
            return self.record_and_close_branch(UNKNOWN, eq.variable_list, node_info, eq)

        self.current_deep += 1
        if self.current_deep > MAX_DEEP:
            self.current_deep = 0
            return self.record_and_close_branch(UNKNOWN, eq.variable_list, node_info, eq)



        ################################ branching ################################
        satisfiability_list = []
        for i, branch in enumerate(branch_methods):
            l, r, v, edge_label = branch(eq.left_terms, eq.right_terms, eq.variable_list)

            satisfiability, branch_variables = self.explore_paths(Equation(l, r),
                                                                  {"node_number": current_node_number,
                                                                   "label": edge_label})
            satisfiability_list.append(satisfiability)

        ################################ gnn branching ################################

        # branches = []
        # graph_list = []
        #
        # for method in branch_methods:
        #     l, r, v, edge_label = method(eq.left_terms, eq.right_terms, eq.variable_list)
        #     new_eq = Equation(l, r)
        #     nodes, edges = self.graph_func(new_eq.left_terms, new_eq.right_terms)
        #     graph_dict = new_eq.graph_to_gnn_format(nodes, edges)
        #     tuple_data = (new_eq, v, edge_label)
        #
        #     branches.append(tuple_data)
        #     graph_list.append(graph_dict)
        #
        # # Load data
        # evaluation_dataset = WordEquationDataset(graph_folder="", data_fold="eval", graphs_from_memory=graph_list)
        # evaluation_dataloader = GraphDataLoader(evaluation_dataset, batch_size=1, drop_last=False)
        #
        # # Call gnn to predict
        # prediction_list = []
        #
        # for (batched_graph, labels), eq_tuple in zip(evaluation_dataloader, branches):
        #     pred = self.gnn_model(batched_graph, batched_graph.ndata["feat"].float())  # pred is a float between 0 and 1
        #     prediction_list.append([pred, eq_tuple])
        #
        # sorted_prediction_list = sorted(prediction_list, key=lambda x: x[0], reverse=True)
        #
        # # Perform depth-first search based on the sorted prediction list
        # satisfiability_list = []
        # for i, data in enumerate(sorted_prediction_list):
        #     eq, v, edge_label = data[1]
        #     satisfiability, branch_variables = self.explore_paths(eq,
        #                                                    {"node_number": current_node_number, "label": edge_label})
        #     satisfiability_list.append(satisfiability)

        ################################ output train data ################################
        # if there is an element in satisfiability_list is SAT, return SAT
        if SAT in satisfiability_list:
            return self.record_and_close_branch_and_output_eq(SAT, branch_variables, node_info, eq)
        elif UNKNOWN in satisfiability_list:
            return self.record_and_close_branch_and_output_eq(UNKNOWN, branch_variables, node_info, eq)
        else:
            return self.record_and_close_branch_and_output_eq(UNSAT, branch_variables, node_info, eq)

    def record_and_close_branch_and_output_eq(self, satisfiability: str, variables, node_info, eq: Equation): #non-leaf node
        if satisfiability != UNKNOWN:
            middle_eq_file_name = self.file_name + "_" + str(node_info[0])
            self._output_train_data(middle_eq_file_name, eq, satisfiability, node_info, "diamond")

        return self._record_and_close_branch_without_file(satisfiability, variables, eq)
        # node_info[1]["status"] = satisfiability
        # return satisfiability, variables

    def _record_and_close_branch_with_file(self, satisfiability: str, variables, node_info, eq: Equation): #leaf node
        if satisfiability != UNKNOWN:
            if random.random() < OUTPUT_LEAF_NODE_PERCENTAGE:  # random.random() generates a float between 0.0 and 1.0
                middle_eq_file_name = self.file_name + "_" + str(node_info[0])
                self._output_train_data(middle_eq_file_name, eq, satisfiability, node_info, "box")
            else:
                pass
        return self._record_and_close_branch_without_file(satisfiability,variables,eq)
        # node_info[1]["status"] = satisfiability
        # return satisfiability, variables

    def _record_and_close_branch_without_file(self, satisfiability: str, variables, node_info, eq: Equation): #leaf node
        node_info[1]["status"] = satisfiability
        return satisfiability, variables

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

        return local_left_terms_queue, local_right_terms_queue, updated_variables, left_term.get_value_str + ">" + right_term.get_value_str + ": " + left_term.get_value_str + "=" + right_term.get_value_str + fresh_variable_term.get_value_str

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

        return local_left_terms_queue, local_right_terms_queue, updated_variables, left_term.value.value + "=" + right_term.get_value_str + ": " + left_term.get_value_str + "=" + right_term.get_value_str

    def one_variable_one_terminal_split_branch_1(self, left_terms: List[Term], right_terms: List[Term], variables):
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

        return new_left_terms_queue, new_right_terms_queue, updated_variables, left_term.get_value_str + "=\"\""

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

        return local_left_terms_queue, local_right_terms_queue, updated_variables, left_term.get_value_str + "=" + right_term.get_value_str + fresh_variable_term.get_value_str

    def _create_fresh_variables(self, variables: List[Variable]) -> Term:
        # fresh_variable_term = Term(Variable(left_term.value.value + "'"))  # V1'
        available_caps = identify_available_capitals("".join([v.value for v in variables]))
        fresh_variable_term = Term(Variable(available_caps.pop()))  # a capital rather than V1
        return fresh_variable_term

    def replace_a_term(self, old_term: Term, new_term: Term, terms_queue: Deque[Term]):
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

    def get_leading_terminals(self, term_list: Deque[Term]) -> List[Term]:
        leading_terminal_list = []
        first_variable = None
        for t in term_list:
            if t.value_type == Variable:
                first_variable = t
                break
            else:
                leading_terminal_list.append(t)
        return leading_terminal_list, first_variable

    def visualize(self, file_path, graph_func):
        visualize_path_html(self.nodes, self.edges, file_path)
        visualize_path_png(self.nodes, self.edges, file_path)

        self.equation_list[0].visualize_graph(file_path, graph_func)
