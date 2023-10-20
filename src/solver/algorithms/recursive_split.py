import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED, recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR, SAT, UNSAT, UNKNOWN
from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, EMPTY_TERMINAL
from src.solver.utils import assemble_parsed_content
from src.solver.independent_utils import remove_duplicates, flatten_list, strip_file_name_suffix, \
    dump_to_json_with_format
from src.solver.visualize_util import visualize_path, visualize_path_html
from src.solver.algorithms.abstract_algorithm import AbstractAlgorithm
from src.solver.models.utils import load_model
from src.solver.models.Dataset import WordEquationDataset
from dgl.dataloading import GraphDataLoader
import sys

sys.path.append(
    "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/solver/models")


class ElimilateVariablesRecursive(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation],
                 parameters: Dict):

        super().__init__(terminals, variables, equation_list)

        self.assignment = Assignment()
        self.parameters = parameters
        self.total_explore_paths_call = 0
        self.nodes = []
        self.edges = []
        self.temp_graph_folder = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/solver/temp/graph_prediction"
        self.gnn_model = None
        sys.setrecursionlimit(recursion_limit)
        # print("recursion limit number", sys.getrecursionlimit())

        if self.parameters["use_gnn"] == True:
            # Load the model
            model_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/models/model.pth"
            self.gnn_model = load_model(model_path)

    def run(self):

        first_equation = self.equation_list[0]
        left_terms = first_equation.left_terms
        right_terms = first_equation.right_terms
        try:
            satisfiability, variables = self.explore_paths(deque(left_terms), deque(right_terms),
                                                           self.variables, {"node_number": None, "label": None})
        except RecursionError as e:
            if "maximum recursion depth exceeded" in str(e):
                satisfiability = RECURSION_DEPTH_EXCEEDED
                print(RECURSION_DEPTH_EXCEEDED)
            else:
                satisfiability = RECURSION_ERROR
                print(RECURSION_ERROR)

        result_dict = {"result": satisfiability, "assignment": self.assignment, "equation_list": self.equation_list,
                       "variables": self.variables, "terminals": self.terminals,
                       "total_explore_paths_call": self.total_explore_paths_call}
        return result_dict

    def explore_paths(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term], variables: List[Variable],
                      previous_dict) -> Tuple[str, List[Variable]]:

        ################################ Record nodes and edges ################################

        string_equation, string_terminals, string_variables = self.pretty_print_current_equation(left_terms_queue,
                                                                                                 right_terms_queue)
        self.total_explore_paths_call += 1
        current_node_number = self.total_explore_paths_call
        node_info = (current_node_number, {"label": string_equation, "status": None})
        self.nodes.append(node_info)
        if previous_dict["node_number"] != None:
            self.edges.append((previous_dict["node_number"], current_node_number, {'label': previous_dict["label"]}))

        ################################ Check terminate conditions ################################

        ## both side only have terminals
        if len(variables) == 0:
            satisfiability = SAT if self.check_equation(left_terms_queue, right_terms_queue) == True else UNSAT
            return self.record_and_close_branch(satisfiability, variables, node_info)

        ## both side only have variables
        left_contains_no_terminal = not any(isinstance(term.value, Terminal) for term in left_terms_queue)
        right_contains_no_terminal = not any(isinstance(term.value, Terminal) for term in right_terms_queue)
        if left_contains_no_terminal and right_contains_no_terminal:
            return self.record_and_close_branch(SAT, variables, node_info)

        ## both side contains variables and terminals
        ###both side empty
        if len(left_terms_queue) == 0 and len(right_terms_queue) == 0:
            return self.record_and_close_branch(SAT, variables, node_info)
        ### left side empty
        if len(left_terms_queue) == 0 and len(right_terms_queue) != 0:
            return self.record_and_close_branch(UNSAT, variables, node_info)  # since one side has terminals
        ### right side empty
        if len(left_terms_queue) != 0 and len(right_terms_queue) == 0:
            return self.record_and_close_branch(UNSAT, variables, node_info)  # since one side has terminals

        ################################ Split equation ################################

        left_term = left_terms_queue[0]
        right_term = right_terms_queue[0]
        # both side are the same
        if left_term.value == right_term.value:
            return self.both_side_same_terms(left_terms_queue, right_terms_queue, variables, current_node_number,
                                             node_info)

        # both side are different
        else:
            ## both side are differernt variables
            if type(left_term.value) == Variable and type(right_term.value) == Variable:
                return self.both_side_different_variables(left_terms_queue, right_terms_queue, variables,
                                                          current_node_number, node_info)

            ## left side is variable, right side is terminal
            elif type(left_term.value) == Variable and type(right_term.value) == Terminal:
                return self.left_side_variable_right_side_terminal(left_terms_queue, right_terms_queue, variables,
                                                                   current_node_number, node_info)

            ## left side is terminal, right side is variable
            elif type(left_term.value) == Terminal and type(right_term.value) == Variable:
                return self.left_side_variable_right_side_terminal(right_terms_queue, left_terms_queue, variables,
                                                                   current_node_number, node_info)

            ## both side are different terminals
            elif type(left_term.value) == Terminal and type(right_term.value) == Terminal:
                return self.record_and_close_branch(UNSAT, variables, node_info)

    def both_side_same_terms(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term], variables,
                             current_node_number, node_info):
        left_poped_list_str, right_poped_list_str = self.pop_both_same_terms(left_terms_queue, right_terms_queue)
        updated_variables = self.update_variables(left_terms_queue, right_terms_queue)
        branch_satisfiability, branch_variables = self.explore_paths(left_terms_queue, right_terms_queue,
                                                                     updated_variables,
                                                                     {"node_number": current_node_number,
                                                                      "label": left_poped_list_str + "=" + right_poped_list_str})
        return self.record_and_close_branch(branch_satisfiability, branch_variables, node_info)

    def both_side_different_variables(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term], variables,
                                      current_node_number, node_info):
        # Define the methods for each branch
        branch_methods = [
            self.two_variables_split_branch_1,
            self.two_variables_split_branch_2,
            self.two_variables_split_branch_3
        ]
        return self._execute_branching(left_terms_queue, right_terms_queue, variables, current_node_number, node_info,
                                       branch_methods)

    def left_side_variable_right_side_terminal(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term],
                                               variables, current_node_number, node_info):
        # Define the methods for each branch
        branch_methods = [
            self.one_variable_one_terminal_split_branch_1,
            self.one_variable_one_terminal_split_branch_2
        ]

        return self._execute_branching(left_terms_queue, right_terms_queue, variables, current_node_number, node_info,
                                       branch_methods)

    def _execute_branching(self, left_terms_queue, right_terms_queue, variables, current_node_number, node_info,
                           branch_methods):
        if self.parameters["use_gnn"]:
            return self._use_gnn_branching(left_terms_queue, right_terms_queue, variables, current_node_number,
                                             node_info, branch_methods)
        else:
            return self._use_random_branching(left_terms_queue, right_terms_queue, variables, current_node_number,
                                                node_info, branch_methods)

    def _use_gnn_branching(self, left_terms_queue, right_terms_queue, variables, current_node_number, node_info,
                           branch_methods):
        # Compute branches and prepare data structures
        branches = []
        graph_list = []

        for method in branch_methods:
            l, r, v, edge_label = method(left_terms_queue, right_terms_queue, variables)
            eq = Equation(l, r)
            nodes, edges = eq.get_graph_1()
            graph_dict = eq.graph_to_gnn_format(nodes, edges)
            tuple_data = (eq, v, edge_label)

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
            satisfiability, variables = self.explore_paths(eq.left_terms, eq.right_terms, v,
                                                           {"node_number": current_node_number, "label": edge_label})

            # Handle branch outcome
            if satisfiability == SAT:
                return self.record_and_close_branch(SAT, variables, node_info)
            elif i < len(sorted_prediction_list) - 1:  # If not the last branch, mark as UNSAT and continue
                node_info[1]["status"] = UNSAT
            else:  # For the last branch
                return self.record_and_close_branch(satisfiability, variables, node_info)

    def _use_random_branching(self, left_terms_queue, right_terms_queue, variables, current_node_number, node_info,
                              branch_methods):
        random.shuffle(branch_methods)

        for i, branch in enumerate(branch_methods):
            l, r, v, edge_label = branch(left_terms_queue, right_terms_queue, variables)
            satisfiability, branch_variables = self.explore_paths(l, r, v,
                                                                  {"node_number": current_node_number,
                                                                   "label": edge_label})

            if satisfiability == SAT:
                return self.record_and_close_branch(SAT, branch_variables, node_info)

            # If we reach here and it's not the last branch, update status to UNSAT and continue
            if i < len(branch_methods) - 1:
                node_info[1]["status"] = UNSAT
            else:
                # If it's the last branch, record and close regardless of the outcome
                return self.record_and_close_branch(satisfiability, branch_variables, node_info)

    def record_and_close_branch(self, satisfiability: str, variables, node_info):
        node_info[1]["status"] = satisfiability
        return satisfiability, variables

    def pop_both_same_terms(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term]):
        '''
        This will change left_terms_queue and right_terms_queue
        '''
        equal_term_counter = 0
        for l, r in zip(left_terms_queue, right_terms_queue):
            if l.value == r.value:
                equal_term_counter += 1
            else:
                break

        left_poped_list = []
        right_poped_list = []
        for i in range(equal_term_counter):
            left_poped_list.append(left_terms_queue.popleft())
            right_poped_list.append(right_terms_queue.popleft())

        left_poped_list_str = "".join([term.get_value_str for term in left_poped_list])
        right_poped_list_str = "".join([term.get_value_str for term in right_poped_list])

        return left_poped_list_str, right_poped_list_str

    def two_variables_split_branch_1(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term],
                                     variables: List[Variable]):
        '''
        Equation: V1 [Terms] = V2 [Terms]
        Assume |V1| > |V2|
        Replace V1 with V2V1'
        Obtain V1' [Terms] [V1/V2V1'] = [Terms] [V1/V2V1']
        '''
        # local variables
        local_left_terms_queue = left_terms_queue.copy()
        local_right_terms_queue = right_terms_queue.copy()

        # pop both sides to [Terms] = [Terms]
        left_term = local_left_terms_queue.popleft()
        right_term = local_right_terms_queue.popleft()

        # create fresh variable V1'
        fresh_variable_term = Term(Variable(left_term.value.value + "'"))  # V1'
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

    def two_variables_split_branch_2(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term],
                                     variables: List[Variable]):
        '''
        Equation: V1 [Terms] = V2 [Terms]
        Assume |V1| < |V2|
        Replace V2 with V1V2'
        Obtain [Terms] [V2/V1V2'] = V2' [Terms] [V2/V1V2']
        '''
        local_left_terms_queue, local_right_terms_queue, updated_variables, edge_label = self.two_variables_split_branch_1(
            right_terms_queue, left_terms_queue, variables)
        return local_left_terms_queue, local_right_terms_queue, updated_variables, edge_label

    def two_variables_split_branch_3(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term],
                                     variables: List[Variable]):
        '''
        Equation: V1 [Terms] = V2 [Terms]
        Assume |V1| = |V2|
        Replace V1 with V2
        Obtain [Terms] [V1/V2] = [Terms] [V1/V2]
        '''
        # local variables
        local_left_terms_queue = left_terms_queue.copy()
        local_right_terms_queue = right_terms_queue.copy()

        # pop both sides to [Terms] = [Terms]
        left_term = local_left_terms_queue.popleft()
        right_term = local_right_terms_queue.popleft()

        # replace V1 with V2 to obtain [Terms] [V1/V2] = [Terms] [V1/V2]
        self.replace_a_term(old_term=left_term, new_term=right_term, terms_queue=local_left_terms_queue)
        self.replace_a_term(old_term=left_term, new_term=right_term, terms_queue=local_right_terms_queue)

        # update variables
        updated_variables = self.update_variables(local_left_terms_queue, local_right_terms_queue)

        return local_left_terms_queue, local_right_terms_queue, updated_variables, left_term.value.value + "=" + right_term.get_value_str + ": " + left_term.get_value_str + "=" + right_term.get_value_str

    def one_variable_one_terminal_split_branch_1(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term],
                                                 variables: List[Variable]):
        '''
        Equation: V1 [Terms] = a [Terms]
        Assume V1 = ""
        Delete V1
        Obtain [Terms] [V1/""] = a [Terms] [V1/""]
        '''
        # print("*","one_variable_one_terminal_split_branch_1","*")
        # local variables
        local_left_terms_queue = left_terms_queue.copy()
        local_right_terms_queue = right_terms_queue.copy()

        # pop left side to [Terms] = a [Terms]
        left_term = local_left_terms_queue.popleft()

        # delete V1 from both sides to obtain [Terms] [V1/""] = a [Terms] [V1/""]
        new_left_terms_queue = deque(item for item in local_left_terms_queue if item != left_term)
        new_right_terms_queue = deque(item for item in local_right_terms_queue if item != left_term)

        # update variables
        updated_variables = self.update_variables(new_left_terms_queue, new_right_terms_queue)

        return new_left_terms_queue, new_right_terms_queue, updated_variables, left_term.get_value_str + "=\"\""

    def one_variable_one_terminal_split_branch_2(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term],
                                                 variables: List[Variable]):
        '''
        Equation: V1 [Terms] = a [Terms]
        Assume V1 = aV1'
        Replace V1 with aV1'
        Obtain V1' [Terms] [V1/aV1'] = [Terms] [V1/aV1']
        '''
        # print("*","one_variable_one_terminal_split_branch_2","*")
        # local variables
        local_left_terms_queue = left_terms_queue.copy()
        local_right_terms_queue = right_terms_queue.copy()

        # pop both sides to [Terms] = [Terms]
        left_term = local_left_terms_queue.popleft()
        right_term = local_right_terms_queue.popleft()

        # create fresh variable V1'
        fresh_variable_term = Term(Variable(left_term.value.value + "'"))  # V1'

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

    def visualize(self, file_path):
        visualize_path_html(self.nodes, self.edges, file_path)
        self.equation_list[0].visualize_graph(file_path)

    def output_train_data(self, file_path):
        pass
