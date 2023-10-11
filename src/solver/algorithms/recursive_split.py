import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import EMPTY_TERMINAL, BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED, recursion_limit, \
    RECURSION_DEPTH_EXCEEDED, RECURSION_ERROR
from src.solver.DataTypes import Assignment, Term, Terminal, Variable
from src.solver.utils import flatten_list, assemble_parsed_content, remove_duplicates
from src.solver.visualize_util import visualize_path, visualize_path_html
from .abstract_algorithm import AbstractAlgorithm
import sys


class ElimilateVariablesRecursive(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], left_terms: List[Term],
                 right_terms: List[Term], parameters: Dict):
        super().__init__(terminals, variables, left_terms, right_terms)
        self.assignment = Assignment()
        self.parameters = parameters
        self.total_explore_paths_call = 0
        self.nodes = []
        self.edges = []

        sys.setrecursionlimit(recursion_limit)
        # print("recursion limit number", sys.getrecursionlimit())

    def run(self):
        try:
            satisfiability, variables = self.explore_paths(deque(self.left_terms), deque(self.right_terms),
                                                           self.variables, {"node_number": None, "label": None})
        except RecursionError as e:
            if "maximum recursion depth exceeded" in str(e):
                satisfiability = RECURSION_DEPTH_EXCEEDED
                print(RECURSION_DEPTH_EXCEEDED)
            else:
                satisfiability = RECURSION_ERROR
                print(RECURSION_ERROR)

        result_dict = {"result": satisfiability, "assignment": self.assignment, "left_terms": self.left_terms,
                       "right_terms": self.right_terms,
                       "variables": self.variables, "terminals": self.terminals,
                       "total_explore_paths_call": self.total_explore_paths_call}
        return result_dict

    def explore_paths(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term], variables: List[Variable],
                      previous_dict) -> Tuple[str, List[Variable]]:

        string_equation, string_terminals, string_variables = self.pretty_print_current_equation(left_terms_queue,
                                                                                                 right_terms_queue)
        self.total_explore_paths_call += 1
        current_node_number = self.total_explore_paths_call
        node_info = (current_node_number, {"label": string_equation, "status": None})
        self.nodes.append(node_info)
        if previous_dict["node_number"] != None:
            self.edges.append((previous_dict["node_number"], current_node_number, {'label': previous_dict["label"]}))

        # terminate conditions
        # both side only have terminals
        if len(variables) == 0:
            if self.check_equation(left_terms_queue, right_terms_queue) == True:
                node_info[1]["status"] = "SAT"
                return "SAT", variables
            else:
                node_info[1]["status"] = "UNSAT"
                return "UNSAT", variables

        # both side only have variables
        left_contains_no_terminal = not any(isinstance(term.value, Terminal) for term in left_terms_queue)
        right_contains_no_terminal = not any(isinstance(term.value, Terminal) for term in right_terms_queue)
        if left_contains_no_terminal and right_contains_no_terminal:
            node_info[1]["status"] = "SAT"
            return "SAT", variables

        # both side contains variables and terminals
        # both side empty
        if len(left_terms_queue) == 0 and len(right_terms_queue) == 0:
            node_info[1]["status"] = "SAT"
            return "SAT", variables
        # left side empty
        if len(left_terms_queue) == 0 and len(right_terms_queue) != 0:
            node_info[1]["status"] = "UNSAT"
            return "UNSAT", variables  # since one side has terminals
        # right side empty
        if len(left_terms_queue) != 0 and len(right_terms_queue) == 0:
            node_info[1]["status"] = "UNSAT"
            return "UNSAT", variables  # since one side has terminals

        #########################################################################

        # split equation
        left_term = left_terms_queue[0]
        right_term = right_terms_queue[0]
        if left_term.value == right_term.value:  # both side are the same
            # print("*","left = right","*")
            left_terms_queue.popleft()
            right_terms_queue.popleft()
            updated_variables = self.update_variables(left_terms_queue, right_terms_queue)
            branch_satisfiability, branch_variables = self.explore_paths(left_terms_queue, right_terms_queue,
                                                                         updated_variables,
                                                                         {"node_number": current_node_number,
                                                                          "label": "L=R"})
            if branch_satisfiability == "SAT":
                node_info[1]["status"] = "SAT"
                return "SAT", branch_variables
            else:
                node_info[1]["status"] = "UNSAT"
                return "UNSAT", branch_variables
            return self.explore_paths(left_terms_queue, right_terms_queue, updated_variables,
                                      {"node_number": current_node_number, "label": "L=R"})
        else:  # both side are different
            if type(left_term.value) == Variable and type(
                    right_term.value) == Variable:  # both side are differernt variables
                branch_list = [self.two_variables_split_branch_1, self.two_variables_split_branch_2,
                               self.two_variables_split_branch_3]
                random.shuffle(branch_list)

                l, r, v, edge_label = branch_list[0](left_terms_queue, right_terms_queue, variables)  # split equation
                branch_1_satisfiability, branch_1_variables = self.explore_paths(l, r, v,
                                                                                 {"node_number": current_node_number,
                                                                                  "label": edge_label})
                if branch_1_satisfiability == "SAT":
                    node_info[1]["status"] = "SAT"
                    return "SAT", branch_1_variables
                else:
                    node_info[1]["status"] = "UNSAT"
                    branch_list = branch_list[1:]  # branch_1 closed

                l, r, v, edge_label = branch_list[0](left_terms_queue, right_terms_queue, variables)  # split equation
                branch_2_satisfiability, branch_2_variables = self.explore_paths(l, r, v,
                                                                                 {"node_number": current_node_number,
                                                                                  "label": edge_label})
                if branch_2_satisfiability == "SAT":
                    node_info[1]["status"] = "SAT"
                    return "SAT", branch_2_variables
                else:
                    node_info[1]["status"] = "UNSAT"
                    branch_list = branch_list[1:]  # branch_2 closed

                l, r, v, edge_label = branch_list[0](left_terms_queue, right_terms_queue, variables)  # split equation
                branch_3_satisfiability, branch_3_variables = self.explore_paths(l, r, v,
                                                                                 {"node_number": current_node_number,
                                                                                  "label": edge_label})
                if branch_3_satisfiability == "SAT":
                    node_info[1]["status"] = "SAT"
                    return "SAT", branch_3_variables
                else:
                    node_info[1]["status"] = "UNSAT"
                    return "UNSAT", branch_3_variables  # all branches closed

            elif type(left_term.value) == Variable and type(
                    right_term.value) == Terminal:  # left side is variable, right side is terminal
                branch_list = [self.one_variable_one_terminal_split_branch_1,
                               self.one_variable_one_terminal_split_branch_2]
                random.shuffle(branch_list)

                l, r, v, edge_label = branch_list[0](left_terms_queue, right_terms_queue, variables)  # split equation
                branch_1_satisfiability, branch_1_variables = self.explore_paths(l, r, v,
                                                                                 {"node_number": current_node_number,
                                                                                  "label": edge_label})
                if branch_1_satisfiability == "SAT":
                    node_info[1]["status"] = "SAT"
                    return "SAT", branch_1_variables
                else:
                    node_info[1]["status"] = "UNSAT"
                    branch_list = branch_list[1:]  # branch_1 closed

                l, r, v, edge_label = branch_list[0](left_terms_queue, right_terms_queue, variables)  # split equation
                branch_2_satisfiability, branch_2_variables = self.explore_paths(l, r, v,
                                                                                 {"node_number": current_node_number,
                                                                                  "label": edge_label})
                if branch_2_satisfiability == "SAT":
                    node_info[1]["status"] = "SAT"
                    return "SAT", branch_2_variables
                else:
                    node_info[1]["status"] = "UNSAT"
                    return "UNSAT", branch_2_variables  # all branches closed



            elif type(left_term.value) == Terminal and type(
                    right_term.value) == Variable:  # left side is terminal, right side is variable
                branch_list = [self.one_variable_one_terminal_split_branch_1,
                               self.one_variable_one_terminal_split_branch_2]
                random.shuffle(branch_list)

                l, r, v, edge_label = branch_list[0](right_terms_queue, left_terms_queue, variables)  # split equation
                branch_1_satisfiability, branch_1_variables = self.explore_paths(l, r, v,
                                                                                 {"node_number": current_node_number,
                                                                                  "label": edge_label})
                if branch_1_satisfiability == "SAT":
                    node_info[1]["status"] = "SAT"
                    return "SAT", branch_1_variables
                else:
                    node_info[1]["status"] = "UNSAT"
                    branch_list = branch_list[1:]  # branch_1 closed

                l, r, v, edge_label = branch_list[0](right_terms_queue, left_terms_queue, variables)  # split equation
                branch_3_satisfiability, branch_3_variables = self.explore_paths(l, r, v,
                                                                                 {"node_number": current_node_number,
                                                                                  "label": edge_label})
                if branch_3_satisfiability == "SAT":
                    node_info[1]["status"] = "SAT"
                    return "SAT", branch_3_variables
                else:
                    node_info[1]["status"] = "UNSAT"
                    return "UNSAT", variables  # all branches closed

            elif type(left_term.value) == Terminal and type(
                    right_term.value) == Terminal:  # both side are different terminals
                node_info[1]["status"] = "UNSAT"
                return "UNSAT", variables

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

        return local_left_terms_queue, local_right_terms_queue, updated_variables, "V1>V2"

    def two_variables_split_branch_2(self, left_terms_queue: Deque[Term], right_terms_queue: Deque[Term],
                                     variables: List[Variable]):
        '''
        Equation: V1 [Terms] = V2 [Terms]
        Assume |V1| < |V2|
        Replace V2 with V1V2'
        Obtain [Terms] [V2/V1V2'] = V2' [Terms] [V2/V1V2']
        '''
        local_left_terms_queue, local_right_terms_queue, updated_variables, _ = self.two_variables_split_branch_1(
            right_terms_queue, left_terms_queue, variables)
        return local_left_terms_queue, local_right_terms_queue, updated_variables, "V1<V2"

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

        return local_left_terms_queue, local_right_terms_queue, updated_variables, "V1=V2"

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

        return new_left_terms_queue, new_right_terms_queue, updated_variables, "V1=\"\""

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

        return local_left_terms_queue, local_right_terms_queue, updated_variables, "V1=aV1'"

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
