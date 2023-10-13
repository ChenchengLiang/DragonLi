import random
from collections import deque
from typing import List, Dict, Tuple, Deque, Union, Callable

from src.solver.Constants import BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED
from src.solver.DataTypes import Assignment, Term, Terminal, Variable, EMPTY_TERMINAL
from src.solver.utils import flatten_list, assemble_parsed_content, remove_duplicates
from src.solver.visualize_util import visualize_path
from .abstract_algorithm import AbstractAlgorithm


class ElimilateVariables(AbstractAlgorithm):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], left_terms: List[Term],
                 right_terms: List[Term], parameters: Dict):
        super().__init__(terminals, variables, left_terms, right_terms)
        self.assignment = Assignment()
        self.parameters = parameters
        self.nodes: List[str] = []
        self.edges: List[Tuple[str, str, Dict]] = []
        self.final_nodes: List[str] = []
        self.final_edges: List[Tuple[str, str, Dict]] = []

    def run(self):
        # pre-process
        result, _, preprocessed_local_left_terms, preprocessed_local_right_terms = self.preprocess_equation()
        self.pretty_print_current_equation(preprocessed_local_left_terms, preprocessed_local_right_terms)

        # explore path start
        path_count = 0
        while True:
            print("-" * 10, "path ", path_count, " start", "-" * 10)
            result, assignment, left_term_queue, right_term_queue = self.run_one_path(preprocessed_local_left_terms,
                                                                                      preprocessed_local_right_terms)
            print("path result:", result)
            print("-" * 10, "path ", path_count, " end", "-" * 10)
            path_count += 1

            if result == True or result == False:
                result_dict={"result": result, "assignment": assignment, "left_terms": left_term_queue,
                        "right_terms": right_term_queue,
                        "variables": self.variables, "terminals": self.terminals}
                return result_dict

            if path_count > MAX_PATH:
                return {"result": MAX_PATH_REACHED, "assignment": assignment, "left_terms": left_term_queue,
                        "right_terms": right_term_queue,
                        "variables": self.variables, "terminals": self.terminals}

    def run_one_path(self, preprocessed_local_left_terms, preprocessed_local_right_terms):
        # create local variables
        left_term_queue = deque(preprocessed_local_left_terms)
        right_term_queue = deque(preprocessed_local_right_terms)
        path_depth = 0

        while len(left_term_queue) != 0 and len(right_term_queue) != 0:  # while two sides are not empty
            first_left_term = left_term_queue[0]  # unwrap first left hand term
            first_right_term = right_term_queue[0]  # unwrap first right hand term

            if first_left_term.value == first_right_term.value:  # example: V1 T2 T3 ... = V1 T5 T6 ... and # example: a T2 T3 ... = a T5 T6 ...
                before_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue,
                                                                                          right_term_queue)
                self.nodes.append(before_process_string_equation)

                # remove first term from both sides
                left_term_queue.popleft()
                right_term_queue.popleft()

                self.update_variable_list(left_term_queue, right_term_queue)
                after_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue,
                                                                                         right_term_queue)
                self.edges.append(
                    (before_process_string_equation, after_process_string_equation, {"label": "T1 == T2"}))
            else:  # first_left_term.value != first_right_term.value
                if type(first_left_term.value) == Variable:
                    # first_right_term = right_term_queue[0]  # unwrap first right hand term
                    if type(first_right_term.value) == Variable:  # example: V1 T2 T3 ... = V4 T5 T6 ...
                        # split equation
                        self.split_equation_two_variables(left_term_queue, right_term_queue)
                    elif type(first_right_term.value) == Terminal:  # example: V1 T2 T3 ... = a T5 T6 ...
                        # split equation
                        self.split_equation_one_variable(left_term_queue, right_term_queue)
                elif type(first_left_term.value) == Terminal:
                    # first_right_term = right_term_queue[0]  # unwrap first right hand term
                    if type(first_right_term.value) == Variable:  # example: a T2 T3 ... = V4 T5 T6 ...
                        # split equation
                        self.split_equation_one_variable(right_term_queue, left_term_queue)
                    elif type(first_right_term.value) == Terminal:  # example: a T2 T3 ... = b T5 T6 ...
                        return BRANCH_CLOSED, self.assignment, left_term_queue, right_term_queue

            path_depth += 1
            print("path_depth: ", path_depth)
            # print("length", len(left_term_queue), len(right_term_queue))

        # two side empty
        if len(left_term_queue) == 0 and len(right_term_queue) == 0:
            return True, Assignment(), left_term_queue, right_term_queue
        # one side empty
        if len(left_term_queue) == 0 and len(right_term_queue) != 0:
            result, _ = self.left_terms_empty(left_term_queue, right_term_queue)
            return result, self.assignment, left_term_queue, right_term_queue
        if len(left_term_queue) != 0 and len(right_term_queue) == 0:
            result, _ = self.right_terms_empty(left_term_queue, right_term_queue)
            return result, self.assignment, left_term_queue, right_term_queue

    def split_equation_two_variables(self, left_terms: deque, right_terms: deque):
        print("split_equation_two_variables")

        strategies = [self.left_variable_larger_than_right_variable, self.left_variable_smaller_than_right_variable,
                      self.left_variable_equal_right_variable]
        # Define the probabilities of selecting each strategy
        probabilities = [0.4, 0.3, 0.3]  # Adjust these probabilities as needed

        # Use random.choices() to select a function with specified probabilities
        selected_strategy = random.choices(strategies, probabilities)[0]
        selected_strategy(left_terms, right_terms)


    def split_equation_one_variable(self, left_terms: deque, right_terms: deque):
        print("split_equation_one_variable")
        strategies = [self.left_variable_empty, self.left_variable_not_empty]
        # Define the probabilities of selecting each strategy
        probabilities = [0.5, 0.5]  # Adjust these probabilities as needed

        # Use random.choices() to select a function with specified probabilities
        selected_strategy = random.choices(strategies, probabilities)[0]
        selected_strategy(left_terms, right_terms)


    def left_variable_larger_than_right_variable(self, left_term_queue: Deque[Term], right_term_queue: Deque[Term]):

        def split_rule(left_term_queue, right_term_queue):
            left_term = left_term_queue.popleft()
            right_term = right_term_queue.popleft()

            # split variables
            fresh_variable_term = Term(Variable(left_term.value.value + "_NEW"))
            new_variable_term = [right_term, fresh_variable_term]

            # construct new equation
            left_term_queue.appendleft(fresh_variable_term)
            # replace left_term with new_term
            self.replace_a_term(left_term, new_variable_term, left_term_queue)
            self.replace_a_term(left_term, new_variable_term, right_term_queue)

            # flatten euqation
            left_term_list = flatten_list(left_term_queue)
            right_term_list = flatten_list(right_term_queue)

            # update variable list
            self.update_variable_list(left_term_list, right_term_list)

            left_term_queue.clear()
            right_term_queue.clear()
            left_term_queue.extend(left_term_list)
            right_term_queue.extend(right_term_list)

        return self.log_split(split_rule, left_term_queue, right_term_queue, "left_V != right_V")


    def left_variable_smaller_than_right_variable(self, left_term_queue: deque, right_term_queue: deque):
        self.left_variable_larger_than_right_variable(right_term_queue, left_term_queue)

    def left_variable_equal_right_variable(self, left_term_queue, right_term_queue):

        def split_rule(left_term_queue, right_term_queue):
            left_term = left_term_queue.popleft()
            right_term = right_term_queue.popleft()

            # replace left_term variable with right_term variable
            self.replace_a_term(left_term, right_term, left_term_queue)
            self.replace_a_term(left_term, right_term, right_term_queue)

            self.update_variable_list(left_term_queue, right_term_queue)

        return self.log_split(split_rule,left_term_queue, right_term_queue,"left_V = right_V")




    def left_variable_empty(self, left_term_queue: Deque[Term], right_term_queue: Deque[Term]):

        def split_rule(left_term_queue, right_term_queue):
            left_term = left_term_queue.popleft()

            self.replace_a_term(left_term, Term(EMPTY_TERMINAL), left_term_queue)
            self.replace_a_term(left_term, Term(EMPTY_TERMINAL), right_term_queue)
            self.remove_empty_terminals(left_term_queue)
            self.remove_empty_terminals(right_term_queue)

            self.update_variable_list(left_term_queue, right_term_queue)

        return self.log_split(split_rule, left_term_queue, right_term_queue, "left_V = \"\"")

    def left_variable_not_empty(self, left_term_queue: deque, right_term_queue: deque):
        '''
        replace X with "Terminal X_NEW"
        '''

        def split_rule(left_term_queue, right_term_queue):
            left_term = left_term_queue.popleft()
            right_term = right_term_queue.popleft()

            # split variables
            fresh_variable_term = Term(Variable(left_term.value.value + "_NEW"))
            new_variable_term = [right_term, fresh_variable_term]

            # construct new equation
            left_term_queue.appendleft(fresh_variable_term)
            # replace old variables
            self.replace_a_term(left_term, new_variable_term, left_term_queue)
            self.replace_a_term(left_term, new_variable_term, right_term_queue)

            # flatten euqation
            left_term_list = flatten_list(left_term_queue)
            right_term_list = flatten_list(right_term_queue)

            # update variable list
            self.update_variable_list(left_term_list, right_term_list)

            left_term_queue.clear()
            right_term_queue.clear()
            left_term_queue.extend(left_term_list)
            right_term_queue.extend(right_term_list)

        left_term = left_term_queue[0]
        right_term=right_term_queue[0]

        return self.log_split(split_rule, left_term_queue, right_term_queue, label="left_V = "+right_term.value.value+left_term.value.value+"_NEW")

    def left_terms_empty(self, left_term_queue: Deque[Term], right_term_queue: Deque[Term]):
        right_term_queue_contains_terminal = any(isinstance(item.value, Terminal) for item in right_term_queue)
        if right_term_queue_contains_terminal == True:
            return BRANCH_CLOSED, self.assignment
        else:
            # assign all variables in right_term_queue to empty
            for x in self.variables:
                self.assignment.set_assignment(x, [EMPTY_TERMINAL])
            return True, self.assignment

    def right_terms_empty(self, left_term_queue: deque, right_term_queue: deque):
        return self.left_terms_empty(right_term_queue, left_term_queue)

    def preprocess_equation(self):

        # local variables
        local_left_term_queue = deque(self.left_terms)
        local_right_term_queue = deque(self.right_terms)

        print("-" * 10, "pre-process", "-" * 10)
        self.pretty_print_current_equation(local_left_term_queue, local_right_term_queue)

        if len(self.variables) == 0:  # if no variable
            return self.check_equation(local_left_term_queue,
                                       local_right_term_queue), self.assignment, local_left_term_queue, local_right_term_queue

        #  discard both terms if there are the same Terms
        while len(local_left_term_queue) != 0 and len(local_right_term_queue) != 0:
            left_term = local_left_term_queue[0]
            right_term = local_right_term_queue[0]
            if type(left_term.value) == Terminal and type(
                    right_term.value) == Terminal:  # example: a T2 T3 ... = a T5 T6 ...
                if left_term.value == right_term.value:
                    local_left_term_queue.popleft()
                    local_right_term_queue.popleft()
                else:
                    return False, self.assignment, local_left_term_queue, local_right_term_queue
            elif type(left_term.value) == Variable and type(
                    right_term.value) == Variable:  # example: V1 T2 T3 ... = V1 T5 T6 ...
                if left_term.value == right_term.value:
                    local_left_term_queue.popleft()
                    local_right_term_queue.popleft()
                else:
                    return None, self.assignment, local_left_term_queue, local_right_term_queue
            else:
                return None, self.assignment, local_left_term_queue, local_right_term_queue

        return None, self.assignment, local_left_term_queue, local_right_term_queue

    def replace_a_term(self, old_term: Term, new_term: Term, term_queue: Deque[Term]):
        for i, t in enumerate(term_queue):
            if t.value == old_term.value:
                term_queue[i] = new_term

    def remove_empty_terminals(self, term_queue: deque):
        local_term_queue = deque([t for t in term_queue if t.value != EMPTY_TERMINAL])
        term_queue.clear()
        term_queue.extend(local_term_queue)

    def update_variable_list(self, left_term_list: Union[List[Term], Deque[Term]],
                             right_term_list: Union[List[Term], Deque[Term]]):
        left_term_list = list(left_term_list)
        right_term_list = list(right_term_list)
        self.variables = []
        for t in left_term_list + right_term_list:
            if type(t.value) == Variable:
                self.variables.append(t.value)
        self.variables = remove_duplicates(self.variables)

        if len(self.variables) == 0:
            satisfiability = self.check_equation(left_term_list, right_term_list)
            if satisfiability == True:
                return True, self.assignment, left_term_list, right_term_list
            else:
                return BRANCH_CLOSED, self.assignment, left_term_list, right_term_list

    def log_split(self, split_function: Callable[
        [Deque[Term], Deque[Term]], Tuple[bool, Assignment, Deque[Term], Deque[Term]]],
                  left_term_queue: Deque[Term], right_term_queue: Deque[Term], label: str):
        print("branch: ", label)
        before_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue, right_term_queue)
        self.nodes.append(before_process_string_equation)

        return split_function(left_term_queue, right_term_queue)

        after_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue, right_term_queue)
        self.nodes.append(after_process_string_equation)
        self.edges.append((before_process_string_equation, after_process_string_equation, {"label": label}))

    def visualize(self,file_path):
        visualize_path(self.nodes, self.edges,file_path)

