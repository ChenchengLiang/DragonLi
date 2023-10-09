from abc import ABC, abstractmethod
from itertools import product
from .DataTypes import Assignment, Term, Terminal, Variable
from typing import List, Dict, Tuple, Generator, Deque, Union
from collections import deque
from .utils import flatten_list, assemble_parsed_content, remove_duplicates
from .Constants import EMPTY_TERMINAL, BRANCH_CLOSED, MAX_PATH, MAX_PATH_REACHED
from .visualize_util import visualize_path
import random


class AbstractAlgorithm(ABC):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], left_terms: List[Term],
                 right_terms: List[Term]):
        self.terminals = terminals
        self.variables = variables
        self.left_terms = left_terms.copy()
        self.right_terms = right_terms.copy()

    @abstractmethod
    def run(self):
        pass

    def visualize(self):
        pass

    def check_equation(self, left_terms: List[Term], right_terms: List[Term],
                       assignment: Assignment = Assignment()) -> bool:
        left_side = self.extract_values_from_terms(left_terms, assignment)
        right_side = self.extract_values_from_terms(right_terms, assignment)

        # todo: this need to be improved
        left_str = "".join(left_side).replace("<EMPTY>", "")
        right_str = "".join(right_side).replace("<EMPTY>", "")
        if left_str == right_str:
            return True
        else:
            return False

    def extract_values_from_terms(self, term_list, assignments):
        value_list = []
        for t in term_list:
            if type(t.value) == Variable:
                terminal_list = assignments.get_assignment(t.value)
                for tt in terminal_list:
                    value_list.append(tt.value)
            else:  # type(t.value) == Terminal
                value_list.append(t.value.value)
        return value_list


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
                return {"result": result, "assignment": assignment, "left_terms": left_term_queue,
                        "right_terms": right_term_queue,
                        "variables": self.variables, "terminals": self.terminals}

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
            else: # first_left_term.value != first_right_term.value
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
                        return BRANCH_CLOSED, self.assignment, left_term_queue, right_term_queue #todo check if this can be replaced to return False (UNSAT)

            path_depth += 1
            print("path_depth: ", path_depth)
            # print("length", len(left_term_queue), len(right_term_queue))

        # one side empty
        if len(left_term_queue) == 0 and len(right_term_queue) == 0:
            return True, Assignment(), left_term_queue, right_term_queue
        # two side empty
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

        # self.left_variable_larger_than_right_variable(left_terms, right_terms)
        # self.left_variable_smaller_than_right_variable(left_terms, right_terms)
        # self.left_variable_equal_right_variable(left_terms, right_terms)

    def split_equation_one_variable(self, left_terms: deque, right_terms: deque):
        print("split_equation_one_variable")
        strategies = [self.left_variable_empty, self.left_variable_not_empty]
        # Define the probabilities of selecting each strategy
        probabilities = [0.5, 0.5]  # Adjust these probabilities as needed

        # Use random.choices() to select a function with specified probabilities
        selected_strategy = random.choices(strategies, probabilities)[0]
        selected_strategy(left_terms, right_terms)

        # self.left_variable_empty(left_term, right_term)
        # self.left_variable_not_empty(left_terms, right_terms)

    def left_variable_larger_than_right_variable(self, left_term_queue: Deque[Term], right_term_queue: Deque[Term]):
        print("branch: left_variable > or < right_variable")
        before_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue, right_term_queue)
        self.nodes.append(before_process_string_equation)

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
        # self.variables.append(split_term.value)
        # if right_term.value not in [v for v in left_term_queue+left_term_queue]:
        #     self.variables.remove(right_term.value)
        # if left_term.value not in [v for v in left_term_queue+left_term_queue]:
        #     self.variables.remove(left_term.value)

        left_term_queue.clear()
        right_term_queue.clear()
        left_term_queue.extend(left_term_list)
        right_term_queue.extend(right_term_list)

        after_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue, right_term_queue)
        self.nodes.append(after_process_string_equation)
        self.edges.append((before_process_string_equation, after_process_string_equation, {"label": "V1 != V2"}))

    def left_variable_smaller_than_right_variable(self, left_term_queue: deque, right_term_queue: deque):
        self.left_variable_larger_than_right_variable(right_term_queue, left_term_queue)

    def left_variable_equal_right_variable(self, left_term_queue, right_term_queue):
        print("branch: left_variable = right_variable")
        before_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue, right_term_queue)
        left_term = left_term_queue.popleft()
        right_term = right_term_queue.popleft()

        # replace left_term variable with right_term variable
        self.replace_a_term(left_term, right_term, left_term_queue)
        self.replace_a_term(left_term, right_term, right_term_queue)

        self.update_variable_list(left_term_queue, right_term_queue)
        after_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue, right_term_queue)
        self.edges.append(
            (before_process_string_equation, after_process_string_equation, {"label": "V1 == V2"}))

    def left_variable_empty(self, left_term_queue: Deque[Term], right_term_queue: Deque[Term]):
        print("branch: left_variable_empty")
        before_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue, right_term_queue)
        left_term = left_term_queue.popleft()

        self.replace_a_term(left_term, Term(EMPTY_TERMINAL), left_term_queue)
        self.replace_a_term(left_term, Term(EMPTY_TERMINAL), right_term_queue)
        self.remove_empty_terminals(left_term_queue)
        self.remove_empty_terminals(right_term_queue)

        self.update_variable_list(left_term_queue, right_term_queue)
        after_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue, right_term_queue)
        self.edges.append(
            (before_process_string_equation, after_process_string_equation, {"label": "V1 == \"\""}))

    def left_variable_not_empty(self, left_term_queue: deque, right_term_queue: deque):
        '''
        replace X with "Terminal X_NEW"
        '''
        print("branch: left_variable_not_empty")
        before_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue, right_term_queue)

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

        after_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue, right_term_queue)
        self.edges.append(
            (before_process_string_equation, after_process_string_equation, {"label": "V1 != \"\""}))

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
            if type(left_term.value) == Terminal and type(right_term.value) == Terminal:#example: a T2 T3 ... = a T5 T6 ...
                if left_term.value == right_term.value:
                    local_left_term_queue.popleft()
                    local_right_term_queue.popleft()
                else:
                    return False, self.assignment, local_left_term_queue, local_right_term_queue
            elif type(left_term.value) == Variable and type(right_term.value) == Variable: #example: V1 T2 T3 ... = V1 T5 T6 ...
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

    def update_variable_list(self, left_term_list: Union[List[Term],Deque[Term]], right_term_list: Union[List[Term],Deque[Term]]):
        left_term_list=list(left_term_list)
        right_term_list=list(right_term_list)
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

    def log_split(self, left_term_queue: Deque[Term], right_term_queue: Deque[Term]):
        print("branch: left_variable_not_empty")
        before_process_string_equation, _, _ = self.pretty_print_current_equation(left_term_queue, right_term_queue)


    def pretty_print_current_equation(self, left_terms: Union[List[Term], Deque[Term]],
                                      right_terms: Union[List[Term], Deque[Term]]):
        content_dict = {"left_terms": left_terms, "right_terms": right_terms, "terminals": self.terminals,
                        "variables": self.variables}
        string_equation, string_terminals, string_variables = assemble_parsed_content(content_dict)
        # print("string_terminals:",string_terminals)
        print("string_variables:", string_variables)
        print("string_equation:", string_equation)
        print("-" * 10)
        return string_equation, string_terminals, string_variables

    def visualize(self):
        visualize_path(self.nodes, self.edges)


class EnumerateAssignmentsUsingGenerator(AbstractAlgorithm):
    def __init__(self, terminals, variables, left_terms, right_terms, parameters: Dict):
        super().__init__(terminals, variables, left_terms, right_terms)
        self.max_variable_length = parameters["max_variable_length"]

    def generate_possible_terminal_combinations(self, terminals: List[str], max_length: int) -> Generator[
        Tuple[str, ...], None, None]:
        for length in range(1, max_length + 1):
            for p in product(terminals, repeat=length):
                yield p

    def generate_assignments(self, variables, terminals, max_variable_length):
        possible_terminals = self.generate_possible_terminal_combinations(terminals, max_variable_length)

        # Generate all possible combinations of assignments
        assignments_generator = product(possible_terminals, repeat=len(variables))

        for assignment in assignments_generator:
            assignment_dict = Assignment()
            for var, term in zip(variables, assignment):
                assignment_dict.set_assignment(var, list(term))
            yield assignment_dict

    def run(self):
        assignment_generator = self.generate_assignments(self.variables, self.terminals, self.max_variable_length)

        # Check each assignment dictionary to see if it satisfies the equation
        for assignment in assignment_generator:
            if self.check_equation(self.left_terms, self.right_terms, assignment):
                return {"result": True, "assignment": assignment, "left_terms": self.left_terms,
                        "right_terms": self.right_terms,
                        "variables": self.variables, "terminals": self.terminals}

        return {"result": "max_variable_length_exceeded", "assignment": assignment,
                "left_terms": self.left_terms, "right_terms": self.right_terms, "variables": self.variables,
                "terminals": self.terminals}


class EnumerateAssignments(AbstractAlgorithm):
    def __init__(self, terminals, variables, left_terms, right_terms, parameters):
        super().__init__(terminals, variables, left_terms, right_terms)
        self.max_variable_length = parameters["max_variable_length"]

    def generate_possible_terminal_combinations(self, terminals: List[Terminal], max_length: int) -> List[
        Tuple[Terminal]]:
        combinations:List[Tuple[Terminal]] = []
        for length in range(1, max_length + 1):
            for p in product(terminals, repeat=length):
                combinations.append(p)  # type: ignore
        return combinations

    def run(self):
        possible_terminals = self.generate_possible_terminal_combinations(self.terminals, self.max_variable_length)
        print("possible_terminals", len(possible_terminals))
        print(possible_terminals)
        # Generate all possible combinations of assignments
        assignments_list = list(product(possible_terminals, repeat=len(self.variables)))
        print("assignments_list:", len(assignments_list))

        # Create a list of dictionaries to represent each assignment
        assignment_dicts = []
        for assignment in assignments_list:
            assignment_dict = Assignment()
            for var, term in zip(self.variables, assignment):
                assignment_dict.set_assignment(var, list(term))
            assignment_dicts.append(assignment_dict)

        # Display the list of assignment dictionaries
        print("-" * 10)
        print("Assignment Dictionaries:", len(assignment_dicts))
        for assignment_dict in assignment_dicts:
            print(assignment_dict.assignments)

        # Check each assignment dictionary to see if it satisfies the equation
        for assignment in assignment_dicts:
            if self.check_equation(self.left_terms, self.right_terms, assignment):
                return {"result": True, "assignment": assignment, "left_terms": self.left_terms,
                        "right_terms": self.right_terms,
                        "variables": self.variables, "terminals": self.terminals}

        return {"result": "max_variable_length_exceeded", "assignment": assignment,
                "left_terms": self.left_terms, "right_terms": self.right_terms, "variables": self.variables,
                "terminals": self.terminals}
