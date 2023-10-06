from abc import ABC, abstractmethod
from itertools import product
from .DataTypes import Assignment, Term, Terminal, Variable
from typing import List, Dict, Tuple, Generator
from collections import deque
from .utils import flatten_list, assemble_parsed_content, remove_duplicates
from .Constants import empty_terminal, branch_closed
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

    def check_equation(self, left_terms: List[Term], right_terms: List[Term], assignments: Assignment) -> bool:
        left_side = self.extract_values_from_terms(left_terms, assignments)
        right_side = self.extract_values_from_terms(right_terms, assignments)

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

    def run(self):

        path_count=0
        while True:
            print("-"*10, "path ",path_count," start", "-"*10)
            result, assignment, left_term_queue, right_term_queue = self.run_one_path()
            print("path result:", result)
            print("-"*10, "path ",path_count," end", "-"*10)
            path_count += 1

            if result == True or result == False:
                return {"result":result, "assignment":assignment,"left_terms":left_term_queue, "right_terms":right_term_queue,
                        "variables":self.variables, "terminals":self.terminals}

    def run_one_path(self):
        # create local variables
        left_term_queue = deque(self.left_terms)
        right_term_queue = deque(self.right_terms)
        path_depth = 0

        while len(left_term_queue) != 0 and len(right_term_queue) != 0:  # while two sides are not empty
            first_left_term = left_term_queue[0]  # unwrap first left hand term
            if type(first_left_term.value) == Variable:
                first_right_term = right_term_queue[0]  # unwrap first right hand term
                if type(first_right_term.value) == Variable:  # example: V1 T2 T3 ... = V4 T5 T6 ...
                    # split equation
                    self.split_equation_two_variables(left_term_queue, right_term_queue)

                elif type(first_right_term.value) == Terminal:  # example: V1 T2 T3 ... = a T5 T6 ...
                    # split equation
                    self.split_equation_one_variable(left_term_queue, right_term_queue)
            elif type(first_left_term.value) == Terminal:
                first_right_term = right_term_queue[0]  # unwrap first right hand term
                if type(first_right_term.value) == Variable:  # example: a T2 T3 ... = V4 T5 T6 ...
                    # split equation
                    self.split_equation_one_variable(right_term_queue, left_term_queue)
                elif type(first_right_term.value) == Terminal:  # example: a T2 T3 ... = a|b T5 T6 ...
                    if first_left_term.value == first_right_term.value:  # example: a T2 T3 ... = a T5 T6 ..., dischard both terms
                        l_term=left_term_queue.popleft()
                        r_term=right_term_queue.popleft()
                        print("discard", l_term, "from both side")
                    else:  # example: a T2 T3 ... = b T5 T6 ..., UNSAT
                        return branch_closed, self.assignment, left_term_queue, right_term_queue

            path_depth += 1
            print("path_depth: ", path_depth)
            #print("length", len(left_term_queue), len(right_term_queue))

        if len(left_term_queue) == 0 and len(right_term_queue) == 0:
            return True, Assignment(), left_term_queue, right_term_queue
        if len(left_term_queue) == 0 and len(right_term_queue) != 0:
            result, _ =self.left_terms_empty(left_term_queue, right_term_queue)
            return result, self.assignment, left_term_queue, right_term_queue
        if len(left_term_queue) != 0 and len(right_term_queue) == 0:
            result,_ = self.right_terms_empty(left_term_queue, right_term_queue)
            return result,self.assignment, left_term_queue, right_term_queue

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

    def left_variable_larger_than_right_variable(self, left_term_queue: deque, right_term_queue: deque):
        print("branch: left_variable > or < right_variable")
        self.pretty_print_current_equation(left_term_queue, right_term_queue)

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
        self.update_variable_list(left_term_list + right_term_list)
        # self.variables.append(split_term.value)
        # if right_term.value not in [v for v in left_term_queue+left_term_queue]:
        #     self.variables.remove(right_term.value)
        # if left_term.value not in [v for v in left_term_queue+left_term_queue]:
        #     self.variables.remove(left_term.value)

        left_term_queue.clear()
        right_term_queue.clear()
        left_term_queue.extend(left_term_list)
        right_term_queue.extend(right_term_list)

        self.pretty_print_current_equation(left_term_queue, right_term_queue)

    def left_variable_smaller_than_right_variable(self, left_term_queue: deque, right_term_queue: deque):
        self.left_variable_larger_than_right_variable(right_term_queue, left_term_queue)

    def left_variable_equal_right_variable(self, left_term_queue, right_term_queue):
        print("branch: left_variable = right_variable")
        self.pretty_print_current_equation(left_term_queue, right_term_queue)
        left_term = left_term_queue.popleft()
        right_term = right_term_queue.popleft()
        if left_term.value == right_term.value:
            pass
        elif left_term.value != right_term.value:
            # replace left_term variable with right_term variable
            self.replace_a_term(left_term, right_term, left_term_queue)
            self.replace_a_term(left_term, right_term, right_term_queue)

        self.update_variable_list(left_term_queue + right_term_queue)
        self.pretty_print_current_equation(left_term_queue, right_term_queue)

    def left_variable_empty(self, left_term_queue: deque, right_term_queue: deque):
        print("branch: left_variable_empty")
        self.pretty_print_current_equation(left_term_queue, right_term_queue)
        left_term = left_term_queue.popleft()

        self.replace_a_term(left_term, Term(empty_terminal), left_term_queue)
        self.replace_a_term(left_term, Term(empty_terminal), right_term_queue)

        self.update_variable_list(left_term_queue + right_term_queue)
        self.pretty_print_current_equation(left_term_queue, right_term_queue)

    def left_variable_not_empty(self, left_term_queue: deque, right_term_queue: deque):
        print("branch: left_variable_not_empty")
        self.pretty_print_current_equation(left_term_queue, right_term_queue)

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
        self.update_variable_list(left_term_list + right_term_list)

        left_term_queue.clear()
        right_term_queue.clear()
        left_term_queue.extend(left_term_list)
        right_term_queue.extend(right_term_list)

        self.pretty_print_current_equation(left_term_queue, right_term_queue)

    def left_terms_empty(self, left_term_queue: deque, right_term_queue: deque):
        right_term_queue_contains_terminal = any(isinstance(item.value, Terminal) for item in right_term_queue)
        if right_term_queue_contains_terminal == True:
            return branch_closed, self.assignment
        else:
            # assign all variables in right_term_queue to empty
            for x in self.variables:
                self.assignment.set_assignment(x, [empty_terminal])
            return True, self.assignment

    def right_terms_empty(self, left_term_queue: deque, right_term_queue: deque):
        return self.left_terms_empty(right_term_queue, left_term_queue)

    def replace_a_term(self, old_term: Term, new_term: Term, term_queue: deque):
        for i, t in enumerate(term_queue):
            if t.value == old_term.value:
                term_queue[i] = new_term

    def update_variable_list(self, term_list):
        self.variables = []
        for t in term_list:
            if type(t.value) == Variable:
                self.variables.append(t.value)
        self.variables = remove_duplicates(self.variables)

    def pretty_print_current_equation(self, left_terms: List[Term], right_terms: List[Term]):
        content_dict = {"left_terms": left_terms, "right_terms": right_terms, "terminals": self.terminals,
                        "variables": self.variables}
        string_equation, string_terminals, string_variables = assemble_parsed_content(content_dict)
        # print("string_terminals:",string_terminals)
        print("string_variables:", string_variables)
        print("string_equation:", string_equation)
        print("-" * 10)


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
                return {"result":True, "assignment":assignment,"left_terms":self.left_terms,"right_terms":self.right_terms,
                        "variables":self.variables, "terminals":self.terminals}

        return {"result":"max_variable_length_exceeded", "assignment":assignment,
                "left_terms":self.left_terms,"right_terms":self.right_terms,"variables":self.variables, "terminals":self.terminals}


class EnumerateAssignments(AbstractAlgorithm):
    def __init__(self, terminals, variables, left_terms, right_terms, parameters):
        super().__init__(terminals, variables, left_terms, right_terms)
        self.max_variable_length = parameters["max_variable_length"]

    def generate_possible_terminal_combinations(self, terminals: List[Terminal], max_length: int) -> List[
        Tuple[Terminal]]:
        combinations = []
        for length in range(1, max_length + 1):
            for p in product(terminals, repeat=length):
                combinations.append(p)
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
                return {"result":True, "assignment":assignment,"left_terms":self.left_terms,"right_terms":self.right_terms,
                        "variables":self.variables, "terminals":self.terminals}

        return {"result":"max_variable_length_exceeded", "assignment":assignment,
                "left_terms":self.left_terms,"right_terms":self.right_terms,"variables":self.variables, "terminals":self.terminals}

