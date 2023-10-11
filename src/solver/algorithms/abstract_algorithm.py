from abc import ABC, abstractmethod
from typing import List,Union,Deque
from src.solver.utils import assemble_parsed_content

from src.solver.DataTypes import Assignment, Term, Terminal, Variable


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

    def visualize(self,file_path:str):
        pass

    def pretty_print_current_equation(self, left_terms: Union[List[Term], Deque[Term]],
                                      right_terms: Union[List[Term], Deque[Term]],mute=True):
        content_dict = {"left_terms": left_terms, "right_terms": right_terms, "terminals": self.terminals,
                        "variables": self.variables}
        string_equation, string_terminals, string_variables = assemble_parsed_content(content_dict)
        # print("string_terminals:",string_terminals)
        #print("string_variables:", string_variables)
        if mute==False:
            print("string_equation:", string_equation)
            print("-" * 10)
        return string_equation, string_terminals, string_variables

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

