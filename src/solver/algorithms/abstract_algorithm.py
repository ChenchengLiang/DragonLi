from abc import ABC, abstractmethod
from typing import List, Union, Deque, Callable, Tuple, Dict

from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, EMPTY_TERMINAL, Formula
from src.solver.utils import assemble_one_equation, get_variable_string, get_terminal_string


class AbstractAlgorithm(ABC):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation]):
        self.terminals = terminals
        self.variables = variables
        self.equation_list = equation_list

    @abstractmethod
    def run(self):
        pass

    def visualize(self, file_path: str, graph_func: Callable):
        pass

    def output_train_data(self, file_path: str):
        pass

    def pretty_print_current_equation(self, left_terms: Union[List[Term], Deque[Term]],
                                      right_terms: Union[List[Term], Deque[Term]], mute=True):

        string_equation = assemble_one_equation(left_terms, right_terms, Assignment())
        string_terminals = get_terminal_string(self.terminals)
        string_variables = get_variable_string(self.variables)
        # print("string_terminals:",string_terminals)
        # print("string_variables:", string_variables)
        if mute == False:
            print("string_equation:", string_equation)
            print("-" * 10)
        return string_equation, string_terminals, string_variables

    def check_equation(self, left_terms: List[Term], right_terms: List[Term],
                       assignment: Assignment = Assignment()) -> bool:
        left_side = self.extract_values_from_terms(left_terms, assignment)
        right_side = self.extract_values_from_terms(right_terms, assignment)

        left_str = "".join(left_side)
        right_str = "".join(right_side)
        if left_str == right_str:
            return True
        else:
            return False

    def extract_values_from_terms(self, term_list, assignments):
        value_list = []
        for t in [t for t in term_list if t.value != EMPTY_TERMINAL]: #ignore empty terminal
            if type(t.value) == Variable:
                terminal_list = assignments.get_assignment(t.value)
                for tt in terminal_list:
                    value_list.append(tt.value)
            else:  # type(t.value) == Terminal
                value_list.append(t.get_value_str)
        return value_list


    def record_eq_node_and_edges(self, eq: Equation, previous_node: Tuple[int, Dict], edge_label: str) -> Tuple[int, Dict]:
        current_node_number = self.total_node_number
        label = f"{eq.eq_str}"
        current_node = (
            current_node_number,
            {"label": label, "status": None, "output_to_file": False, "shape": "box", "back_track_count": 0,"gnn_call":False})
        self.nodes.append(current_node)
        self.edges.append((previous_node[0], current_node_number, {'label': edge_label}))
        self.eq_node_number+=1
        self.total_node_number+=1
        return current_node

    def record_node_and_edges(self, f: Formula, previous_node: Tuple[int, Dict], edge_label: str) -> \
            Tuple[int, Dict]:
        current_node_number = self.total_node_number
        label = f"{f.eq_list_str}"
        current_node = (
            current_node_number,
            {"label": label, "status": None, "output_to_file": False, "shape": "ellipse", "back_track_count": 0,"gnn_call":False})
        self.nodes.append(current_node)
        self.edges.append((previous_node[0], current_node_number, {'label': edge_label}))
        self.total_node_number+=1
        return current_node
