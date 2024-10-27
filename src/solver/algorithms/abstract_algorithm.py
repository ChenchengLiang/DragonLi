from abc import ABC, abstractmethod
from typing import List, Union, Deque, Callable, Tuple, Dict

from src.solver.DataTypes import Assignment, Term, Terminal, Variable, Equation, EMPTY_TERMINAL, Formula
from src.solver.utils import assemble_one_equation, get_variable_string, get_terminal_string


class AbstractAlgorithm(ABC):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation]):
        self.terminals = terminals
        self.variables = variables
        self.equation_list = equation_list
        self.gnn_call_flag = False

        self.post_process_ordered_formula_func_map = {"None":self._post_process_ordered_formula_none,"quadratic":self._post_process_ordered_formula_quadratic_pattern}
        self.post_process_ordered_formula_func=self.post_process_ordered_formula_func_map["None"]

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

    def record_node_and_edges(self, f: Formula, previous_node: Tuple[int, Dict], edge_label: str,depth:int) -> \
            Tuple[int, Dict]:
        current_node_number = self.total_node_number
        label = f"{f.eq_list_str}"
        current_node = (
            current_node_number,
            {"label": label, "status": None, "output_to_file": False,
             "shape": "ellipse", "back_track_count": 0,"gnn_call":False,"depth":depth})
        self.nodes.append(current_node)
        self.edges.append((previous_node[0], current_node_number, {'label': edge_label}))
        self.total_node_number+=1
        return current_node

    def order_equations_func_wrapper(self, f: Formula,current_node:Tuple[int,Dict]) -> Formula:
        if f.eq_list_length > 1:
            self.total_rank_call += 1
            f.unsat_core= self.unsat_core
            f.current_total_split_eq_call=self.total_split_eq_call
            ordered_formula, category_call = self.order_equations_func(f, self.total_category_call)
            self.total_category_call = category_call
            if self.gnn_call_flag == True:
                current_node[1]["gnn_call"] = True
                self.gnn_call_flag = False

            ordered_formula=self.post_process_ordered_formula_func(ordered_formula)
            return ordered_formula
        else:
            return f

    def get_first_eq(self, f: Formula) -> Tuple[Equation, Formula]:
        return f.eq_list[0], Formula(f.eq_list[1:])

    def _post_process_ordered_formula_none(self, f: Formula) -> Formula:
        return f

    def _post_process_ordered_formula_quadratic_pattern(self, f: Formula) -> Formula:
        loop_eq=[]
        non_loop_eq=[]

        for i,current_eq in enumerate(f.eq_list):
            #print(f"post process i:{i}/{f_length}")



            first_left_term = current_eq.left_terms[0]
            first_right_term = current_eq.right_terms[0]
            first_left_term_occurrence_in_left_terms = current_eq.left_terms.count(first_left_term)
            first_left_term_occurrence_in_right_terms = current_eq.right_terms.count(first_left_term)
            first_right_term_occurrence_in_left_terms = current_eq.left_terms.count(first_right_term)
            first_right_term_occurrence_in_right_terms = current_eq.right_terms.count(first_right_term)

            last_left_term = current_eq.left_terms[-1]
            last_right_term = current_eq.right_terms[-1]
            last_left_term_occurrence_in_left_terms = current_eq.left_terms.count(last_left_term)
            last_left_term_occurrence_in_right_terms = current_eq.right_terms.count(last_left_term)
            last_right_term_occurrence_in_left_terms = current_eq.left_terms.count(last_right_term)
            last_right_term_occurrence_in_right_terms = current_eq.right_terms.count(last_right_term)

            #print(current_eq.eq_str_pretty)
            #print(first_right_term.get_value_str,first_right_term_occurrence_in_left_terms,first_right_term_occurrence_in_right_terms)


            def prefix_loop():
                if (first_left_term.value_type==Variable and first_right_term.value_type==Terminal and first_left_term_occurrence_in_left_terms<=first_left_term_occurrence_in_right_terms) or (first_left_term.value_type==Terminal and first_right_term.value_type==Variable and first_right_term_occurrence_in_right_terms<=first_right_term_occurrence_in_left_terms) or (first_left_term.value_type == Variable and first_right_term.value_type == Variable and first_left_term!=first_right_term and (first_left_term_occurrence_in_left_terms <= first_left_term_occurrence_in_right_terms or first_right_term_occurrence_in_right_terms<=first_right_term_occurrence_in_left_terms)):
                    #print("prefix_loop")
                    return True
                else:
                    return False
            def suffix_loop():
                if (last_left_term.value_type==Variable and last_right_term.value_type==Terminal and last_left_term_occurrence_in_left_terms<=last_left_term_occurrence_in_right_terms) or (last_left_term.value_type==Terminal and last_right_term.value_type==Variable and last_right_term_occurrence_in_right_terms<=last_right_term_occurrence_in_left_terms) or (last_left_term.value_type == Variable and last_right_term.value_type == Variable and last_left_term!=last_right_term and (last_left_term_occurrence_in_left_terms <= last_left_term_occurrence_in_right_terms or last_right_term_occurrence_in_right_terms<=last_right_term_occurrence_in_left_terms)):
                    #print("suffix_loop")
                    return True
                else:
                    return False

            if prefix_loop() and suffix_loop():
                loop_eq.append(current_eq)
            else:
                non_loop_eq.append(current_eq)

        return Formula(non_loop_eq+loop_eq)
