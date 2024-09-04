import copy
import hashlib
from collections import deque
from typing import Union, List, Tuple, Deque, Callable, Optional, Dict

from src.solver.Constants import UNKNOWN, SAT, UNSAT
from src.solver.independent_utils import remove_duplicates, color_print,int_to_binary_list
from src.solver.visualize_util import draw_graph
import time



class Operator:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return f"Operator({self.value})"

    def __str__(self):
        return "Operator"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, Operator):
            return False
        return self.value == other.value


class Variable:
    def __init__(self, value: str):
        self.value = value
        self.assignment: Optional[List[Terminal]] = None

    def __str__(self):
        return "Variable"

    def __repr__(self):
        return f"Variable({self.value})"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return self.value == other.value


class Terminal:
    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return "Terminal"

    def __repr__(self):
        return f"Terminal({self.value})"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, Terminal):
            return False
        return self.value == other.value


class Symbol:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False

    def __str__(self):
        return self.__class__.__name__


class IsomorphicTailSymbol(Symbol):
    pass  # All necessary methods are inherited from Symbol


class SeparateSymbol(Symbol):
    pass  # All necessary methods are inherited from Symbol

class GlobalVariableOccurrenceSymbol(Symbol):
    pass  # All necessary methods are inherited from Symbol
class GlobalVariableOccurrenceSymbol_0(Symbol):
    pass  # All necessary methods are inherited from Symbol
class GlobalVariableOccurrenceSymbol_1(Symbol):
    pass

class GlobalTerminalOccurrenceSymbol(Symbol):
    pass  # All necessary methods are inherited from Symbol
class GlobalTerminalOccurrenceSymbol_0(Symbol):
    pass  # All necessary methods are inherited from Symbol
class GlobalTerminalOccurrenceSymbol_1(Symbol):
    pass  # All necessary methods are inherited from Symbol



EMPTY_TERMINAL: Terminal = Terminal("\"\"")


class Term:
    def __init__(self, value: Union[Variable, Terminal, List['Term'], SeparateSymbol, IsomorphicTailSymbol]):
        self.value = value

    def __repr__(self):
        return f"Term({self.value})"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        return self.value == other.value

    @property
    def value_type(self):
        if type(self.value) in [Variable, Terminal, list, SeparateSymbol, IsomorphicTailSymbol]:
            return type(self.value)
        else:
            raise Exception("unknown type")

    @property
    def get_value_str(self):
        if isinstance(self.value, Variable):
            return self.value.value
        elif isinstance(self.value, Terminal):
            return self.value.value
        elif isinstance(self.value, list):
            return "".join([t.get_value_str() for t in self.value])
        elif isinstance(self.value, SeparateSymbol):
            return self.value.value
        elif isinstance(self.value, IsomorphicTailSymbol):
            return self.value.value
        else:
            raise Exception("unknown type")


class Equation:
    def __init__(self, left_terms: List[Term], right_terms: List[Term], given_satisfiability: str = UNKNOWN):
        self.left_terms = left_terms
        self.right_terms = right_terms
        self.given_satisfiability = given_satisfiability

    def __repr__(self):
        return f"Equation({self.left_terms}, {self.right_terms})"

    def __hash__(self):
        return hash((tuple(self.left_terms), tuple(self.right_terms)))

    def __eq__(self, other):
        if not isinstance(other, Equation):
            return False
        return (self.left_terms == other.left_terms and self.right_terms == other.right_terms) or (
                self.left_terms == other.right_terms and self.right_terms == other.left_terms)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)


    @property
    def term_list(self) -> List[Term]:
        return self.left_terms + self.right_terms

    @property
    def term_length(self):
        return len(self.term_list)

    @property
    def left_hand_side_length(self):
        return len(self.left_terms)

    @property
    def right_hand_side_length(self):
        return len(self.right_terms)

    @property
    def left_hand_side_type_list(self):
        return [term.value_type for term in self.left_terms]

    @property
    def right_hand_side_type_list(self):
        return [term.value_type for term in self.right_terms]

    @property
    def left_hand_side_string_list(self):
        return [term.get_value_str for term in self.left_terms]
    @property
    def right_hand_side_string_list(self):
        return [term.get_value_str for term in self.right_terms]

    @property
    def variable_list(self) -> List[Variable]:
        return remove_duplicates([item.value for item in self.term_list if isinstance(item.value, Variable)])

    @property
    def variable_str(self) -> str:
        return " ".join([item.value for item in self.variable_list])

    @property
    def variable_number(self) -> int:
        return len(self.variable_list)

    @property
    def terminal_list(self) -> List[Terminal]:
        terminals = remove_duplicates([item.value for item in self.term_list if isinstance(item.value, Terminal)])
        if len(terminals) == 0:
            return [EMPTY_TERMINAL]
        elif EMPTY_TERMINAL in terminals:
            return terminals
        else:
            return terminals + [EMPTY_TERMINAL]

    @property
    def termimal_list_without_empty_terminal(self) -> List[Terminal]:
        return remove_duplicates([item.value for item in self.term_list if isinstance(item.value, Terminal)])

    @property
    def terminal_str(self) -> str:
        return " ".join([x.value for x in self.termimal_list_without_empty_terminal])

    @property
    def terminal_numbers(self) -> int:
        return len(self.terminal_list)

    @property
    def terminal_numbers_without_empty_terminal(self):
        return len(self.termimal_list_without_empty_terminal)


    @property
    def eq_left_str(self) -> str:
        return "".join([t.get_value_str for t in self.left_terms])

    @property
    def eq_right_str(self) -> str:
        return "".join([t.get_value_str for t in self.right_terms])

    @property
    def eq_str(self) -> str:
        return self.eq_left_str + " = " + self.eq_right_str

    @property
    def number_of_special_symbols(self) -> int:
        return self.eq_str.count("#")

    def simplify(self):
        # pop the same prefix
        for index in range(min(len(self.left_terms), len(self.right_terms))):
            if self.left_terms[0] == self.right_terms[0]:
                self.left_terms.pop(0)
                self.right_terms.pop(0)
            else:
                break

    def check_satisfiability_simple(self) -> str:
        if len(self.term_list) == 0:  # both sides are empty
            return SAT
        elif len(self.left_terms) == 0 and len(self.right_terms) > 0:  # left side is empty
            if any(isinstance(term.value, Terminal) for term in self.right_terms):
                return UNSAT
            else:
                return UNKNOWN
        elif len(self.left_terms) > 0 and len(self.right_terms) == 0:  # right side is empty
            if any(isinstance(term.value, Terminal) for term in self.left_terms):
                return UNSAT
            else:
                return UNKNOWN
        else:  # both sides are not empty
            first_left_term = self.left_terms[0]
            first_right_term = self.right_terms[0]
            last_left_term = self.left_terms[-1]
            last_right_term = self.right_terms[-1]
            # all terms are terminals
            if all(isinstance(term.value, Terminal) for term in self.term_list):
                return self.check_both_side_all_terminal_case()
            # mismatch prefix terminal
            elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value != first_right_term.value:
                return UNSAT
            # mistmatch suffix terminal
            elif last_left_term.value_type == Terminal and last_right_term.value_type == Terminal and last_left_term.value != last_right_term.value:
                return UNSAT
            else:
                return UNKNOWN

    def check_both_side_all_terminal_case(self):
        left_str = "".join(
            [t.get_value_str for t in self.left_terms if t.value != EMPTY_TERMINAL])  # ignore empty terminal
        right_str = "".join([t.get_value_str for t in self.right_terms if t.value != EMPTY_TERMINAL])
        if left_str == right_str:
            return SAT
        else:
            return UNSAT

    def visualize_graph(self, file_path, graph_func: Callable):
        nodes, edges = graph_func(self.left_terms, self.right_terms)
        draw_graph(nodes, edges, file_path)

    def output_eq_file_rank(self, file_name, satisfiability=UNKNOWN, answer_file=False):
        left_str_list: List[str] = [x.get_value_str for x in self.left_terms]
        right_str_list: List[str] = [x.get_value_str for x in self.right_terms]

        # Format the content of the file
        content = f"Variables {{{''.join(self.variable_str)}}}\n"
        content += f"Terminals {{{''.join(self.terminal_str)}}}\n"
        left_str = " ".join(left_str_list)
        right_str = " ".join(right_str_list)
        content += f"Equation: {left_str} = {right_str}\n"
        content += "SatGlucose(100)"
        with open(file_name + ".eq", "w") as f:
            f.write(content)
        if answer_file:
            with open(file_name + ".answer", "w") as f:
                f.write(satisfiability)

    def output_eq_file(self, file_name, satisfiability=UNKNOWN, answer_file=False):
        # replaced_v,replaced_eq=replace_primed_vars(self.terminal_str,self.eq_str)

        eq = self.eq_str.split("=")
        left_str_list: List[str] = eq[0].split("#")
        right_str_list: List[str] = eq[1].split("#")

        # Format the content of the file
        content = f"Variables {{{''.join(self.variable_str)}}}\n"
        content += f"Terminals {{{''.join(self.terminal_str)}}}\n"
        for l_str, r_str in zip(left_str_list, right_str_list):
            content += f"Equation: {l_str.strip()} = {r_str.strip()}\n"
        content += "SatGlucose(100)"
        with open(file_name + ".eq", "w") as f:
            f.write(content)
        if answer_file:
            with open(file_name + ".answer", "w") as f:
                f.write(satisfiability)


class Formula:
    def __init__(self, eq_list: List[Equation]):
        self.eq_list = eq_list

        self.sat_equations: List[Equation] = []
        self.unsat_equations: List[Equation] = []
        self.unknown_equations: List[Equation] = []

    def check_satisfiability_2(self) -> str:
        if self.eq_list_length == 0:
            return SAT
        else:
            for eq in self.eq_list:
                satisfiability = eq.check_satisfiability_simple()
                if satisfiability == UNSAT:
                    return UNSAT
            return UNKNOWN

    def simplify_eq_list(self):
        for eq in self.eq_list:
            eq.simplify()
        # simplify empty equation
        self.eq_list = [eq for eq in self.eq_list if eq.term_length > 0]
        #get rid of duplicated equations
        self.eq_list = remove_duplicates(self.eq_list)


    def print_eq_list(self):
        print("Equation list:")
        for index, eq in enumerate(self.eq_list):
            print(index, eq.eq_str)

    @property
    def eq_list_str(self):
        return " | ".join([eq.eq_str for eq in self.eq_list])  # they are conjuncted, use | for easy to read

    @property
    def eq_list_length(self):
        return len(self.eq_list)
    @property
    def formula_size(self):
        total_size=0
        for eq in self.eq_list:
            total_size+=eq.term_length
        return total_size


    @property
    def unknown_number(self) -> int:
        return len(self.unknown_equations)

    @property
    def unsat_number(self) -> int:
        return len(self.unsat_equations)

    @property
    def sat_number(self) -> int:
        return len(self.sat_equations)

    def get_variable_list(self):
        variable_list=[]
        for eq in self.eq_list:
            variable_list.extend(eq.variable_list)
        return remove_duplicates(variable_list)
    def get_terminal_list(self):
        terminal_list=[]
        for eq in self.eq_list:
            terminal_list.extend(eq.termimal_list_without_empty_terminal)
        return remove_duplicates(terminal_list)

    def eq_string_for_file(self):
        variable_list_string=[v.value for v in self.get_variable_list()]
        terminal_list_string=[t.value for t in self.get_terminal_list()]
        eq_list_string = [(eq.left_hand_side_string_list,eq.right_hand_side_string_list)for eq in self.eq_list]

        return formatting_results_v2(variable_list_string, terminal_list_string, eq_list_string)


class Assignment:
    def __init__(self):
        self.assignments = {}  # Dictionary to hold variable assignments

    def __repr__(self):
        return f"Assignment({self.assignments})"

    @property
    def assigned_variables(self) -> List[Variable]:
        """Returns a list of assigned variables."""
        return list(self.assignments.keys())

    def is_empty(self) -> bool:
        """Returns True if the assignment is empty, False otherwise."""
        return len(self.assignments) == 0

    def set_assignment(self, variable: Variable, value: List[Terminal]):
        """Assigns a List[Terminal] to a variable."""
        self.assignments[variable] = value

    def get_assignment(self, variable: Variable) -> List[Terminal]:
        """Returns the assigned List[Terminal] for a variable."""
        return self.assignments[variable]

    def pretty_print(self):
        print("Assignment:")
        for key, value in self.assignments.items():
            print(key.value, "=", "".join([v.value for v in value]))
        print("-" * 10)


class Node:
    def __init__(self, id, type, content, label):
        self.id = id
        self.type = type
        self.content = content
        self.label = label

    def __repr__(self):
        return f"Node({self.id}, {self.label}, {self.type},{self.content})"

    def __hash__(self):
        return hash((self.label, self.type, self.id, self.content))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.label == other.label and self.type == other.type and self.id == other.id and self.content == other.content


class Edge:
    def __init__(self, source, target, type, content, label):
        self.source = source
        self.target = target
        self.type = type
        self.content = content
        self.label = label

    def __repr__(self):
        return f"Edge({self.source}, {self.target},{self.type},{self.content},{self.label})"




def _construct_graph_for_prediction(left_terms: List[Term], right_terms: List[Term], global_info: Dict = {}):
    global_node_counter = 0
    nodes = []
    edges = []

    equation_node,global_node_counter=_add_a_node_no_object(nodes,global_node_counter)
    # Construct left tree
    global_node_counter = _construct_tree_no_object(nodes,edges, left_terms, equation_node, global_node_counter, global_info=global_info)
    # Construct right tree
    global_node_counter = _construct_tree_no_object(nodes, edges, right_terms, equation_node, global_node_counter, global_info=global_info)
    return global_node_counter


def _add_a_node_no_object(nodes,global_node_counter):
    current_node=global_node_counter
    nodes.append(current_node)
    global_node_counter += 1
    return current_node,global_node_counter

def _construct_tree_no_object(nodes, edges,term_list, previous_node, global_node_counter, global_info: Dict = {}):
    for current_term in term_list:
        current_node=global_node_counter
        global_node_counter += 1
        nodes.append(current_node)
        edges.append([previous_node,current_node])

        # add global info
        if global_info == {}:  # no global info
            pass
        else:
            if current_term.value_type == Variable and current_term.value in global_info["variable_global_occurrences"]:

                add_global_variable_start = time.time()
                current_variable_occurrence_node = current_node
                for i in range(global_info["variable_global_occurrences"][current_term.value] - 1):
                    new_variable_occurrence_node = global_node_counter
                    global_node_counter += 1
                    nodes.append(new_variable_occurrence_node)
                    edges.append([new_variable_occurrence_node,current_variable_occurrence_node])
                    current_variable_occurrence_node = new_variable_occurrence_node

                add_global_variable_time = time.time() - add_global_variable_start
                if add_global_variable_time > 0.1:
                    print("add_global_variable_time", add_global_variable_time)
                    print("current_term.value", current_term.value)
                    print("variable_global_occurrences", global_info["variable_global_occurrences"][current_term.value])




            elif current_term.value_type == Terminal and current_term.value in global_info[
                "terminal_global_occurrences"]:

                add_global_terminal_start = time.time()

                current_terminal_occurrence_node = current_node
                for i in range(global_info["terminal_global_occurrences"][current_term.value] - 1):
                    new_terminal_occurrence_node = global_node_counter
                    global_node_counter += 1
                    nodes.append(new_terminal_occurrence_node)
                    edges.append([new_terminal_occurrence_node,current_terminal_occurrence_node])
                    current_terminal_occurrence_node = new_terminal_occurrence_node

                add_global_terminal_time = time.time() - add_global_terminal_start
                if add_global_terminal_time > 0.1:
                    print("add_global_terminal_time", add_global_terminal_time)
                    print("current_term.value", current_term.get_value_str)
                    print("terminal_global_occurrences", global_info["terminal_global_occurrences"][current_term.value])
                    print("nodes length", len(nodes))
                    print("edges length", len(edges))
            else:
                pass

        previous_node = current_node

    return global_node_counter

def _construct_graph(left_terms: List[Term], right_terms: List[Term], graph_type: str, global_info: Dict = {}):
    global_node_counter = 0
    nodes = []
    edges = []
    variable_nodes = []
    terminal_nodes = []

    # Add "=" node
    equation_node, global_node_counter = add_a_node(nodes, global_node_counter, type=Operator, content="=", label=None)

    if graph_type == "graph_3":  # Add variable nodes
        global_node_counter = add_variable_nodes(left_terms, right_terms, nodes, variable_nodes, global_node_counter)
    if graph_type == "graph_4":  # Add terminal nodes
        global_node_counter = add_terminal_nodes(left_terms, right_terms, nodes, terminal_nodes, global_node_counter)
    if graph_type == "graph_5":  # Add add variable and terminal nodes
        global_node_counter = add_variable_nodes(left_terms, right_terms, nodes, variable_nodes, global_node_counter)
        global_node_counter = add_terminal_nodes(left_terms, right_terms, nodes, terminal_nodes, global_node_counter)


    # Construct left tree
    global_node_counter = construct_tree(nodes, edges, graph_type, equation_node, variable_nodes, terminal_nodes,
                                         left_terms, equation_node, global_node_counter, global_info=global_info)
    # Construct right tree
    global_node_counter = construct_tree(nodes, edges, graph_type, equation_node, variable_nodes, terminal_nodes,
                                         right_terms, equation_node, global_node_counter, global_info=global_info)


    return nodes, edges


def add_a_node(nodes, global_node_counter, type, content, label):
    current_node = Node(id=global_node_counter, type=type, content=content, label=label)
    nodes.append(current_node)
    global_node_counter += 1
    return current_node, global_node_counter


def add_variable_nodes(left_terms, right_terms, nodes, variable_nodes, global_node_counter):
    for v in Equation(left_terms, right_terms).variable_list:
        v_node, global_node_counter = add_a_node(nodes, global_node_counter, type=Variable, content=v.value,
                                                 label=None)
        variable_nodes.append(v_node)
    return global_node_counter


def add_terminal_nodes(left_terms, right_terms, nodes, terminal_nodes, global_node_counter):
    for t in Equation(left_terms, right_terms).termimal_list_without_empty_terminal:
        t_node, global_node_counter = add_a_node(nodes, global_node_counter, type=Terminal, content=t.value,
                                                 label=None)
        terminal_nodes.append(t_node)
    return global_node_counter


#
# def construct_tree(nodes, edges, graph_type, equation_node, variable_nodes, terminal_nodes, term_list: Deque[Term],
#                    previous_node: Node, global_node_counter):
#     if len(term_list) == 0:
#         return global_node_counter
#     else:
#         current_term:Term = term_list.popleft()
#         current_node = Node(id=global_node_counter, type=current_term.value_type,
#                             content=current_term.get_value_str, label=None)
#         global_node_counter += 1
#         nodes.append(current_node)
#         edges.append(Edge(source=previous_node.id, target=current_node.id, type=None, content="", label=None))
#         if graph_type == "graph_2":  # add edge back to equation node
#             edges.append(
#                 Edge(source=current_node.id, target=equation_node.id, type=None, content="", label=None))
#         if graph_type == "graph_3":  # add edge to corresponding variable node
#             if current_node.type == Variable:
#                 for v_node in variable_nodes:
#                     if v_node.content == current_node.content:
#                         edges.append(
#                             Edge(source=current_node.id, target=v_node.id, type=None, content="", label=None))
#                         break
#         if graph_type == "graph_4":
#             if current_node.type == Terminal:
#                 for t_node in terminal_nodes:
#                     if t_node.content == current_node.content:
#                         edges.append(
#                             Edge(source=current_node.id, target=t_node.id, type=None, content="", label=None))
#                         break
#         if graph_type == "graph_5":
#             if current_node.type == Variable:
#                 for v_node in variable_nodes:
#                     if v_node.content == current_node.content:
#                         edges.append(
#                             Edge(source=current_node.id, target=v_node.id, type=None, content="", label=None))
#                         break
#             if current_node.type == Terminal:
#                 for t_node in terminal_nodes:
#                     if t_node.content == current_node.content:
#                         edges.append(
#                             Edge(source=current_node.id, target=t_node.id, type=None, content="", label=None))
#                         break
#
#         return construct_tree(nodes, edges, graph_type, equation_node, variable_nodes, terminal_nodes,
#                                        term_list, current_node, global_node_counter)





def construct_tree(nodes, edges, graph_type, equation_node, variable_nodes, terminal_nodes, term_list, previous_node,
                   global_node_counter, global_info: Dict = {}):
    node_type_content_map={GlobalVariableOccurrenceSymbol:"V#",GlobalTerminalOccurrenceSymbol:"T#",
                           GlobalVariableOccurrenceSymbol_0:"V#0",GlobalVariableOccurrenceSymbol_1:"V#1",
                           GlobalTerminalOccurrenceSymbol_0:"T#0",GlobalTerminalOccurrenceSymbol_1:"T#1"}

    for current_term in term_list:
        current_node = Node(id=global_node_counter, type=current_term.value_type,
                            content=current_term.get_value_str, label=None)
        global_node_counter += 1
        nodes.append(current_node)
        edges.append(Edge(source=previous_node.id, target=current_node.id, type=None, content="", label=None))

        # add global info
        if global_info == {}:  # no global info
            pass
        else:
            if current_term.value_type == Variable and current_term.value in global_info["variable_global_occurrences"]:

                # generate binary nodes
                binary_list = int_to_binary_list(global_info["variable_global_occurrences"][current_term.value])
                current_variable_occurrence_node = current_node
                for i in binary_list:
                    node_type = GlobalVariableOccurrenceSymbol_0 if i == 0 else GlobalVariableOccurrenceSymbol_1
                    new_variable_occurrence_node = Node(id=global_node_counter, type=node_type,
                                                        content=node_type_content_map[node_type], label=None)
                    global_node_counter += 1
                    nodes.append(new_variable_occurrence_node)
                    edges.append(
                        Edge(source=new_variable_occurrence_node.id, target=current_variable_occurrence_node.id,
                             type=None, content="", label=None))
                    current_variable_occurrence_node = new_variable_occurrence_node

                # # unary nodes
                # current_variable_occurrence_node = current_node
                # for i in range(global_info["variable_global_occurrences"][current_term.value] - 1):
                #     new_variable_occurrence_node = Node(id=global_node_counter, type=GlobalVariableOccurrenceSymbol,
                #                                         content="V#", label=None)
                #     global_node_counter += 1
                #     nodes.append(new_variable_occurrence_node)
                #     edges.append(
                #         Edge(source=new_variable_occurrence_node.id, target=current_variable_occurrence_node.id,
                #              type=None, content="", label=None))
                #     current_variable_occurrence_node = new_variable_occurrence_node



            elif current_term.value_type == Terminal and current_term.value in global_info[
                "terminal_global_occurrences"]:


                # generate binary nodes
                binary_list = int_to_binary_list(global_info["terminal_global_occurrences"][current_term.value])
                current_terminal_occurrence_node = current_node
                for i in binary_list:
                    node_type = GlobalTerminalOccurrenceSymbol_0 if i == 0 else GlobalTerminalOccurrenceSymbol_1
                    new_terminal_occurrence_node = Node(id=global_node_counter, type=node_type,
                                                        content=node_type_content_map[node_type], label=None)
                    global_node_counter += 1
                    nodes.append(new_terminal_occurrence_node)
                    edges.append(
                        Edge(source=new_terminal_occurrence_node.id, target=current_terminal_occurrence_node.id,
                             type=None, content="", label=None))
                    current_terminal_occurrence_node = new_terminal_occurrence_node


                # # unary nodes
                # current_terminal_occurrence_node = current_node
                # for i in range(global_info["terminal_global_occurrences"][current_term.value] - 1):
                #     new_terminal_occurrence_node = Node(id=global_node_counter, type=GlobalTerminalOccurrenceSymbol,
                #                                         content="T#", label=None)
                #     global_node_counter += 1
                #     nodes.append(new_terminal_occurrence_node)
                #     edges.append(
                #         Edge(source=new_terminal_occurrence_node.id, target=current_terminal_occurrence_node.id,
                #              type=None, content="", label=None))
                #     current_terminal_occurrence_node = new_terminal_occurrence_node
            else:
                pass

        if graph_type == "graph_2" and current_node.type != SeparateSymbol and current_node.type != IsomorphicTailSymbol:  # add edge back to equation node
            edges.append(Edge(source=current_node.id, target=equation_node.id, type=None, content="", label=None))

        if graph_type in ["graph_3", "graph_5"] and current_node.type == Variable:
            for v_node in variable_nodes:
                if v_node.content == current_node.content:
                    edges.append(Edge(source=current_node.id, target=v_node.id, type=None, content="", label=None))
                    # edges.append(Edge(source=v_node.id, target=current_node.id, type=None, content="", label=None))
                    break

        if graph_type in ["graph_4", "graph_5"] and current_node.type == Terminal:
            for t_node in terminal_nodes:
                if t_node.content == current_node.content:
                    edges.append(Edge(source=current_node.id, target=t_node.id, type=None, content="", label=None))
                    # edges.append(Edge(source=t_node.id, target=current_node.id, type=None, content="", label=None))
                    break

        previous_node = current_node

    return global_node_counter


def get_eq_graph_1(left_terms: List[Term], right_terms: List[Term], global_info: Dict = {}):
    return _construct_graph(left_terms, right_terms, graph_type="graph_1", global_info=global_info)


def get_eq_graph_2(left_terms: List[Term], right_terms: List[Term],
                   global_info: Dict = {}):  # add edge back to equation node
    return _construct_graph(left_terms, right_terms, graph_type="graph_2", global_info=global_info)


def get_eq_graph_3(left_terms: List[Term], right_terms: List[Term],
                   global_info: Dict = {}):  # add edge to corresponding variable nodes
    return _construct_graph(left_terms, right_terms, graph_type="graph_3", global_info=global_info)


def get_eq_graph_4(left_terms: List[Term], right_terms: List[Term],
                   global_info: Dict = {}):  # add edge to corresponding terminal nodes
    return _construct_graph(left_terms, right_terms, graph_type="graph_4", global_info=global_info)


def get_eq_graph_5(left_terms: List[Term],
                   right_terms: List[Term],
                   global_info: Dict = {}):  # add edge to corresponding variable and terminal nodes
    return _construct_graph(left_terms, right_terms, graph_type="graph_5", global_info=global_info)


def _update_term_in_eq_list(eq_list: List[Equation], old_term: Term, new_term: List[Term]) -> List[Equation]:
    new_eq_list = []
    for eq_in_formula in eq_list:
        new_left = _update_term_list(old_term, new_term, eq_in_formula.left_terms)
        new_right = _update_term_list(old_term, new_term, eq_in_formula.right_terms)
        new_eq_list.append(Equation(new_left, new_right))
    return new_eq_list


def _update_term_list(old_term: Term, new_term: List[Term], term_list: List[Term]) -> List[Term]:
    new_term_list = []
    for t in term_list:
        if t == old_term:
            for new_t in new_term:
                new_term_list.append(new_t)
        else:
            new_term_list.append(t)
    return new_term_list

def formatting_results(variables: List[str], terminals: List[str], eq_list: List[Tuple[str, str]]) -> str:
    # Format the result
    result = f"Variables {{{''.join(variables)}}}\n"
    result += f"Terminals {{{''.join(terminals)}}}\n"
    # for eq in eq_list:
    #     result += f"Equation: {eq[0]} = {eq[1]}\n"
    for eq in eq_list:
        result += f"Equation: {''.join(eq[0])} = {''.join(eq[1])}\n"
    result += "SatGlucose(100)"
    return result


def formatting_results_v2(variables: List[str], terminals: List[str],
                          eq_list: List[Tuple[List[str], List[str]]]) -> str:
    if len(variables) > 26 or len(terminals) > 26:
        joint_space=" "
    else:
        joint_space=""

    # Format the result
    result = f"Variables {{{joint_space.join(variables)}}}\n"
    result += f"Terminals {{{joint_space.join(terminals)}}}\n"
    for eq in eq_list:
        result += f"Equation: {joint_space.join(eq[0])} = {joint_space.join(eq[1])}\n"
    result += "SatGlucose(100)"
    return result
