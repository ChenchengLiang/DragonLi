from typing import Union, List, Tuple, Deque
from src.solver.Constants import UNKNOWN, SAT, UNSAT, satisfiability_to_int_label
from src.solver.independent_utils import remove_duplicates
from collections import deque
from src.solver.visualize_util import draw_graph


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


class Operator:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return f"Operator({self.value})"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, Operator):
            return False
        return self.value == other.value


class Variable:
    def __init__(self, value: str):
        self.value = value
        self.assignment: List[Terminal] = None

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

    def __repr__(self):
        return f"Terminal({self.value})"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, Terminal):
            return False
        return self.value == other.value


EMPTY_TERMINAL: Terminal = Terminal("\"\"")


class Term:
    def __init__(self, value: Union[Variable, Terminal, List['Term']]):
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
        if isinstance(self.value, Variable):
            return Variable
        elif isinstance(self.value, Terminal):
            return Terminal
        elif isinstance(self.value, list):
            return list
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
        else:
            raise Exception("unknown type")


class Equation:
    def __init__(self, left_terms: List[Term], right_terms: List[Term]):
        self.left_terms = left_terms
        self.right_terms = right_terms

    def __repr__(self):
        return f"Equation({self.left_terms}, {self.right_terms})"

    def __hash__(self):
        return hash((tuple(self.left_terms), tuple(self.right_terms)))

    def __eq__(self, other):
        if not isinstance(other, Equation):
            return False
        return self.left_terms == other.left_terms and self.right_terms == other.right_terms

    @property
    def term_list(self) -> List[Term]:
        return self.left_terms + self.right_terms

    @property
    def variable_list(self) -> List[Variable]:
        return remove_duplicates([item.value for item in self.term_list if isinstance(item.value, Variable)])

    @property
    def variable_numbers(self) -> int:
        return len(self.variable_list)

    @property
    def terminal_list(self) -> List[Terminal]:
        terminals = remove_duplicates([item.value for item in self.term_list if isinstance(item.value, Terminal)])
        if len(terminals) == 0:
            return [EMPTY_TERMINAL]
        else:
            return terminals + [EMPTY_TERMINAL]

    @property
    def terminal_numbers(self) -> int:
        return len(self.terminal_list)

    @property
    def eq_str(self) -> str:
        return "".join([t.get_value_str for t in self.left_terms]) + " = " + "".join(
            [t.get_value_str for t in self.right_terms])

    def is_fact(self) -> (bool, List[Tuple[Variable, List[Terminal]]]):

        # Condition: "" = List[Variable]
        if len(self.left_terms) == 0 and len(self.right_terms) > 0:  # left side is empty
            if all(isinstance(term.value, Variable) for term in
                   self.right_terms):  # if all right hand side are variables
                return True, [(term.value, [EMPTY_TERMINAL]) for term in self.right_terms]
        # Condition: List[Variable] = ""
        elif len(self.left_terms) > 0 and len(self.right_terms) == 0:  # right side is empty
            if all(isinstance(term.value, Variable) for term in self.left_terms):  # if all left hand side are variables
                return True, [(term.value, [EMPTY_TERMINAL]) for term in self.left_terms]
        # Condition: A=AA
        elif len(self.left_terms) > 0 and len(self.right_terms) > 0 and len(self.left_terms) != len(
                self.right_terms) and self.variable_numbers == 1 and self.terminal_numbers <= 1:
            return True, [(self.variable_list[0], [EMPTY_TERMINAL])]
        # Condition: Variable=List[Terminal]
        elif len(self.left_terms) == 1 and isinstance(self.left_terms[0].value, Variable) and all(
                isinstance(term.value, Terminal) for term in self.right_terms):
            return True, [(self.left_terms[0].value, [t.value for t in self.right_terms])]
        # Condition: List[Terminal]=Variable
        elif len(self.right_terms) == 1 and isinstance(self.right_terms[0].value, Variable) and all(
                isinstance(term.value, Terminal) for term in self.left_terms):
            return True, [(self.right_terms[0].value, [t.value for t in self.left_terms])]
        else:
            return False, []

    def check_satisfiability(self) -> str:
        if len(self.term_list) == 0:  # both sides are empty
            return SAT
        elif len(self.left_terms) == 0 and len(self.right_terms) > 0:  # left side is empty
            return self.satisfiability_one_side_empty(self.right_terms)
        elif len(self.left_terms) > 0 and len(self.right_terms) == 0:  # right side is empty
            return self.satisfiability_one_side_empty(self.left_terms)
        else:  # both sides are not empty
            # both sides are exatcly the same
            if self.left_terms == self.right_terms:
                return SAT
            elif all(isinstance(term.value, Variable) for term in self.term_list):  # if all terms are variables
                result, _ = self.is_fact()
                return SAT if result == True else UNKNOWN
            elif all(isinstance(term.value, Terminal) for term in self.term_list):  # if all terms are terminals
                return self.check_all_terminal_case()
            else:
                result, _ = self.is_fact()
                return SAT if result == True else UNKNOWN

    def satisfiability_one_side_empty(self, not_empty_side: List[Term]) -> str:
        '''
        Assume another side is empty.
        there are three conditions for one side: (1). terminals + variables (2). only terminals (3). only variables
        '''
        # (1) + (2): if there are any Terminal in the not_empty_side, then it is UNSAT
        if any(isinstance(term.value, Terminal) for term in not_empty_side):
            return UNSAT
        # (3): if there are only Variables in the not_empty_side
        else:
            result, _ = self.is_fact()
            return SAT if result == True else UNKNOWN

    def check_all_terminal_case(self):
        left_str = "".join(
            [t.get_value_str for t in self.left_terms if t.value != EMPTY_TERMINAL])  # ignore empty terminal
        right_str = "".join([t.get_value_str for t in self.right_terms if t.value != EMPTY_TERMINAL])
        if left_str == right_str:
            return SAT
        else:
            return UNSAT

    def graph_to_gnn_format(self, nodes: List[Node], edges: List[Edge], satisfiability: str=UNKNOWN):
        '''
        output format:
        {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2],
        "edges": [[1, 2], [2, 3], [3, 0]], "edge_types": [1, 1, 1],
        "label": 1}
        '''
        node_type_to_int_map = {Operator: 0, Terminal: 1, Variable: 2}
        edge_type_to_int_map = {None: 1}
        graph_dict = {"nodes": [], "node_types": [], "edges": [], "edge_types": [],
                      "label": satisfiability_to_int_label[satisfiability]}
        for node in nodes:
            graph_dict["nodes"].append(node.id)
            graph_dict["node_types"].append(node_type_to_int_map[node.type])
        for edge in edges:
            graph_dict["edges"].append([edge.source, edge.target])
            graph_dict["edge_types"].append(edge_type_to_int_map[edge.type])

        return graph_dict

    def visualize_graph(self, file_path):
        nodes, edges = self.get_graph_1()
        draw_graph(nodes, edges, file_path)

    def get_graph_1(self):
        global_node_counter = 0
        nodes = []
        edges = []

        def construct_tree(term_list: Deque[Term], previous_node: Node, global_node_counter):
            if len(term_list) == 0:
                return global_node_counter
            else:
                current_term = term_list.popleft()
                current_node = Node(id=global_node_counter, type=current_term.value_type,
                                    content=current_term.get_value_str, label=None)
                global_node_counter += 1
                nodes.append(current_node)
                edges.append(Edge(source=previous_node.id, target=current_node.id, type=None, content="", label=None))
                return construct_tree(term_list, current_node, global_node_counter)

        # Add "=" node
        equation_node = Node(id=global_node_counter, type=Operator, content="=", label=None)
        nodes.append(equation_node)
        global_node_counter += 1

        local_left_terms = deque(self.left_terms.copy())
        local_right_terms = deque(self.right_terms.copy())

        global_node_counter = construct_tree(local_left_terms, equation_node, global_node_counter)
        global_node_counter = construct_tree(local_right_terms, equation_node, global_node_counter)

        return nodes, edges


class Formula:
    def __init__(self, eq_list: List[Equation]):
        self.formula = eq_list
        self.facts = []
        self.sat_equations = []
        self.unsat_equations = []
        self.unknown_equations = []
        for eq in self.formula:
            satisfiability = eq.check_satisfiability()
            if satisfiability == SAT:
                self.sat_equations.append(eq)
                is_fact, fact_assignment = eq.is_fact()
                if is_fact:
                    self.facts.append((eq, fact_assignment))
            elif satisfiability == UNSAT:
                self.unsat_equations.append(eq)
            else:
                self.unknown_equations.append(eq)

    @property
    def fact_number(self) -> int:
        return len(self.facts)

    @property
    def unknown_number(self) -> int:
        return len(self.unknown_equations)

    @property
    def unsat_number(self) -> int:
        return len(self.unsat_equations)

    @property
    def satisfiability(self) -> str:
        if self.unknown_number == 0:
            if self.unsat_number == 0:
                return SAT
            else:
                return UNSAT
        else:
            return UNKNOWN


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
