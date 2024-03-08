import copy
from collections import deque
from typing import Union, List, Tuple, Deque, Callable, Optional

from src.solver.Constants import UNKNOWN, SAT, UNSAT
from src.solver.independent_utils import remove_duplicates, color_print
from src.solver.visualize_util import draw_graph


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


class SeparateSymbol:
    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return "SeparateSymbol"

    def __repr__(self):
        return f"SeparateSymbol({self.value})"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, SeparateSymbol):
            return False
        return self.value == other.value


EMPTY_TERMINAL: Terminal = Terminal("\"\"")


class Term:
    def __init__(self, value: Union[Variable, Terminal, SeparateSymbol, List['Term']]):
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
        elif isinstance(self.value, SeparateSymbol):
            return SeparateSymbol
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
    def variable_list(self) -> List[Variable]:
        return remove_duplicates([item.value for item in self.term_list if isinstance(item.value, Variable)])

    @property
    def variable_str(self) -> str:
        return "".join([item.value for item in self.variable_list])

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
        return "".join([x.value for x in self.termimal_list_without_empty_terminal])

    @property
    def terminal_numbers(self) -> int:
        return len(self.terminal_list)

    @property
    def terminal_numbers_without_empty_terminal(self):
        return len(self.termimal_list_without_empty_terminal)

    @property
    def eq_str(self) -> str:
        return "".join([t.get_value_str for t in self.left_terms]) + " = " + "".join(
            [t.get_value_str for t in self.right_terms])

    @property
    def eq_left_str(self) -> str:
        return "".join([t.get_value_str for t in self.left_terms])

    @property
    def eq_right_str(self) -> str:
        return "".join([t.get_value_str for t in self.right_terms])

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
                self.right_terms) and self.variable_number == 1 and self.terminal_numbers <= 1:
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
        else: # both sides are not empty
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
            #mistmatch suffix terminal
            elif last_left_term.value_type == Terminal and last_right_term.value_type == Terminal and last_left_term.value != last_right_term.value:
                return UNSAT
            else:
                return UNKNOWN


    def check_satisfiability(self) -> str:
        if len(self.term_list) == 0:  # both sides are empty
            return SAT
        elif len(self.left_terms) == 0 and len(self.right_terms) > 0:  # left side is empty
            return self.satisfiability_one_side_empty(self.right_terms)
        elif len(self.left_terms) > 0 and len(self.right_terms) == 0:  # right side is empty
            return self.satisfiability_one_side_empty(self.left_terms)
        else:  # both sides are not empty
            first_left_term = self.left_terms[0]
            first_right_term = self.right_terms[0]
            last_left_term = self.left_terms[-1]
            last_right_term = self.right_terms[-1]

            # both sides are exatcly the same
            if self.left_terms == self.right_terms:
                return SAT
            # all terms are variables
            elif all(isinstance(term.value, Variable) for term in self.term_list):
                result, _ = self.is_fact()
                return SAT if result == True else UNKNOWN
            # all terms are terminals
            elif all(isinstance(term.value, Terminal) for term in self.term_list):
                return self.check_both_side_all_terminal_case()
            # mismatch prefix terminal
            elif first_left_term.value_type == Terminal and first_right_term.value_type == Terminal and first_left_term.value != first_right_term.value:
                return UNSAT
            # mistmatch suffix terminal
            elif last_left_term.value_type == Terminal and last_right_term.value_type == Terminal and last_left_term.value != last_right_term.value:
                return UNSAT
            # todo trivial other conditions
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

    def output_eq_file(self, file_name, satisfiability=UNKNOWN):
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
        with open(file_name + ".answer", "w") as f:
            f.write(satisfiability)


class Formula:
    def __init__(self, eq_list: List[Equation]):
        self.eq_list = eq_list

        self.facts: List[Tuple[Equation, Tuple[Variable, List[Terminal]]]] = []
        self.sat_equations: List[Equation] = []
        self.unsat_equations: List[Equation] = []
        self.unknown_equations: List[Equation] = []

    def categorize_equations_1(self):
        self.sat_equations = []
        self.unsat_equations = []
        self.unknown_equations = []
        for eq in self.eq_list:
            satisfiability = eq.check_satisfiability_simple()
            #print(eq.eq_str,satisfiability)
            if satisfiability == SAT:
                self.sat_equations.append(eq)
            elif satisfiability == UNSAT:
                self.unsat_equations.append(eq)
            else:
                self.unknown_equations.append(eq)

    def categorize_equations(self):
        self.facts = []
        self.sat_equations = []
        self.unsat_equations = []
        self.unknown_equations = []

        # check satisfiability for each equation
        for eq in self.eq_list:
            satisfiability = eq.check_satisfiability()
            # color_print(f"{satisfiability},{eq.eq_str}", "green" )

            if satisfiability == SAT:
                self.sat_equations.append(eq)
                is_fact, fact_assignment = eq.is_fact()
                if is_fact:
                    for fact in fact_assignment:
                        self.facts.append((eq, fact))

            elif satisfiability == UNSAT:
                self.unsat_equations.append(eq)
            else:
                self.unknown_equations.append(eq)

    def propagate_facts(self):
        propagate_count = 0
        while True:
            previous_fact_eq_list = [ff[0] for ff in self.facts]
            self.simplify_eq_list()
            self.categorize_equations()
            current_fact_eq_list = [ff[0] for ff in self.facts]

            if current_fact_eq_list == previous_fact_eq_list:
                # print("No new facts are found")
                break

            # print("Propagate facts")
            propagate_count += 1
            current_eq_list = list(self.eq_list)

            for fact in self.facts:
                fact_eq: Equation = fact[0]
                fact_variable: Variable = fact[1][0]
                fact_terminal_list: List[Terminal] = fact[1][1]

                if fact_eq in current_eq_list:
                    eq_list_without_fact = list(current_eq_list)
                    eq_list_without_fact.remove(fact_eq)

                    updated_eq_list_without_fact = _update_term_in_eq_list(eq_list_without_fact, Term(fact_variable),
                                                                           [Term(t) for t in fact_terminal_list])
                    current_eq_list = updated_eq_list_without_fact + [fact_eq]

            self.eq_list = list(current_eq_list)

        print(f"Propagate facts {propagate_count} times")

    def check_satisfiability_2(self) -> str:
        if self.eq_list_length==0:
            return SAT
        else:
            for eq in self.eq_list:
                satisfiability=eq.check_satisfiability_simple()
                if satisfiability==UNSAT:
                    return UNSAT
            return UNKNOWN


    def check_satisfiability_1(self) -> str:
        self.categorize_equations_1()
        if self.eq_list_length==0:
            return SAT
        else:
            if self.unsat_number != 0:
                return UNSAT
            else:
                return UNKNOWN


    def check_satisfiability(
            self) -> str:  # todo this require to check the relation between the equations, is done in propagate_facts
        if self.unknown_number == 0:
            if self.unsat_number == 0:
                return SAT
            else:
                return UNSAT
        else:
            return UNKNOWN

    def simplify_eq_list(self):
        for eq in self.eq_list:
            eq.simplify()

    def print_eq_list(self):
        for index, eq in enumerate(self.eq_list):
            print(index, eq.eq_str)

    @property
    def eq_list_str(self):
        return " | ".join([eq.eq_str for eq in self.eq_list])  # they are conjuncted, use | for easy to read

    @property
    def eq_list_length(self):
        return len(self.eq_list)

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
    def sat_number(self) -> int:
        return len(self.sat_equations)


#
# class Formula:
#     def __init__(self, eq_list: List[Equation]):
#         self.formula = eq_list
#         self.facts: List[Tuple[Equation, List[Tuple[Variable, List[Terminal]]]]] = []
#         self.sat_equations: List[Equation] = []
#         self.unsat_equations: List[Equation] = []
#         self.unknown_equations: List[Equation] = []
#         for eq in self.formula:
#             satisfiability = eq.check_satisfiability()
#             if satisfiability == SAT:
#                 self.sat_equations.append(eq)
#                 is_fact, fact_assignment = eq.is_fact()
#                 if is_fact:
#                     self.facts.append((eq, fact_assignment))
#                     # for f in fact_assignment:
#                     #     self.facts.append((eq, f))
#             elif satisfiability == UNSAT:
#                 self.unsat_equations.append(eq)
#             else:
#                 self.unknown_equations.append(eq)
#
#     @property
#     def fact_number(self) -> int:
#         return len(self.facts)
#
#     @property
#     def unknown_number(self) -> int:
#         return len(self.unknown_equations)
#
#     @property
#     def unsat_number(self) -> int:
#         return len(self.unsat_equations)
#
#     @property
#     def satisfiability(self) -> str:
#         if self.unknown_number == 0:
#             if self.unsat_number == 0:
#                 return SAT
#             else:
#                 return UNSAT
#         else:
#             return UNKNOWN
#
#     def print_eq_list(self):
#         for eq in self.formula:
#             print(eq.eq_str)


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


def _construct_graph(left_terms: List[Term], right_terms: List[Term], graph_type: str):
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

    local_left_terms = deque(left_terms.copy())
    local_right_terms = deque(right_terms.copy())

    global_node_counter = construct_tree(nodes, edges, graph_type, equation_node, variable_nodes, terminal_nodes,
                                         local_left_terms, equation_node, global_node_counter)
    global_node_counter = construct_tree(nodes, edges, graph_type, equation_node, variable_nodes, terminal_nodes,
                                         local_right_terms, equation_node, global_node_counter)

    return nodes, edges


def add_a_node(nodes, global_node_counter, type, content, label):
    current_node = Node(id=global_node_counter, type=type, content=content, label=label)
    nodes.append(current_node)
    global_node_counter += 1
    return current_node, global_node_counter


def add_variable_nodes(left_terms, right_terms, nodes, variable_nodes, global_node_counter):
    for v in Equation(left_terms.copy(), right_terms.copy()).variable_list:
        v_node, global_node_counter = add_a_node(nodes, global_node_counter, type=Variable, content=v.value,
                                                 label=None)
        variable_nodes.append(v_node)
    return global_node_counter


def add_terminal_nodes(left_terms, right_terms, nodes, terminal_nodes, global_node_counter):
    for t in Equation(left_terms.copy(), right_terms.copy()).termimal_list_without_empty_terminal:
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
                   global_node_counter):
    while len(term_list) > 0:
        current_term = term_list.popleft()
        current_node = Node(id=global_node_counter, type=current_term.value_type,
                            content=current_term.get_value_str, label=None)
        global_node_counter += 1
        nodes.append(current_node)
        edges.append(Edge(source=previous_node.id, target=current_node.id, type=None, content="", label=None))

        if graph_type == "graph_2" and current_node.type != SeparateSymbol:  # add edge back to equation node
            edges.append(Edge(source=current_node.id, target=equation_node.id, type=None, content="", label=None))

        if graph_type in ["graph_3", "graph_5"] and current_node.type == Variable:
            for v_node in variable_nodes:
                if v_node.content == current_node.content:
                    edges.append(Edge(source=current_node.id, target=v_node.id, type=None, content="", label=None))
                    break

        if graph_type in ["graph_4", "graph_5"] and current_node.type == Terminal:
            for t_node in terminal_nodes:
                if t_node.content == current_node.content:
                    edges.append(Edge(source=current_node.id, target=t_node.id, type=None, content="", label=None))
                    break

        previous_node = current_node

    return global_node_counter


def get_eq_graph_1(left_terms: List[Term], right_terms: List[Term]):
    return _construct_graph(left_terms, right_terms, graph_type="graph_1")


def get_eq_graph_2(left_terms: List[Term], right_terms: List[Term]):  # add edge back to equation node
    return _construct_graph(left_terms, right_terms, graph_type="graph_2")


def get_eq_graph_3(left_terms: List[Term], right_terms: List[Term]):  # add edge to corresponding variable nodes
    return _construct_graph(left_terms, right_terms, graph_type="graph_3")


def get_eq_graph_4(left_terms: List[Term], right_terms: List[Term]):  # add edge to corresponding terminal nodes
    return _construct_graph(left_terms, right_terms, graph_type="graph_4")


def get_eq_graph_5(left_terms: List[Term],
                   right_terms: List[Term]):  # add edge to corresponding variable and terminal nodes
    return _construct_graph(left_terms, right_terms, graph_type="graph_5")


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
