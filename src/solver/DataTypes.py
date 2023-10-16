from typing import Union, List, Tuple
from .Constants import UNKNOWN, SAT, UNSAT
from .independent_utils import remove_duplicates


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
            return terminals

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
        elif len(self.left_terms) == 1 and isinstance(self.left_terms[0].value, Variable):
            if all(isinstance(term.value, Terminal) for term in self.right_terms):
                return True, [(self.left_terms[0].value, [t.value for t in self.right_terms])]
        # Condition: List[Terminal]=Variable
        elif len(self.right_terms) == 1 and isinstance(self.right_terms[0].value, Variable):
            if all(isinstance(term.value, Terminal) for term in self.left_terms):
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
        left_str = "".join([t.get_value_str for t in self.left_terms if t.value != EMPTY_TERMINAL]) #ignore empty terminal
        right_str = "".join([t.get_value_str for t in self.right_terms if t.value != EMPTY_TERMINAL])
        if left_str == right_str:
            return SAT
        else:
            return UNSAT



class EquationChain:
    def __init__(self, equation: Equation):
        self.equation_chain = [equation]
        self.transformation_chain = []




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
        print("-"*10)
