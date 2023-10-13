from typing import Union, List
from Constants import UNKNOWN, SAT, UNSAT



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

EMPTY_TERMINAL:Terminal = Terminal("\"\"")

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
    def variable_numbers(self) -> int:
        return len(self.variable_list)

    @property
    def variable_list(self) -> List[Variable]:
        return [item.value for item in self.term_list if isinstance(item.value, Variable)]

    @property
    def terminal_list(self) -> List[Terminal]:
        return [item.value for item in self.term_list if isinstance(item.value, Terminal)]

    @property
    def is_fact(self) -> bool:
        # Check if left side has a single Variable and right side has only Terminals or is empty
        print(self.eq_str)
        if len(self.left_terms) == 1 and isinstance(self.left_terms[0].value, Variable):
            return all(isinstance(term.value, Terminal) for term in self.right_terms)

        # Check if right side has a single Variable and left side has only Terminals or is empty
        elif len(self.right_terms) == 1 and isinstance(self.right_terms[0].value, Variable):
            return all(isinstance(term.value, Terminal) for term in self.left_terms)
        # If neither condition is met, it's not a fact
        else:
            return False

    @property
    def eq_str(self) -> str:
        return "".join([t.get_value_str for t in self.left_terms]) + " = " + "".join(
            [t.get_value_str for t in self.right_terms])

    def satisfiability(self):
        if len(self.term_list) == 0:
            return SAT
        elif len(self.left_terms) == 0 and len(self.right_terms) > 0:
            pass
        elif len(self.left_terms) > 0 and len(self.right_terms) == 0:
            pass
        else:
            pass

    def satisfiability_one_side_empty(self,not_empty_side:List[Term]):
        '''
        This assume another side is empty.
        there are three conditions for one side: (1). terminals + variables (2). only terminals (3). only variables

        '''
        #(1) + (2): if there are any Terminal in the not_empty_side, then it is UNSAT
        if any(isinstance(term.value, Terminal) for term in not_empty_side):
            return UNSAT
        #(3): if there are only Variables in the not_empty_side
        else:
            return SAT if self.is_fact else UNKNOWN


class Assignment:
    def __init__(self):
        self.assignments = {}  # Dictionary to hold variable assignments

    def __repr__(self):
        return f"Assignment({self.assignments})"

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
