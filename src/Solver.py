from typing import Dict, List, Set
from DataTypes import Variable, Terminal, Term


class Solver:
    def __init__(self):
        pass

    def solve(self, string_equation: Dict) -> bool:
        variables: Set[Variable] = string_equation["variables"]
        terminals: Set[Terminal] = string_equation["terminals"]
        left_terms: List[Term] = string_equation["left_terms"]
        right_terms: List[Term] = string_equation["right_terms"]
        assignments = []

        if len(left_terms) != len(right_terms): # If the number of terms on the left and right sides are not equal, then the equation is unsat
            return False, {}
        else:
            for lt, rt in zip(left_terms, right_terms):
                l=lt.value
                r=rt.value
                if type(l) == Terminal and type(r)==Terminal:
                    if l != r:
                        return False, {}
                    else:
                        continue
                elif type(l) == Variable and type(r)==Terminal:
                    if l in assignments:
                        if assignments[l] != r:
                            return False, {}

                elif type(lt.value) == Terminal and type(rt.value)==Variable:
                    pass

                elif type(lt.value) == Variable and type(rt.value)==Variable:
                    pass
                




        return True, assignments
