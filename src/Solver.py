
class Solver:
    def __init__(self, variables, terminals, left_terms, right_terms):
        self.variables = variables
        self.terminals = terminals
        self.left_terms = left_terms
        self.right_terms = right_terms

    def solve(self):
        assignments = {}

        for lt, rt in zip(self.left_terms, self.right_terms):
            if lt == rt:
                continue
            elif lt in self.variables:
                if lt in assignments and assignments[lt] != rt:
                    return False, {}
                assignments[lt] = rt
            elif rt in self.variables:
                if rt in assignments and assignments[rt] != lt:
                    return False, {}
                assignments[rt] = lt
            else:
                return False, {}

        return True, assignments

