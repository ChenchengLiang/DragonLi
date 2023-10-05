from .DataTypes import Terminal

max_variable_length = 8
empty_terminal = Terminal("\"\"")
algorithm_timeout = 20
shell_timeout = 20
branch_closed="BRANCH_CLOSED"

'''
Syntax:
Formula : Equation | Formula ∧ Formula
-----------------------------------------------

Equation : List[Term] = List[Term]
Term : Variable | Terminal | List[Term]
Variable : v
Terminal : c 

c:str \in letters alphabet


'''