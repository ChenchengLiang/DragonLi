from .DataTypes import Terminal

max_variable_length = 8
EMPTY_TERMINAL = Terminal("\"\"")
algorithm_timeout = 20
shell_timeout = 20
BRANCH_CLOSED= "BRANCH_CLOSED"
MAX_PATH=3
MAX_PATH_REACHED="MAX_PATH_REACHED"

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