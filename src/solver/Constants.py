from .DataTypes import Terminal

max_variable_length = 5
empty_terminal = Terminal("<EMPTY>")
algorithm_timeout = 5
shell_timeout = 5

'''
Syntax:
Formula : Equation | Formula ∧ Formula
-----------------------------------------------

Equation : List[Term] = List[Term]
Term : Variable | Terminal | List[Term]
Variable : List[Terminal]
Terminal : c | <EMPTY>

c:str \in letters alphabet


'''