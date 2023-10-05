from .DataTypes import Terminal

max_variable_length = 100
empty_terminal = Terminal("<EMPTY>")
algorithm_timeout = 20
shell_timeout = 20

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