from .DataTypes import Terminal
from typing import Dict, List, Set
max_variable_length:int = 8
EMPTY_TERMINAL:Terminal = Terminal("\"\"")
algorithm_timeout:int = 30
shell_timeout:int = 30
BRANCH_CLOSED:str = "BRANCH_CLOSED"
MAX_PATH:int = 10000000
MAX_PATH_REACHED:str = "MAX_PATH_REACHED"

branch_name_map:Dict = {"BRANCH_CLOSED": "Branch closed"}

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
