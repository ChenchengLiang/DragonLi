from .DataTypes import Terminal
from typing import Dict, List, Set
max_variable_length:int = 8
EMPTY_TERMINAL:Terminal = Terminal("\"\"")
algorithm_timeout:int = 30
shell_timeout:int = 30
BRANCH_CLOSED:str = "BRANCH_CLOSED"
MAX_PATH:int = 10000000
MAX_PATH_REACHED:str = "MAX_PATH_REACHED"
INTERNAL_TIMEOUT:str = "INTERNAL_TIMEOUT"
RECURSION_DEPTH_EXCEEDED:str = "RECURSION_DEPTH_EXCEEDED"
RECURSION_ERROR:str = "RECURSION_ERROR"
recursion_limit:int = 10000

solver_command_map={"z3":"z3",
                    "this":"python3 /home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/process_benchmarks/main_parameter.py",
                    "woorpje":"/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/other_solvers/woorpje-0_2/bin/woorpje",
                    "ostrich":"/home/cheli243/Desktop/CodeToGit/ostrich-fork-master/ostrich/ostrich",
                    "cvc5":"/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/other_solvers/cvc5/cvc5-Linux"}


'''
Syntax:
Formula : Equation | Formula ∧ Formula

Equation : List[Term] = List[Term]
Term : Variable | Terminal 
Variable : v
Terminal : c 

c:str \in letters alphabet


'''
