from typing import Dict, List, Set
import configparser



config = configparser.ConfigParser()
config.read('config.ini')
project_folder = config['Path']['local']
bench_folder = config['Path']['woorpje_benchmarks']
max_variable_length: int = 8
algorithm_timeout: int = 30
shell_timeout: int = algorithm_timeout
MAX_DEEP=100
MAX_SPLIT_CALL=10
MAX_EQ_LENGTH=300
MAX_ONE_SIDE_LENGTH=150
BRANCH_CLOSED: str = "BRANCH_CLOSED"
MAX_PATH: int = 10000000
MAX_PATH_REACHED: str = "MAX_PATH_REACHED"
INTERNAL_TIMEOUT: str = "INTERNAL_TIMEOUT"
RECURSION_DEPTH_EXCEEDED: str = "RECURSION_DEPTH_EXCEEDED"
RECURSION_ERROR: str = "RECURSION_ERROR"
recursion_limit: int = 10000000
OUTPUT_LEAF_NODE_PERCENTAGE=0.01
GNN_BRANCH_RATIO=0.5
UNKNOWN = "UNKNOWN"
SAT = "SAT"
UNSAT = "UNSAT"

satisfiability_to_int_label = {SAT: 1, UNSAT: 0, UNKNOWN: -1}
int_label_to_satisfiability = {1: SAT, 0: UNSAT, -1: UNKNOWN}

solver_command_map = {"z3": "z3",
                      "this": "python3 "+project_folder+"/src/process_benchmarks/main_parameter.py",
                      "woorpje": project_folder+"/other_solvers/woorpje-0_2/bin/woorpje",
                      "ostrich": "/home/cheli243/Desktop/CodeToGit/ostrich-fork-master/ostrich/ostrich",
                      "cvc5": project_folder+"/other_solvers/cvc5/cvc5-Linux"}

'''
Syntax:
Formula : Equation | Formula ∧ Formula

Equation : List[Term] = List[Term]
Term : Variable | Terminal 
Variable : v
Terminal : c 

c:str \in letters alphabet


'''
