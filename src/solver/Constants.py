import configparser
from enum import Enum

config = configparser.ConfigParser()
config.read('config.ini')
project_folder = config['Path']['local']
bench_folder = config['Path']['woorpje_benchmarks']
mlflow_folder = config['Path']['mlflow_folder']
checkpoint_folder = config['Path']['checkpoint_folder']
summary_folder = config['Path']['summary_folder']
eval_container_path = config['Path']['eval_container_path']

max_variable_length: int = 8

algorithm_timeout: int = 300
SHELL_TIMEOUT: int = algorithm_timeout


# order equations parameters
HYBRID_ORDER_EQUATION_RATE = 0.5 # higher number less random
HYBRID_BRANCH_RATE = 0.5 # higher number less random
RANDOM_SEED = 42

# control termination when executing algorithm
INITIAL_MAX_DEEP = 500  # 500
MAX_DEEP_STEP = 250
MAX_DEEP = 1000000

RESTART_INITIAL_MAX_DEEP = 20
RESTART_MAX_DEEP_STEP = 1

#extract data parameters
EXTRACT_ONE_PATH = False
MAX_SPLIT_CALL_FOR_TRAIN_DATA_COLLECTION = 1000
MAX_ONE_SIDE_LENGTH = 300
MAX_EQ_LENGTH = MAX_ONE_SIDE_LENGTH * 2



BRANCH_CLOSED: str = "BRANCH_CLOSED"
MAX_PATH: int = 10000000
MAX_PATH_REACHED: str = "MAX_PATH_REACHED"
INTERNAL_TIMEOUT: str = "INTERNAL_TIMEOUT"
RECURSION_DEPTH_EXCEEDED: str = "RECURSION_DEPTH_EXCEEDED"
RECURSION_ERROR: str = "RECURSION_ERROR"
recursion_limit: int = 10000000
OUTPUT_LEAF_NODE_PERCENTAGE = 0.001
GNN_BRANCH_RATIO = 0.5
OUTPUT_NON_SAT_PATH_PERCENTAGE = 0.0005

UNKNOWN: str = "UNKNOWN"
SAT: str = "SAT"
UNSAT: str = "UNSAT"
SUCCESS: str = "SUCCESS"
FAIL: str = "FAIL"
compress_image = True

satisfiability_to_int_label = {SAT: 1, UNSAT: 0, UNKNOWN: -1}
int_label_to_satisfiability = {1: SAT, 0: UNSAT, -1: UNKNOWN}

rank_task_label_size_map={0:2,1:2,2:20,None:-1}
rank_task_node_type_map={0:5,1:5,2:6}
rank_task_node_type_map={0:7,1:7,2:8}
suffix_dict = {"z3": ".smt2","z3-noodler": ".smt2", "woorpje": ".eq", "this": ".eq", "ostrich": ".smt2", "cvc5": ".smt2"}



solver_command_map = {"z3": "/z3/build/z3",
                      "z3-noodler": "/z3-noodler/build/z3",
                      "this": "python3 " + project_folder + "/src/process_benchmarks/main_parameter.py",
                      "woorpje": project_folder + "/other_solvers/woorpje-0_2/bin/woorpje",
                      "ostrich": project_folder + "/other_solvers/ostrich/ostrich",
                      "ostrich_export": project_folder + "/other_solvers/ostrich_export/ostrich",
                      "cvc5": project_folder + "/other_solvers/cvc5/cvc5-Linux"}

# ANSI escape code for red color
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
COLORRESET = "\033[0m"  # Resets the color to default
'''
Syntax:
Formula : Equation | Formula ∧ Formula

Equation : List[Term] = List[Term]
Term : Variable | Terminal 
Variable : v
Terminal : c 

c:str \in letters alphabet


'''
