
import os
import shutil
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

import glob
import sys
from src.solver.Constants import project_folder
sys.path.append(project_folder)
from src.solver.Parser import Parser, EqParser, EqReader
from src.solver.Solver import Solver
from src.solver.utils import print_results,graph_func_map
from src.solver.algorithms import EnumerateAssignments,EnumerateAssignmentsUsingGenerator,ElimilateVariables,ElimilateVariablesRecursive,SplitEquations
from src.solver.Constants import max_variable_length, algorithm_timeout
from src.solver.DataTypes import Equation
from src.solver.Constants import project_folder,bench_folder,UNKNOWN,SAT,UNSAT
from src.solver.independent_utils import strip_file_name_suffix
from src.train_data_collection.utils import output_one_eq_graph
def main():

    benchmark="test_track"#"01_track_generated_SAT_train"
    algorithm_parameters = {"branch_method": "extract_branching_data_task_2"} #extract_branching_data_task_2

    #prepare train folder
    all_eq_folder = bench_folder + "/" + benchmark + "/ALL/ALL"
    train_eq_folder=bench_folder + "/" + benchmark+"/train"

    if not os.path.exists(train_eq_folder):
        os.mkdir(train_eq_folder)
    else:
        shutil.rmtree(train_eq_folder)
        os.mkdir(train_eq_folder)
    for f in glob.glob(all_eq_folder + "/*.eq") + glob.glob(all_eq_folder + "/*.answer"):
        shutil.copy(f, train_eq_folder)

    # extract train data
    for i, file_path in enumerate(glob.glob(train_eq_folder + "/*.eq")):
        file_name = strip_file_name_suffix(file_path)
        print(i, file_path)

        # read file answer
        with open(file_name + ".answer", "r") as f:
            satisfiability = f.read()
        if satisfiability == SAT or satisfiability == UNSAT:
            parser_type = EqParser()
            parser = Parser(parser_type)
            parsed_content = parser.parse(file_path)
            #print("parsed_content:", parsed_content)

            # solver = Solver(algorithm=SplitEquations,algorithm_parameters=algorithm_parameters)
            solver = Solver(algorithm=ElimilateVariablesRecursive, algorithm_parameters=algorithm_parameters)

            result_dict = solver.solve(parsed_content, visualize=False, output_train_data=True)

            #print_results(result_dict)


if __name__ == '__main__':
    main()
