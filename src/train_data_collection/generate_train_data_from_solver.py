
import os
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
from src.solver.utils import print_results
from src.solver.algorithms import EnumerateAssignments,EnumerateAssignmentsUsingGenerator,ElimilateVariables,ElimilateVariablesRecursive,SplitEquations
from src.solver.Constants import max_variable_length, algorithm_timeout
from src.solver.DataTypes import Equation
from src.solver.Constants import project_folder,bench_folder,UNKNOWN,SAT,UNSAT
from src.solver.independent_utils import strip_file_name_suffix
def main():

    for file_path in glob.glob(bench_folder+"/01_track_generated_train_data_sat_with_some_leafs/graph_1/*.eq"):
        file_name=strip_file_name_suffix(file_path)
        print(file_path)

        #read file answer
        with open(file_name + ".answer", "r") as f:
            satisfiability = f.read()
        if satisfiability == SAT or satisfiability==UNSAT:

            parser_type = EqParser()
            parser = Parser(parser_type)
            parsed_content = parser.parse(file_path)
            print("parsed_content:", parsed_content)

            algorithm_parameters = {"branch_method":"extract_branching_data","graph_type":"graph_1","graph_func":Equation.get_graph_1} # branch_method [gnn.random,fixed]

            #solver = Solver(algorithm=SplitEquations,algorithm_parameters=algorithm_parameters)
            solver = Solver(algorithm=ElimilateVariablesRecursive,algorithm_parameters=algorithm_parameters)

            result_dict = solver.solve(parsed_content,visualize=False,output_train_data=True)

            print_results(result_dict)


if __name__ == '__main__':
    main()
