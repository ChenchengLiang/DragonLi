import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

from src.solver.Parser import Parser, EqParser, EqReader
from src.solver.Solver import Solver
from src.solver.utils import print_results
from src.solver.algorithms import EnumerateAssignments, EnumerateAssignmentsUsingGenerator, ElimilateVariables, \
    ElimilateVariablesRecursive
from src.solver.DataTypes import Equation
import argparse


def main(args):
    #parse argument
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')

    arg_parser.add_argument('file_path', type=str, help='Path to the file')
    arg_parser.add_argument('branch_method', type=str, choices=['gnn', 'random', 'fixed'],
                        help='Branching method to be used')
    arg_parser.add_argument('--graph_type', type=str, choices=['graph_1', 'graph_2'], default=None,
                        help='Type of graph (optional)')

    args = arg_parser.parse_args()

    # Accessing the arguments
    file_path = args.file_path
    branch_method = args.branch_method
    graph_type = args.graph_type

    print(file_path, branch_method, graph_type)



    #handle parameters
    graph_func_map={None:Equation.get_graph_1,"graph_1":Equation.get_graph_1,"graph_2":Equation.get_graph_2}

    #parse file
    parser_type = EqParser()
    parser = Parser(parser_type)
    parsed_content = parser.parse(file_path)
    print("parsed_content:", parsed_content)

    algorithm_parameters = {"branch_method":branch_method,"graph_type":graph_type,"graph_func":graph_func_map[graph_type]} # branch_method [gnn,random,fixed]

    #solver = Solver(algorithm=SplitEquations,algorithm_parameters=algorithm_parameters)
    solver = Solver(algorithm=ElimilateVariablesRecursive,algorithm_parameters=algorithm_parameters)
    #solver = Solver(algorithm=ElimilateVariables,algorithm_parameters=algorithm_parameters)
    #solver = Solver(EnumerateAssignmentsUsingGenerator, max_variable_length=max_variable_length,algorithm_parameters=algorithm_parameters)
    #solver = Solver(algorithm=EnumerateAssignments,max_variable_length=max_variable_length,algorithm_parameters=algorithm_parameters)
    result_dict = solver.solve(parsed_content, visualize=False,output_train_data=False)

    print_results(result_dict)


if __name__ == '__main__':
    main(sys.argv[1:])
