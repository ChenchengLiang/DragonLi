import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

from src.solver.Constants import bench_folder
from src.solver.Parser import Parser, EqParser, EqReader
from src.solver.Solver import Solver
from src.solver.utils import print_results
from src.solver.algorithms import EnumerateAssignments,EnumerateAssignmentsUsingGenerator,ElimilateVariables,ElimilateVariablesRecursive,SplitEquations
from src.solver.DataTypes import Equation
def main():
    # example path
    file_path=bench_folder +"/examples/test.eq"
    #file_path = bench_folder +"/examples/01_track_2.eq"
    #file_path= bench_folder +"/examples/01_track_4.eq"
    #file_path = bench_folder +"/test/01_track_3.eq"
    # Woorpje_benchmarks path
    #SAT
    #file_path = bench_folder +"/01_track/01_track_1.eq"
    #file_path = bench_folder +"/01_track/01_track_2.eq"
    #file_path = bench_folder +"/01_track/01_track_3.eq"
    #file_path = bench_folder +"/01_track/01_track_4.eq"
    #file_path = bench_folder +"/01_track/01_track_5.eq"
    #file_path = bench_folder +"/01_track/01_track_36.eq"
    #file_path = bench_folder +"/01_track/01_track_37.eq"
    #file_path = bench_folder +"/01_track/01_track_58.eq"
    #file_path = bench_folder +"/01_track/01_track_93.eq"
    #file_path = bench_folder +"/01_track/01_track_192.eq"

    #UNSAT
    #file_path = bench_folder +"/03_track/03_track_14.eq"
    #file_path = bench_folder +"/03_track/03_track_7.eq"
    #file_path = bench_folder +"/03_track/03_track_11.eq"
    #file_path = bench_folder +"/03_track/03_track_17.eq"

    #multiple equations
    #file_path=bench_folder +"/examples/test2.eq"
    #file_path=bench_folder +"/04_track/04_track_10.eq"


    parser_type = EqParser()
    parser = Parser(parser_type)
    parsed_content = parser.parse(file_path)
    print("parsed_content:", parsed_content)

    graph_type="graph_5"
    graph_func_map = {None: Equation.get_graph_1, "graph_1": Equation.get_graph_1,
                      "graph_2": Equation.get_graph_2,"graph_3":Equation.get_graph_3,"graph_4":Equation.get_graph_4,
                      "graph_5":Equation.get_graph_5}

    algorithm_parameters = {"branch_method":"fixed","graph_type":graph_type,"graph_func":graph_func_map[graph_type]} # branch_method [gnn.random,fixed]

    #solver = Solver(algorithm=SplitEquations,algorithm_parameters=algorithm_parameters)
    solver = Solver(algorithm=ElimilateVariablesRecursive,algorithm_parameters=algorithm_parameters)
    #solver = Solver(algorithm=ElimilateVariables,algorithm_parameters=algorithm_parameters)
    #solver = Solver(EnumerateAssignmentsUsingGenerator, max_variable_length=max_variable_length,algorithm_parameters=algorithm_parameters)
    #solver = Solver(algorithm=EnumerateAssignments,max_variable_length=max_variable_length,algorithm_parameters=algorithm_parameters)
    result_dict = solver.solve(parsed_content,visualize=True,output_train_data=False)

    print_results(result_dict)


if __name__ == '__main__':
    main()
