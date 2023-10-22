import sys
sys.path.append("/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs")
from src.solver.Parser import Parser, EqParser, EqReader
from src.solver.Solver import Solver
from src.solver.utils import print_results
from src.solver.algorithms import EnumerateAssignments,EnumerateAssignmentsUsingGenerator,ElimilateVariables,ElimilateVariablesRecursive,SplitEquations
from src.solver.Constants import max_variable_length, algorithm_timeout
from src.solver.DataTypes import Equation
def main():
    # example path
    #file_path="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/examples/test.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/examples/01_track_2.eq"
    #file_path= "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/examples/01_track_4.eq"
    # Woorpje_benchmarks path
    #SAT
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/01_track_1.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/01_track_2.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/01_track_3.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/01_track_4.eq"
    file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/01_track_5.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/01_track_36.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/01_track_37.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/01_track_58.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/01_track_93.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/01_track_192.eq"

    #UNSAT
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/03_track/03_track_14.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/03_track/03_track_7.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/03_track/03_track_11.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/03_track/03_track_17.eq"

    #multiple equations
    #file_path="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/examples/test2.eq"
    #file_path="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/04_track/04_track_10.eq"


    parser_type = EqParser()
    parser = Parser(parser_type)
    parsed_content = parser.parse(file_path)
    print("parsed_content:", parsed_content)

    algorithm_parameters = {"branch_method":"random","graph_type":"graph_1","graph_func":Equation.get_graph_1} # branch_method [gnn.random,fixed]

    #solver = Solver(algorithm=SplitEquations,algorithm_parameters=algorithm_parameters)
    solver = Solver(algorithm=ElimilateVariablesRecursive,algorithm_parameters=algorithm_parameters)
    #solver = Solver(algorithm=ElimilateVariables,algorithm_parameters=algorithm_parameters)
    #solver = Solver(EnumerateAssignmentsUsingGenerator, max_variable_length=max_variable_length,algorithm_parameters=algorithm_parameters)
    #solver = Solver(algorithm=EnumerateAssignments,max_variable_length=max_variable_length,algorithm_parameters=algorithm_parameters)
    result_dict = solver.solve(parsed_content,visualize=True,output_train_data=True)

    print_results(result_dict)


if __name__ == '__main__':
    main()
