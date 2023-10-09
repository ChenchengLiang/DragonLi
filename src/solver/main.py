import sys
sys.path.append("/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs")
from src.solver.Parser import Parser, EqParser, EqReader
from src.solver.Solver import Solver
from src.solver.utils import print_results
from src.solver.Algorithms import EnumerateAssignments, EnumerateAssignmentsUsingGenerator,ElimilateVariables
from src.solver.Constants import max_variable_length, algorithm_timeout
def main():
    # example path
    #file_path="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/examples/test.eq"
    #todo handle XX=a not terminated

    # Woorpje_benchmarks path
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje/01_track_1.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje/01_track_2.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje/01_track_3.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje/01_track_4.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje/01_track_37.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje/01_track_58.eq"
    file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje/01_track_185.eq"


    parser_type = EqParser()
    parser = Parser(parser_type)
    parsed_content = parser.parse(file_path)
    print("parsed_content:", parsed_content)


    solver = Solver(algorithm=ElimilateVariables)
    #solver = Solver(EnumerateAssignmentsUsingGenerator, max_variable_length=max_variable_length)
    #solver = Solver(algorithm=EnumerateAssignments,max_variable_length=max_variable_length)
    result_dict = solver.solve(parsed_content,visualize=False)

    print_results(result_dict)


if __name__ == '__main__':
    main()
