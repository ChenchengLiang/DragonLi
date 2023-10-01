from Parser import Parser, EqParser, EqReader
from Solver import Solver
from utils import print_results
from Algorithms import EnumerateAssignments, EnumerateAssignmentsUsingGenerator


def main():
    parser_type = EqParser()
    # example path
    #file_path="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/examples/1.eq"
    # Woorpje_benchmarks path
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje/01_track_1.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje/01_track_2.eq"
    #file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje/01_track_3.eq"
    file_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track/woorpje/01_track_4.eq"

    reader = EqReader()
    content = reader.read(file_path)
    print("content:", content)

    parser = Parser(parser_type)
    parsed_content = parser.parse(file_path)
    print("parsed_content:", parsed_content)

    solver = Solver(algorithm=EnumerateAssignmentsUsingGenerator)
    satisfiability, assignment = solver.solve(parsed_content)

    print_results(satisfiability, assignment, parsed_content)


if __name__ == '__main__':
    main()
