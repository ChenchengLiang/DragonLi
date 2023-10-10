import sys
sys.path.append("/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs")
from src.solver.Parser import Parser, EqParser, EqReader
from src.solver.Solver import Solver
from src.solver.utils import print_results
from src.solver.algorithms import EnumerateAssignments, EnumerateAssignmentsUsingGenerator,ElimilateVariables,ElimilateVariablesRecursive
from src.solver.Constants import max_variable_length, algorithm_timeout

def main(args):
    #print("args:", args)

    # example path
    file_path=args[0]


    parser_type = EqParser()
    parser = Parser(parser_type)
    parsed_content = parser.parse(file_path)
    print("parsed_content:", parsed_content)

    solver = Solver(algorithm=ElimilateVariables)
    # solver = Solver(EnumerateAssignmentsUsingGenerator, max_variable_length=max_variable_length)
    result_dict = solver.solve(parsed_content)

    print_results(result_dict)




if __name__ == '__main__':
    main(sys.argv[1:])
