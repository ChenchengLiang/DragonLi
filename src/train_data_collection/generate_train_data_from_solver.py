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
from src.solver.Constants import project_folder,bench_folder
def main():

    for file_path in glob.glob(bench_folder+"/01_track_generated_train_data/SAT_from_solver/graph_1/*.eq"):

        parser_type = EqParser()
        parser = Parser(parser_type)
        parsed_content = parser.parse(file_path)
        print("parsed_content:", parsed_content)

        algorithm_parameters = {"branch_method":"fixed","graph_type":"graph_1","graph_func":Equation.get_graph_1} # branch_method [gnn.random,fixed]

        #solver = Solver(algorithm=SplitEquations,algorithm_parameters=algorithm_parameters)
        solver = Solver(algorithm=ElimilateVariablesRecursive,algorithm_parameters=algorithm_parameters)

        result_dict = solver.solve(parsed_content,visualize=False,output_train_data=True)

        print_results(result_dict)


if __name__ == '__main__':
    main()
