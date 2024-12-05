import glob
from typing import List, Dict, Tuple

from src.process_benchmarks.eq2smt_utils import one_eq_file_to_smt2
from src.solver.DataTypes import Terminal, Variable, Equation, Formula
from src.solver.Parser import EqParser, Parser, SMT2Parser
from src.solver.Solver import Solver
from src.solver.algorithms import SplitEquations
from src.solver.Constants import bench_folder, project_folder, UNKNOWN, rank_task_label_size_map, benchmark_A_model, \
    benchmark_B_model, mlflow_folder
from src.solver.independent_utils import strip_file_name_suffix
from src.solver.utils import print_results, graph_func_map







def main():
    file_folder = f"{bench_folder}/eval_unsatcore_01_track_multi_word_equations_eq_2_50_generated_eval_1_1000/ALL"


    benchmark_model = benchmark_B_model


    rank_task = 1
    label_size = rank_task_label_size_map[rank_task]

    gnn_model_path = f"{mlflow_folder}/{benchmark_model['experiment_id']}/{benchmark_model['run_id']}/artifacts/model_0_{benchmark_model['graph_type']}_{benchmark_model['model_type']}.pth"

    algorithm_parameters_SplitEquations_gnn = {"branch_method": "fixed",
                                               "order_equations_method": "gnn_first_n_iterations_category",
                                               "gnn_model_path": gnn_model_path,
                                               "termination_condition": "termination_condition_0",
                                               "graph_type": benchmark_model['graph_type'], "graph_func": graph_func_map[benchmark_model['graph_type']],
                                               "label_size": label_size, "rank_task": rank_task}

    for file_path in glob.glob(f"{file_folder}/*.eq"):

        parser_type = EqParser() if file_path.endswith(".eq") else SMT2Parser()
        parser = Parser(parser_type)
        parsed_content = parser.parse(file_path)
        #print("parsed_content:", parsed_content)


        solver = Solver(algorithm=SplitEquationsOutputEqs, algorithm_parameters=algorithm_parameters_SplitEquations_gnn)

        result_dict = solver.solve(parsed_content, visualize=False, output_train_data=False)


class SplitEquationsOutputEqs(SplitEquations):
    def __init__(self, terminals: List[Terminal], variables: List[Variable], equation_list: List[Equation],
                 parameters: Dict):
        super().__init__(terminals, variables, equation_list, parameters)

    override = True
    def split_eq(self, input_formula: Formula, current_depth: int, previous_node: Tuple[int, Dict],
                 edge_label: str) -> Tuple[str, Formula]:

        current_node = self.record_node_and_edges(input_formula, previous_node, edge_label, current_depth)

        ranked_formula = self.order_equations_func_wrapper(input_formula, current_node)




        eq_string_to_file = ranked_formula.eq_string_for_file()
        # create eq file
        unsat_core_eq_file = f"{strip_file_name_suffix(self.parameters['file_path'])}.predicted_unsatcore"
        print("unsat_core_eq_file:", unsat_core_eq_file)
        with open(unsat_core_eq_file, "w") as f:
            f.write(eq_string_to_file)
        # store to smt2 file
        #unsat_core_smt2_file = one_eq_file_to_smt2(unsat_core_eq_file)

        return UNKNOWN, ranked_formula
if __name__ == '__main__':
    main()