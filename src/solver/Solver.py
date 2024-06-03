import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable, Any, Tuple
from typing import Dict, List, Set

from src.solver.algorithms import AbstractAlgorithm
from .Constants import max_variable_length, algorithm_timeout
from .DataTypes import Variable, Terminal, Term, Assignment, Equation


class Solver:
    def __init__(self, algorithm: AbstractAlgorithm, algorithm_parameters: Dict):
        self._algorithm = algorithm
        self._algorithm_parameters = algorithm_parameters

    def solve(self, parsed_equations: Dict, visualize=False, output_train_data=False) -> (bool, Assignment):
        variables, terminals, equation_list = self.preprocess(parsed_equations)

        self._algorithm_parameters["file_path"] = parsed_equations["file_path"]
        self._algorithm_parameters["visualize"] = visualize
        self._algorithm_parameters["output_train_data"] = output_train_data

        print("-" * 10, "Solving equation", "-" * 10)
        self._algorithm = self._algorithm(terminals, variables, equation_list, self._algorithm_parameters)
        result_dict, running_time = self.count_time(self._algorithm.run, algorithm_timeout)
        result_dict["running_time"] = running_time
        if visualize == True:
            self._algorithm.visualize(parsed_equations["file_path"], self._algorithm_parameters["graph_func"])

        return result_dict

    def preprocess(self, parsed_equations: Dict, log=False):
        variables: List[Variable] = parsed_equations["variables"]
        terminals: List[Terminal] = parsed_equations["terminals"]
        equation_list: List[Equation] = parsed_equations["equation_list"]

        deduplicated_equations: List[Equation] = self.deduplicate_equations(equation_list)
        sorted_equations = sorted(deduplicated_equations, key=lambda x: x.eq_str)
        final_equations = sorted_equations

        if log == True:
            print(f"- before deduplication {len(equation_list)}-")
            for e in equation_list:
                print(e.eq_str)
            print(f"- after deduplication {len(final_equations)}-")
            for e in final_equations:
                print(e.eq_str)

        return variables, terminals, final_equations

    def deduplicate_equations(self, equation_list: List[Equation]) -> List[Equation]:
        unique_equations = []
        seen_equations = []

        for eq in equation_list:
            eq_tuple = (eq.left_terms, eq.right_terms)
            if eq_tuple not in seen_equations:
                unique_equations.append(eq)
                seen_equations.append(eq_tuple)
                seen_equations.append((eq_tuple[1],eq_tuple[0]))


        return unique_equations

    def count_time(self, func: Callable[..., Any], timeout=algorithm_timeout, *args, **kwargs) -> Tuple[float, Any]:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)  # Submit the function to the executor

            start_time = time.time()
            try:
                # Wait for the function to complete, or for the timeout to expire
                result = future.result(timeout=timeout)
            except TimeoutError:
                future.cancel()  # Cancel the function if it times out
                end_time = time.time()
                return {
                    "result": None}, end_time - start_time  # Return the elapsed time and None if the function times out

            end_time = time.time()
        return result, end_time - start_time  # Return the elapsed time and the result if the function completes
