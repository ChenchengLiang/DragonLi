import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable, Any, Tuple
from typing import Dict, List, Set

from src.solver.algorithms import AbstractAlgorithm
from .Constants import max_variable_length, algorithm_timeout
from .DataTypes import Variable, Terminal, Term, Assignment


class Solver:
    def __init__(self, algorithm: AbstractAlgorithm, **kwargs):
        self._algorithm = algorithm
        self.kwargs=kwargs


    def solve(self, parsed_equation: Dict,visualize=False) -> (bool, Assignment):
        variables: List[Variable] = parsed_equation["variables"]
        terminals: List[Terminal] = parsed_equation["terminals"]
        left_terms: List[Term] = parsed_equation["left_terms"]
        right_terms: List[Term] = parsed_equation["right_terms"]

        print("-"*10, "Solving equation", "-"*10)
        self._algorithm = self._algorithm(terminals, variables, left_terms, right_terms, self.kwargs)
        result_dict, running_time = self.count_time(self._algorithm.run, algorithm_timeout)
        result_dict["running_time"] = running_time
        if visualize==True:
            self._algorithm.visualize(parsed_equation["file_path"])
        return result_dict


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
                return {"result":None}, end_time - start_time  # Return the elapsed time and None if the function times out

            end_time = time.time()
        return result, end_time - start_time  # Return the elapsed time and the result if the function completes
