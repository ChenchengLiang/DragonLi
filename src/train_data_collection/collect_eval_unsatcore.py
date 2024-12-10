import os.path
from typing import List

from src.solver.independent_utils import custom_stdev
from src.solver.Constants import bench_folder
import glob
import statistics
from src.solver.Parser import EqParser, Parser
from src.solver.independent_utils import strip_file_name_suffix
import json


def main():
    benchmark = "eval_unsatcore_01_track_multi_word_equations_eq_2_50_generated_eval_1_1000"

    folder_list = glob.glob(f"{bench_folder}/{benchmark}/divided_*")
    parser = Parser(EqParser())

    original_problem_size_common_list = []

    predicted_unsatcore_size_common_list = []
    predicted_unsatcore_solver_common_list = []
    predicted_unsatcore_satisfiability_common_list = []
    predicted_unsatcore_solving_time_common_list= []

    unsatcore_size_common_list = []
    unsatcore_solver_common_list = []
    unsatcore_satisfiability_common_list = []
    unsatcore_solving_time_common_list = []

    predicted_unsatcore_size_all_list = []
    predicted_unsatcore_solver_all_list = []
    predicted_unsatcore_satisfiability_all_list = []
    predicted_unsatcore_solving_time_all_list = []
    unsatcore_size_all_list = []
    unsatcore_solver_all_list = []
    unsatcore_satisfiability_all_list = []
    unsatcore_solving_time_all_list = []

    for folder in folder_list:

        file_list = glob.glob(folder + "/*.eq")
        for eq_file in file_list:
            eq_list = parser.parse(eq_file)["equation_list"]
            eq_list_size = len(eq_list)
            striped_file_name = strip_file_name_suffix(eq_file)
            striped_base_file_name = os.path.basename(striped_file_name)

            # read predictd unsatcore files and unsatcore files
            predicted_unsatcore_folder = f"{striped_file_name}_predicted_unsatcore_eval"
            unsatcore_folder = f"{striped_file_name}_unsatcore"
            predicted_unsatcore_file = f"{predicted_unsatcore_folder}/{striped_base_file_name}.current_unsatcore"
            unsatcore_file = f"{unsatcore_folder}/{striped_base_file_name}.current_unsatcore"
            predicted_unsatcore_summary_file = f"{predicted_unsatcore_folder}/summary.json"
            unsatcore_summary_file = f"{unsatcore_folder}/summary.json"

            if os.path.exists(predicted_unsatcore_folder) == True and os.path.exists(
                    predicted_unsatcore_file) == True and os.path.exists(unsatcore_folder) == True and os.path.exists(
                    unsatcore_file) == True:
                predicted_eq_list = parser.parse(f"{folder}/{striped_base_file_name}.predicted_unsatcore")[
                    "equation_list"]
                predicted_unsatcore_eq_list = \
                    parser.parse(predicted_unsatcore_file)[
                        "equation_list"]
                # read summary json file
                with open(predicted_unsatcore_summary_file) as f:
                    predicted_unsatcore_summary_dict = json.load(f)

                print(predicted_unsatcore_summary_dict)

                unsatcore_eq_list = parser.parse(unsatcore_file)[
                    "equation_list"]
                # read summary json file
                with open(unsatcore_summary_file) as f:
                    unsatcore_summary_dict = json.load(f)

                print(unsatcore_summary_dict)

                original_problem_size_common_list.append(eq_list_size)
                predicted_unsatcore_size_common_list.append(
                    predicted_unsatcore_summary_dict["current_unsatcore_eq_number"])
                unsatcore_size_common_list.append(unsatcore_summary_dict["current_unsatcore_eq_number"])
                predicted_unsatcore_solver_common_list.append(predicted_unsatcore_summary_dict["first_solved_solver"])
                unsatcore_solver_common_list.append(unsatcore_summary_dict["first_solved_solver"])
                predicted_unsatcore_satisfiability_common_list.append(
                    predicted_unsatcore_summary_dict["satisfiability"])
                unsatcore_satisfiability_common_list.append(unsatcore_summary_dict["satisfiability"])
                predicted_unsatcore_solving_time_common_list.append(predicted_unsatcore_summary_dict["solving_time"])
                unsatcore_solving_time_common_list.append(unsatcore_summary_dict["solving_time"])

            if os.path.exists(predicted_unsatcore_folder) == True and os.path.exists(unsatcore_summary_file) == True:
                # read summary json file
                with open(predicted_unsatcore_summary_file) as f:
                    predicted_unsatcore_summary_dict = json.load(f)
                predicted_unsatcore_size_all_list.append(
                    predicted_unsatcore_summary_dict["current_unsatcore_eq_number"])
                predicted_unsatcore_solver_all_list.append(predicted_unsatcore_summary_dict["first_solved_solver"])
                predicted_unsatcore_satisfiability_all_list.append(predicted_unsatcore_summary_dict["satisfiability"])
                predicted_unsatcore_solving_time_all_list.append(predicted_unsatcore_summary_dict["solving_time"])

            if os.path.exists(unsatcore_folder) == True and os.path.exists(unsatcore_summary_file) == True:
                # read summary json file
                with open(unsatcore_summary_file) as f:
                    unsatcore_summary_dict = json.load(f)
                unsatcore_size_all_list.append(unsatcore_summary_dict["current_unsatcore_eq_number"])
                unsatcore_solver_all_list.append(unsatcore_summary_dict["first_solved_solver"])
                unsatcore_satisfiability_all_list.append(unsatcore_summary_dict["satisfiability"])
                unsatcore_solving_time_all_list.append(unsatcore_summary_dict["solving_time"])

            if os.path.exists(predicted_unsatcore_folder) == True and os.path.exists(predicted_unsatcore_file) == False:
                print(f"predicted_unsatcore_folder:{predicted_unsatcore_folder} does not exist")

            predicted_unsatcore_size_unsatcore_size_diff_list = [
                predicted_unsatcore_size_common_list[i] - unsatcore_size_common_list[i] for i
                in range(len(original_problem_size_common_list))]
            unsatcore_size_predicted_unsatcore_size_ratio = sum(unsatcore_size_common_list) / sum(
                predicted_unsatcore_size_common_list)
            unsatcore_size_predicted_unsatcore_size_ratio_list = [
                unsatcore_size_common_list[i] / predicted_unsatcore_size_common_list[i] for
                i in range(len(original_problem_size_common_list))]
            unsatcore_size_original_problem_size_ratio = sum(unsatcore_size_common_list) / sum(
                original_problem_size_common_list)
            unsatcore_size_original_problem_size_ratio_list = [
                unsatcore_size_common_list[i] / original_problem_size_common_list[i] for i in
                range(len(original_problem_size_common_list))]
            predicted_unsatcore_size_original_problem_size_ratio = sum(predicted_unsatcore_size_common_list) / sum(
                original_problem_size_common_list)
            predicted_unsatcore_size_original_problem_size_ratio_list = [
                predicted_unsatcore_size_common_list[i] / original_problem_size_common_list[i] for i in
                range(len(original_problem_size_common_list))]

            predicted_solver_count_common_dict = {"z3": predicted_unsatcore_solver_common_list.count("z3"),
                                                  "cvc5": predicted_unsatcore_solver_common_list.count("cvc5"),
                                                  "z3-noodler": predicted_unsatcore_solver_common_list.count(
                                                      "z3-noodler"),
                                                  "ostrich": predicted_unsatcore_solver_common_list.count("ostrich"),
                                                  "woorpje": predicted_unsatcore_solver_common_list.count("woorpje"),
                                                  "this": predicted_unsatcore_solver_common_list.count("this"),
                                                  "this:category_gnn_first_n_iterations": predicted_unsatcore_solver_common_list.count(
                                                      "this:category_gnn_first_n_iterations"),
                                                  "this:gnn_first_n_iterations_category": predicted_unsatcore_solver_common_list.count(
                                                      "this:gnn_first_n_iterations_category"),
                                                  "this:category_gnn": predicted_unsatcore_solver_common_list.count(
                                                      "this:category_gnn"),
                                                  "this:category_gnn_formula_size": predicted_unsatcore_solver_common_list.count(
                                                      "this:category_gnn_formula_size"),
                                                  "this:category_gnn_each_n_iterations": predicted_unsatcore_solver_common_list.count(
                                                      "this:category_gnn_each_n_iterations"),
                                                  }

            predicted_solver_count_common_dict["total_verify"] = sum(predicted_solver_count_common_dict.values())
            unsatcore_solver_count_common_dict = {"z3": unsatcore_solver_common_list.count("z3"),
                                                  "cvc5": unsatcore_solver_common_list.count("cvc5"),
                                                  "z3-noodler": unsatcore_solver_common_list.count("z3-noodler"),
                                                  "ostrich": unsatcore_solver_common_list.count("ostrich"),
                                                  "woorpje": unsatcore_solver_common_list.count("woorpje"),
                                                  "this": unsatcore_solver_common_list.count("this"), }
            unsatcore_solver_count_common_dict["total_verify"] = sum(unsatcore_solver_count_common_dict.values())

            predicted_unsatcore_satisfiability_common_dict = {
                "UNSAT": predicted_unsatcore_satisfiability_common_list.count("UNSAT"),
                "SAT": predicted_unsatcore_satisfiability_common_list.count("SAT"),
                "UNKNOWN": predicted_unsatcore_satisfiability_common_list.count("UNKNOWN")}
            predicted_unsatcore_satisfiability_common_dict["total_satisfiability_verify"] = sum(
                predicted_unsatcore_satisfiability_common_dict.values())
            unsatcore_satisfiability_common_dict = {"UNSAT": unsatcore_satisfiability_common_list.count("UNSAT"),
                                                    "SAT": unsatcore_satisfiability_common_list.count("SAT"),
                                                    "UNKNOWN": unsatcore_satisfiability_common_list.count("UNKNOWN")}
            unsatcore_satisfiability_common_dict["total_satisfiability_verify"] = sum(
                unsatcore_satisfiability_common_dict.values())

            predicted_solver_count_all_dict = {"z3": predicted_unsatcore_solver_all_list.count("z3"),
                                               "cvc5": predicted_unsatcore_solver_all_list.count("cvc5"),
                                               "z3-noodler": predicted_unsatcore_solver_all_list.count("z3-noodler"),
                                               "ostrich": predicted_unsatcore_solver_all_list.count("ostrich"),
                                               "woorpje": predicted_unsatcore_solver_all_list.count("woorpje"),
                                               "this": predicted_unsatcore_solver_all_list.count("this"),
                                               "this:category_gnn_formula_size": predicted_unsatcore_solver_all_list.count(
                                                   "this:category_gnn_formula_size"),
                                               "this:gnn_first_n_iterations_category": predicted_unsatcore_solver_all_list.count(
                                                   "this:gnn_first_n_iterations_category"), }
            predicted_solver_count_all_dict["total_verify"] = sum(predicted_solver_count_all_dict.values())
            unsatcore_solver_count_all_dict = {"z3": unsatcore_solver_all_list.count("z3"),
                                               "cvc5": unsatcore_solver_all_list.count("cvc5"),
                                               "z3-noodler": unsatcore_solver_all_list.count("z3-noodler"),
                                               "ostrich": unsatcore_solver_all_list.count("ostrich"),
                                               "woorpje": unsatcore_solver_all_list.count("woorpje"),
                                               "this": unsatcore_solver_all_list.count("this"), }
            unsatcore_solver_count_all_dict["total_verify"] = sum(unsatcore_solver_count_all_dict.values())

            predicted_unsatcore_satisfiability_all_dict = {
                "UNSAT": predicted_unsatcore_satisfiability_all_list.count("UNSAT"),
                "SAT": predicted_unsatcore_satisfiability_all_list.count("SAT"),
                "UNKNOWN": predicted_unsatcore_satisfiability_all_list.count("UNKNOWN")}
            predicted_unsatcore_satisfiability_all_dict["total_satisfiability_verify"] = sum(
                predicted_unsatcore_satisfiability_all_dict.values())
            unsatcore_satisfiability_all_dict = {"UNSAT": unsatcore_satisfiability_all_list.count("UNSAT"),
                                                 "SAT": unsatcore_satisfiability_all_list.count("SAT"),
                                                 "UNKNOWN": unsatcore_satisfiability_all_list.count("UNKNOWN")}
            unsatcore_satisfiability_all_dict["total_satisfiability_verify"] = sum(
                unsatcore_satisfiability_all_dict.values())

            summary_dict = {
                "total_summarized_files_common": len(original_problem_size_common_list),
                "predicted_solver_count_common_dict": predicted_solver_count_common_dict,
                "unsatcore_solver_count_common_dict": unsatcore_solver_count_common_dict,
                "predicted_unsatcore_satisfiability_common_dict": predicted_unsatcore_satisfiability_common_dict,
                "unsatcore_satisfiability_common_dict": unsatcore_satisfiability_common_dict,

                "predicted_unsatcore_solving_time_common_min": min(predicted_unsatcore_solving_time_common_list),
                "predicted_unsatcore_solving_time_common_max": max(predicted_unsatcore_solving_time_common_list),
                "predicted_unsatcore_solving_time_common_mean": statistics.mean(predicted_unsatcore_solving_time_common_list),
                "predicted_unsatcore_solving_time_common_stdev": custom_stdev(predicted_unsatcore_solving_time_common_list),
                "unsatcore_solving_time_common_min": min(unsatcore_solving_time_common_list),
                "unsatcore_solving_time_common_max": max(unsatcore_solving_time_common_list),
                "unsatcore_solving_time_common_mean": statistics.mean(unsatcore_solving_time_common_list),
                "unsatcore_solving_time_common_stdev": custom_stdev(unsatcore_solving_time_common_list),

                "original_problem_size_min": min(original_problem_size_common_list),
                "original_problem_size_max": max(original_problem_size_common_list),
                "original_problem_size_mean": statistics.mean(original_problem_size_common_list),
                "original_problem_size_stdev": custom_stdev(original_problem_size_common_list),
                "predicted_unsatcore_size_min": min(predicted_unsatcore_size_common_list),
                "predicted_unsatcore_size_max": max(predicted_unsatcore_size_common_list),
                "predicted_unsatcore_size_mean": statistics.mean(predicted_unsatcore_size_common_list),
                "predicted_unsatcore_size_stdev": custom_stdev(predicted_unsatcore_size_common_list),
                "unsatcore_size_min": min(unsatcore_size_common_list),
                "unsatcore_size_max": max(unsatcore_size_common_list),
                "unsatcore_size_mean": statistics.mean(unsatcore_size_common_list),
                "unsatcore_size_stdev": custom_stdev(unsatcore_size_common_list),
                "predicted_unsatcore_unsatcore_size_size_diff_min": min(
                    predicted_unsatcore_size_unsatcore_size_diff_list),
                "predicted_unsatcore_unsatcore_size_size_diff_max": max(
                    predicted_unsatcore_size_unsatcore_size_diff_list),
                "predicted_unsatcore_unsatcore_size_size_diff_mean": statistics.mean(
                    predicted_unsatcore_size_unsatcore_size_diff_list),
                "predicted_unsatcore_unsatcore_size_size_diff_stdev": custom_stdev(
                    predicted_unsatcore_size_unsatcore_size_diff_list),
                "unsatcore_size/predicted_unsatcore_size_total_ratio": unsatcore_size_predicted_unsatcore_size_ratio,
                "unsatcore_size/predicted_unsatcore_size_ratio_min": min(
                    unsatcore_size_predicted_unsatcore_size_ratio_list),
                "unsatcore_size/predicted_unsatcore_size_ratio_max": max(
                    unsatcore_size_predicted_unsatcore_size_ratio_list),
                "unsatcore_size/original_problem_size_total_ratio": unsatcore_size_original_problem_size_ratio,
                "unsatcore_size/original_problem_size_ratio_min": min(unsatcore_size_original_problem_size_ratio_list),
                "unsatcore_size/original_problem_size_ratio_max": max(unsatcore_size_original_problem_size_ratio_list),
                "predicted_unsatcore_size/original_problem_size_total_ratio": predicted_unsatcore_size_original_problem_size_ratio,
                "predicted_unsatcore_size/original_problem_size_ratio_min": min(
                    predicted_unsatcore_size_original_problem_size_ratio_list),
                "predicted_unsatcore_size/original_problem_size_ratio_max": max(
                    predicted_unsatcore_size_original_problem_size_ratio_list),

                "total_summarized_files_predicted_unsatcore": len(predicted_unsatcore_size_all_list),
                "predicted_solver_count_all_dict": predicted_solver_count_all_dict,
                "predicted_unsatcore_satisfiability_all_dict": predicted_unsatcore_satisfiability_all_dict,
                "predicted_unsatcore_solving_time_all_min": min(predicted_unsatcore_solving_time_all_list),
                "predicted_unsatcore_solving_time_all_max": max(predicted_unsatcore_solving_time_all_list),
                "predicted_unsatcore_solving_time_all_mean": statistics.mean(predicted_unsatcore_solving_time_all_list),
                "predicted_unsatcore_solving_time_all_stdev": custom_stdev(predicted_unsatcore_solving_time_all_list),

                "total_summarized_files_unsatcore": len(unsatcore_size_all_list),
                "unsatcore_solver_count_all_dict": unsatcore_solver_count_all_dict,
                "unsatcore_satisfiability_all_dict": unsatcore_satisfiability_all_dict,
                "unsatcore_solving_time_all_min": min(unsatcore_solving_time_all_list),
                "unsatcore_solving_time_all_max": max(unsatcore_solving_time_all_list),
                "unsatcore_solving_time_all_mean": statistics.mean(unsatcore_solving_time_all_list),
                "unsatcore_solving_time_all_stdev": custom_stdev(unsatcore_solving_time_all_list),
            }

            # write dict to json file
            with open(f"{bench_folder}/{benchmark}/unsatcore_summary.json", "w") as f:
                json.dump(summary_dict, f, indent=4)


if __name__ == '__main__':
    main()
