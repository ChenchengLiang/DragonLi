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

    original_problem_size_list = []
    predicted_unsatcore_size_list = []
    unsatcore_size_list = []
    predicted_unsatcore_solver_list = []
    unsatcore_solver_list = []

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

            if os.path.exists(predicted_unsatcore_folder) == True and os.path.exists(unsatcore_folder) == True:
                predicted_eq_list = parser.parse(f"{folder}/{striped_base_file_name}.predicted_unsatcore")[
                    "equation_list"]
                predicted_unsatcore_eq_list = \
                    parser.parse(f"{predicted_unsatcore_folder}/{striped_base_file_name}.current_unsatcore")[
                        "equation_list"]
                # read summary json file
                with open(f"{predicted_unsatcore_folder}/summary.json") as f:
                    predicted_unsatcore_summary_dict = json.load(f)

                print(predicted_unsatcore_summary_dict)

                unsatcore_eq_list = parser.parse(f"{unsatcore_folder}/{striped_base_file_name}.current_unsatcore")[
                    "equation_list"]
                # read summary json file
                with open(f"{unsatcore_folder}/summary.json") as f:
                    unsatcore_summary_dict = json.load(f)

                print(unsatcore_summary_dict)

                original_problem_size_list.append(eq_list_size)
                predicted_unsatcore_size_list.append(predicted_unsatcore_summary_dict["current_unsatcore_eq_number"])
                unsatcore_size_list.append(unsatcore_summary_dict["current_unsatcore_eq_number"])
                predicted_unsatcore_solver_list.append(predicted_unsatcore_summary_dict["first_solved_solver"])
                unsatcore_solver_list.append(unsatcore_summary_dict["first_solved_solver"])

    predicted_unsatcore_size_unsatcore_size_diff_list = [predicted_unsatcore_size_list[i] - unsatcore_size_list[i] for i
                                                         in range(len(original_problem_size_list))]
    unsatcore_size_predicted_unsatcore_size_ratio = sum(unsatcore_size_list) / sum(predicted_unsatcore_size_list)
    unsatcore_size_predicted_unsatcore_size_ratio_list = [unsatcore_size_list[i] / predicted_unsatcore_size_list[i] for
                                                          i in range(len(original_problem_size_list))]
    unsatcore_size_original_problem_size_ratio = sum(unsatcore_size_list) / sum(original_problem_size_list)
    unsatcore_size_original_problem_size_ratio_list = [unsatcore_size_list[i] / original_problem_size_list[i] for i in
                                                       range(len(original_problem_size_list))]
    predicted_unsatcore_size_original_problem_size_ratio = sum(predicted_unsatcore_size_list) / sum(
        original_problem_size_list)
    predicted_unsatcore_size_original_problem_size_ratio_list = [
        predicted_unsatcore_size_list[i] / original_problem_size_list[i] for i in
        range(len(original_problem_size_list))]

    predicted_solver_count_dict = {"z3": predicted_unsatcore_solver_list.count("z3"),
                                   "cvc5": predicted_unsatcore_solver_list.count("cvc5"),
                                   "z3-noodler": predicted_unsatcore_solver_list.count("z3-noodler"),
                                   "ostrich": predicted_unsatcore_solver_list.count("ostrich"),
                                   "woorpje": predicted_unsatcore_solver_list.count("woorpje"),
                                   "this": predicted_unsatcore_solver_list.count("this"), }
    predicted_solver_count_dict["total_verify"] = sum(predicted_solver_count_dict.values())
    unsatcore_solver_count_dict = {"z3": unsatcore_solver_list.count("z3"),
                                   "cvc5": unsatcore_solver_list.count("cvc5"),
                                   "z3-noodler": unsatcore_solver_list.count("z3-noodler"),
                                   "ostrich": unsatcore_solver_list.count("ostrich"),
                                   "woorpje": unsatcore_solver_list.count("woorpje"),
                                   "this": unsatcore_solver_list.count("this"), }
    unsatcore_solver_count_dict["total_verify"] = sum(unsatcore_solver_count_dict.values())

    summary_dict = {
        "total_summarized_files": len(original_problem_size_list),
        "predicted_solver_count_dict": predicted_solver_count_dict,
        "unsatcore_solver_count_dict": unsatcore_solver_count_dict,
        "original_problem_size_min": min(original_problem_size_list),
        "original_problem_size_max": max(original_problem_size_list),
        "original_problem_size_mean": statistics.mean(original_problem_size_list),
        "original_problem_size_stdev": custom_stdev(original_problem_size_list),
        "predicted_unsatcore_size_min": min(predicted_unsatcore_size_list),
        "predicted_unsatcore_size_max": max(predicted_unsatcore_size_list),
        "predicted_unsatcore_size_mean": statistics.mean(predicted_unsatcore_size_list),
        "predicted_unsatcore_size_stdev": custom_stdev(predicted_unsatcore_size_list),
        "unsatcore_size_min": min(unsatcore_size_list),
        "unsatcore_size_max": max(unsatcore_size_list),
        "unsatcore_size_mean": statistics.mean(unsatcore_size_list),
        "unsatcore_size_stdev": custom_stdev(unsatcore_size_list),
        "predicted_unsatcore_unsatcore_size_size_diff_min": min(predicted_unsatcore_size_unsatcore_size_diff_list),
        "predicted_unsatcore_unsatcore_size_size_diff_max": max(predicted_unsatcore_size_unsatcore_size_diff_list),
        "predicted_unsatcore_unsatcore_size_size_diff_mean": statistics.mean(
            predicted_unsatcore_size_unsatcore_size_diff_list),
        "predicted_unsatcore_unsatcore_size_size_diff_stdev": custom_stdev(
            predicted_unsatcore_size_unsatcore_size_diff_list),
        "unsatcore_size/predicted_unsatcore_size_total_ratio": unsatcore_size_predicted_unsatcore_size_ratio,
        "unsatcore_size/predicted_unsatcore_size_ratio_min": min(unsatcore_size_predicted_unsatcore_size_ratio_list),
        "unsatcore_size/predicted_unsatcore_size_ratio_max": max(unsatcore_size_predicted_unsatcore_size_ratio_list),
        "unsatcore_size/original_problem_size_total_ratio": unsatcore_size_original_problem_size_ratio,
        "unsatcore_size/original_problem_size_ratio_min": min(unsatcore_size_original_problem_size_ratio_list),
        "unsatcore_size/original_problem_size_ratio_max": max(unsatcore_size_original_problem_size_ratio_list),
        "predicted_unsatcore_size/original_problem_size_total_ratio": predicted_unsatcore_size_original_problem_size_ratio,
        "predicted_unsatcore_size/original_problem_size_ratio_min": min(
            predicted_unsatcore_size_original_problem_size_ratio_list),
        "predicted_unsatcore_size/original_problem_size_ratio_max": max(
            predicted_unsatcore_size_original_problem_size_ratio_list),
    }

    # write dict to json file
    with open(f"{bench_folder}/{benchmark}/unsatcore_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=4)


if __name__ == '__main__':
    main()
