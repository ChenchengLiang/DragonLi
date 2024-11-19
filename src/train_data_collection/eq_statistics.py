import json
import os
from typing import List
from src.solver.Constants import bench_folder
from src.solver.DataTypes import Equation

from src.solver.Parser import EqParser, Parser
from src.solver.independent_utils import strip_file_name_suffix

from tqdm import tqdm


def main():
    folder = f"{bench_folder}/01_track_multi_word_equations_generated_eval_1001_2000/ALL/ALL"

    # get parser
    parser = Parser(EqParser())

    # get eq file list
    eq_file_list = []
    all_files = os.scandir(folder)
    for file in tqdm(all_files, desc="Processing files"):
        if file.is_file() and file.name.endswith(".eq"):
            eq_file_list.append(file.name)

    # get statistics for each eq file
    statistic_file_name_list = []
    for eq_file in tqdm(eq_file_list, total=len(eq_file_list), desc="Processing eq files"):
        eq_file_path = os.path.join(folder, eq_file)

        # read one file
        parsed_content = parser.parse(eq_file_path)
        variable_list: List[str] = [v.value for v in parsed_content["variables"]]
        terminal_list: List[str] = [t.value for t in parsed_content["terminals"]]
        terminal_list.remove("\"\"")  # remove empty string
        eq_list: List[Equation] = parsed_content["equation_list"]

        eq_length_list = []
        variable_occurrence_list = []
        terminal_occurrence_list = []
        for eq in eq_list:
            # get equation length
            eq_length_list.append(eq.term_length)

            # get variable occurrence
            variable_occurrence_map = {v: 0 for v in variable_list}
            for v in variable_list:
                variable_occurrence_map[v] += (eq.eq_str).count(v)
            variable_occurrence_list.append(sum(list(variable_occurrence_map.values())))

            # get terminal occurrence
            terminal_occurrence_map = {t: 0 for t in terminal_list}
            for t in terminal_list:
                terminal_occurrence_map[t] += (eq.eq_str).count(t)
            terminal_occurrence_list.append(sum(list(terminal_occurrence_map.values())))

        # summary info
        line_offset = 3
        info_summary_with_id = {i + line_offset: "" for i in range(len(eq_list))}
        for i, (eq, eq_length, viariable_occurrences, terminal_occurrence) in enumerate(
                zip(eq_list, eq_length_list, variable_occurrence_list, terminal_occurrence_list)):
            info_summary_with_id[i + line_offset] = (
                f"length: {eq_length}, variables ({eq.variable_number}): {''.join([v.value for v in eq.variable_list])},"
                f" terminals ({eq.terminal_numbers_without_empty_terminal}): {''.join([t.value for t in eq.termimal_list_without_empty_terminal])},"
                f" variable_occurrence: {viariable_occurrences},"
                f" terminal_occurrence: {terminal_occurrence},"
                f" variable_occurrence_ratio: {viariable_occurrences / eq_length},"
                f" terminal_occurrence_ratio: {terminal_occurrence / eq_length}")

        # get statistics
        statistic_dict = {"number_of_equations": len(eq_list),
                          "min_eq_length": min(eq_length_list),
                          "max_eq_length": max(eq_length_list),
                          "average_eq_length": sum(eq_length_list) / len(eq_length_list),
                          "number_of_variables": len(variable_list),
                          "number_of_terminals": len(terminal_list),
                          "total_variable_occurrence_ratio": sum(variable_occurrence_list) / sum(eq_length_list),
                          "total_terminal_occurrence_ratio": sum(terminal_occurrence_list) / sum(eq_length_list),
                          "min_variable_occurrence": min(variable_occurrence_list),
                          "max_variable_occurrence": max(variable_occurrence_list),
                          "average_variable_occurrence": sum(variable_occurrence_list) / len(variable_occurrence_list),
                          "min_terminal_occurrence": min(terminal_occurrence_list),
                          "max_terminal_occurrence": max(terminal_occurrence_list),
                          "average_terminal_occurrence": sum(terminal_occurrence_list) / len(terminal_occurrence_list),
                          "info_summary_with_id": info_summary_with_id,
                          "eq_length_list": eq_length_list,
                          "variable_occurrence_list": variable_occurrence_list,
                          "terminal_occurrence_list": terminal_occurrence_list}
        # save statistics to file
        statistic_file_name = f"{strip_file_name_suffix(eq_file_path)}_statistics.json"
        statistic_file_name_list.append(statistic_file_name)
        with open(statistic_file_name, 'w') as file:
            json.dump(statistic_dict, file, indent=4)

    # get final_statistic_dict
    final_statistic_dict = {"total_variable_occurrence": 0,
                            "total_terminal_occurrence": 0,
                            "total_eq_symbol": 0,
                            "total_variable_occurrence_ratio":0,
                            "total_terminal_occurrence_ratio":0,
                            "min_eq_length": 0,
                            "max_eq_length": 0,
                            "average_eq_length": 0,
                            "min_eq_number_of_problems": 0,
                            "max_eq_number_of_problems": 0,
                            "average_eq_number_of_problems": 0,
                            "min_variable_occurrence_of_problem": 0,
                            "max_variable_occurrence_of_problem": 0,
                            "average_variable_occurrence_of_problem": 0,
                            "min_terminal_occurrence_of_problem": 0,
                            "max_terminal_occurrence_of_problem": 0,
                            "average_terminal_occurrence_of_problem": 0,


                            "equation_number:number_of_problems": {i: 0 for i in range(1, 101)},
                            "variable_number:number_of_problems": {i: 0 for i in range(0, 27)},
                            "terminal_number:number_of_problems": {i: 0 for i in range(0, 27)},
                            }

    eq_number_list_of_problems = []
    eq_length_list_of_problems = []
    variable_occurrence_list_of_problems = []
    terminal_occurrence_list_of_problems = []
    for statistic_file_name in statistic_file_name_list:
        with open(statistic_file_name, 'r') as file:
            statistic = json.load(file)
            final_statistic_dict["equation_number:number_of_problems"][statistic["number_of_equations"]] += 1
            final_statistic_dict["variable_number:number_of_problems"][statistic["number_of_variables"]] += 1
            final_statistic_dict["terminal_number:number_of_problems"][statistic["number_of_terminals"]] += 1
            final_statistic_dict["total_eq_symbol"] += sum(statistic["eq_length_list"])
            final_statistic_dict["total_variable_occurrence"] += sum(statistic["variable_occurrence_list"])
            final_statistic_dict["total_terminal_occurrence"] += sum(statistic["terminal_occurrence_list"])
            eq_length_list_of_problems.extend(statistic["eq_length_list"])
            eq_number_list_of_problems.append(statistic["number_of_equations"])
            variable_occurrence_list_of_problems.append(sum(statistic["variable_occurrence_list"]))
            terminal_occurrence_list_of_problems.append(sum(statistic["terminal_occurrence_list"]))



    final_statistic_dict["total_variable_occurrence_ratio"] = final_statistic_dict["total_variable_occurrence"] / \
                                                              final_statistic_dict["total_eq_symbol"]
    final_statistic_dict["total_terminal_occurrence_ratio"] = final_statistic_dict["total_terminal_occurrence"] / \
                                                              final_statistic_dict["total_eq_symbol"]
    final_statistic_dict["min_eq_number_of_problems"] = min(eq_number_list_of_problems)
    final_statistic_dict["max_eq_number_of_problems"] = max(eq_number_list_of_problems)
    final_statistic_dict["average_eq_number_of_problems"] = sum(eq_number_list_of_problems) / len(
        eq_number_list_of_problems)
    final_statistic_dict["min_eq_length"] = min(eq_length_list_of_problems)
    final_statistic_dict["max_eq_length"] = max(eq_length_list_of_problems)
    final_statistic_dict["average_eq_length"] = sum(eq_length_list_of_problems) / len(eq_length_list_of_problems)
    final_statistic_dict["min_variable_occurrence_of_problem"] = min(variable_occurrence_list_of_problems)
    final_statistic_dict["max_variable_occurrence_of_problem"] = max(variable_occurrence_list_of_problems)
    final_statistic_dict["average_variable_occurrence_of_problem"] = sum(variable_occurrence_list_of_problems) / len(
        variable_occurrence_list_of_problems)
    final_statistic_dict["min_terminal_occurrence_of_problem"] = min(terminal_occurrence_list_of_problems)
    final_statistic_dict["max_terminal_occurrence_of_problem"] = max(terminal_occurrence_list_of_problems)
    final_statistic_dict["average_terminal_occurrence_of_problem"] = sum(terminal_occurrence_list_of_problems) / len(
        terminal_occurrence_list_of_problems)


    # save final_statistic_dict to final_statistic.json
    final_statistic_file = f"{os.path.dirname(os.path.dirname(folder))}/final_statistic.json"
    with open(final_statistic_file, "w") as f:
        json.dump(final_statistic_dict, f, indent=4)


if __name__ == '__main__':
    main()
