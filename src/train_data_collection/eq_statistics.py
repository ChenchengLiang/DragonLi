import json
import os
from typing import List
from src.solver.Constants import bench_folder
from src.solver.DataTypes import Equation

from src.solver.Parser import EqParser,Parser
from src.solver.independent_utils import strip_file_name_suffix


def main():
    folder=f"{bench_folder}/04_track_DragonLi_train_1_100/ALL/ALL"

    parser = Parser(EqParser())

    statistic_file_name_list = []
    all_files=os.scandir(folder)
    for file in all_files:
        if file.is_file() and file.name.endswith(".eq"):
            eq_file=file.name
            eq_file_path=os.path.join(folder,eq_file)

            # read one file
            parsed_content = parser.parse(eq_file_path)
            variable_list:List[str] = [v.value for v in parsed_content["variables"]]
            terminal_list:List[str] = [t.value for t in parsed_content["terminals"]]
            terminal_list.remove("\"\"")#remove empty string
            eq_list:List[Equation] = parsed_content["equation_list"]

            eq_length_list=[]
            variable_occurrence_list=[]
            terminal_occurrence_list=[]
            for eq in eq_list:
                #get equation length
                eq_length_list.append(eq.term_length)

                #get variable occurrence
                variable_occurrence_map = {v: 0 for v in variable_list}
                for v in variable_list:
                    variable_occurrence_map[v] += (eq.eq_str).count(v)
                variable_occurrence_list.append(sum(list(variable_occurrence_map.values())))

                #get terminal occurrence
                terminal_occurrence_map = {t: 0 for t in terminal_list}
                for t in terminal_list:
                    terminal_occurrence_map[t] += (eq.eq_str).count(t)
                terminal_occurrence_list.append(sum(list(terminal_occurrence_map.values())))

            # get statistics
            statistic_dict = {"number_of_equations": len(eq_list),
                              "number_of_variables": len(variable_list),
                              "number_of_terminals": len(terminal_list),
                              "eq_length_list": eq_length_list,
                              "variable_occurrence_list": variable_occurrence_list,
                              "terminal_occurrence_list": terminal_occurrence_list}
            statistic_file_name = f"{strip_file_name_suffix(eq_file_path)}_statistics.json"
            statistic_file_name_list.append(statistic_file_name)
            with open(statistic_file_name, 'w') as file:
                json.dump(statistic_dict, file, indent=4)


    # get final_statistic_dict
    final_statistic_dict = {"equation_number:number_of_problems": {i: 0 for i in range(1, 101)},
                             "variable_number:number_of_problems": {i: 0 for i in range(1, 27)},
                             "terminal_number:number_of_problems": {i: 0 for i in range(1, 27)}}

    for statistic_file_name in statistic_file_name_list:
        with open(statistic_file_name, 'r') as file:
            statistic = json.load(file)
            final_statistic_dict["equation_number:number_of_problems"][statistic["number_of_equations"]] += 1
            final_statistic_dict["variable_number:number_of_problems"][statistic["number_of_variables"]] += 1
            final_statistic_dict["terminal_number:number_of_problems"][statistic["number_of_terminals"]] += 1
            print(statistic)


    # save final_statistic_dict to final_statistic.json
    final_statistic_file = f"{os.path.dirname(os.path.dirname(folder))}/final_statistic.json"
    with open(final_statistic_file, "w") as f:
        json.dump(final_statistic_dict, f, indent=4)








if __name__ == '__main__':
    main()