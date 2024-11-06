import shutil
import sys
import configparser

from src.solver.algorithms.split_equation_utils import get_unsat_label

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)


from src.solver.Constants import bench_folder,UNKNOWN,UNSAT
from src.solver.independent_utils import create_folder, strip_file_name_suffix, dump_to_json_with_format, \
    copy_relative_files, zip_folder
import glob
from src.solver.utils_parser import perse_eq_file
import os

def main():
    benchmark_name="01_track_multi_word_equations_eq_2_50_generated_train_1_40000_one_core"
    for fo in os.listdir(bench_folder + "/" + benchmark_name):
        folder_basename=os.path.basename(fo)
        if "divided" in folder_basename:
            folder=folder_basename

            #folder="divided_2"

            all_eq_folder=f"{bench_folder}/{benchmark_name}/{folder}/UNSAT"
            train_folder=f"{bench_folder}/{benchmark_name}/{folder}/train"
            create_folder(train_folder)

            for eq_file in glob.glob(f"{all_eq_folder}/*.eq"):
                file_name=strip_file_name_suffix(eq_file)
                unsatcore_file=f"{file_name}.unsatcore"

                parsed_eq = perse_eq_file(eq_file)["equation_list"]
                parsed_unsatcore = perse_eq_file(unsatcore_file)["equation_list"]

                branch_eq_satisfiability_list = []
                for eq in parsed_eq:
                    if eq in parsed_unsatcore:
                        branch_eq_satisfiability_list.append((eq,UNSAT,eq.term_length))
                    else:
                        branch_eq_satisfiability_list.append((eq, UNKNOWN, eq.term_length))

                node_id=1
                label_list = [0] * len(branch_eq_satisfiability_list)
                satisfiability_list = []
                back_track_count_list = []
                middle_branch_eq_file_name_list = []
                one_train_data_name = f"{train_folder}/{os.path.basename(file_name)}@{node_id}"
                for index, (eq, satisfiability, branch_number) in enumerate(branch_eq_satisfiability_list):
                    satisfiability_list.append(satisfiability)
                    back_track_count_list.append(branch_number)
                    one_eq_file_name = f"{one_train_data_name}:{index}"

                    eq.output_eq_file_rank(one_eq_file_name, satisfiability)
                    middle_branch_eq_file_name_list.append(os.path.basename(one_eq_file_name) + ".eq")

                get_unsat_label(satisfiability_list,label_list,back_track_count_list)

                label_dict = {"satisfiability_list": satisfiability_list, "back_track_count_list": back_track_count_list,
                              "label_list": label_list, "middle_branch_eq_file_name_list": middle_branch_eq_file_name_list}
                dump_to_json_with_format(label_dict, one_train_data_name + ".label.json")
                copy_relative_files(prefix=file_name,target_folder=train_folder)

            #zip train folder
            zip_folder(folder_path=train_folder, output_zip_file=train_folder + ".zip")
            shutil.rmtree(train_folder)









if __name__ == '__main__':
    main()