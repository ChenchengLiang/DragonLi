import sys
import configparser

from sphinx.cmd.quickstart import suffix

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

import os
from src.solver.Constants import project_folder, bench_folder
from src.solver.independent_utils import strip_file_name_suffix, create_folder, copy_relative_files_and_directories, \
    copy_relative_files
import glob
import shutil

def main():
    benchmark_name="01_track_multi_word_equations_generated_eval_1001_2000_new_trainable_data"
    divided_folder_list=[fo for fo in os.listdir(f"{bench_folder}/{benchmark_name}") if "divided" in os.path.basename(fo)]

    create_folder(f"{bench_folder}/{benchmark_name}/merged_for_proof_tree")
    merged_folder_for_proof_tree=create_folder(f"{bench_folder}/{benchmark_name}/merged_for_proof_tree/UNSAT")
    create_folder(f"{bench_folder}/{benchmark_name}/merged_unsatcores")
    merged_unsatcores_folder=create_folder(f"{bench_folder}/{benchmark_name}/merged_unsatcores/UNSAT")


    for divided_folder in divided_folder_list:
        for eq_file in glob.glob(f"{bench_folder}/{benchmark_name}/{divided_folder}/UNSAT/*.eq"):
            file_name=strip_file_name_suffix(eq_file)
            #merged_folder_for_proof_tree
            if os.path.exists(f"{file_name}.unsatcore"):
                #copy_relative_files_and_directories(prefix=file_name,target_folder=merged_folder)
                copy_relative_files(prefix=file_name,target_folder=merged_folder_for_proof_tree)
            else:
                print(os.path.basename(eq_file),"don't have .unsatcore")
            #merged_unsatcores_folder
            if os.path.exists(f"{file_name}_unsat_cores"):
                copy_relative_files(prefix=file_name, target_folder=merged_unsatcores_folder)
                if not os.path.exists(f"{file_name}.unsatcore"):
                    # move smallest unsatecore to file .unsatcore
                    smallest_unsatcore_file=sorted(glob.glob(f"{file_name}_unsat_cores/*.eq"),reverse=True)[-1]
                    shutil.copy(smallest_unsatcore_file,f"{merged_unsatcores_folder}/{os.path.basename(file_name)}.unsatcore")
                else:
                    pass
            else:
                print(os.path.basename(eq_file),"don't have unsatcore")







if __name__ == '__main__':
    main()
