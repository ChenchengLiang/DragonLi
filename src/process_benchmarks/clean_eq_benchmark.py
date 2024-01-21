import os.path
import shutil

from src.solver.independent_utils import handle_duplicate_files,handle_files_with_target_string,apply_to_all_files,delete_duplicate_lines
from src.solver.Constants import project_folder,bench_folder


def main():

    benchmark_list=["all","BigSat","BigUnsat","SmallSat","SmallUnsat","kaluzaSmallSatExtracted","kaluzaWoorpje"]

    for benchmark in benchmark_list:
        eq_folder=f"{bench_folder}/kaluza/{benchmark}/eq"
        directory_path = f"{bench_folder}/kaluza/{benchmark}/eq_cleaned"
        if os.path.exists(eq_folder) and not os.path.exists(directory_path):
            clean_eq_files(eq_folder,directory_path)
        elif os.path.exists(eq_folder) and os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            clean_eq_files(eq_folder, directory_path)
        else:
            print("eq_folder not exists")


def clean_eq_files(eq_folder,directory_path):
    shutil.copytree(eq_folder, directory_path)

    target_content = "Variables {}\nTerminals {}\nSatGlucose(0)"  # no variables
    empty_file_list = handle_files_with_target_string(directory_path, target_content,move_to_folder_name="empty_eq", log=False)

    duplicated_files_list = handle_duplicate_files(directory_path, log=False)

    apply_to_all_files(directory_path, delete_duplicate_lines)


if __name__ == '__main__':
    main()