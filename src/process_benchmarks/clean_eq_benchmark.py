import os.path
import shutil

from src.solver.independent_utils import handle_duplicate_files,handle_files_with_target_string,apply_to_all_files,delete_duplicate_lines
from src.solver.Constants import project_folder,bench_folder
from src.process_benchmarks.utils import clean_eq_files

def main():
    folder=bench_folder+"/kaluza/all/eq"
    clean_eq_files(folder)



if __name__ == '__main__':
    main()