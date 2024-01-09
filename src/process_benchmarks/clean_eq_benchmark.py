
from src.solver.independent_utils import delete_duplicate_files,delete_files_with_content,apply_to_all_files,delete_duplicate_lines
from src.solver.Constants import project_folder,bench_folder


def main():
    directory_path = bench_folder+'/kaluzaWoorpje/eq_test_delete_diplicated'
    duplicated_files_list = delete_duplicate_files(directory_path)

    target_content = "Variables {}\nTerminals {}\nSatGlucose(0)" # no variables
    empty_file_list=delete_files_with_content(directory_path, target_content)

    apply_to_all_files(directory_path, delete_duplicate_lines)



if __name__ == '__main__':
    main()