
from src.solver.Constants import project_folder,bench_folder
from src.process_benchmarks.utils import clean_eq_files, get_clean_statistics, collect_cleaned_files
from src.solver.independent_utils import find_leaf_folders, remove_duplicates
import os
import glob
import shutil

def main():

    # Create the target folder for merging
    benchmark_name="merge_test"
    merged_dir = f"{bench_folder}/{benchmark_name}/merged_divided"
    os.makedirs(merged_dir, exist_ok=True)

    # Find all folders that match the pattern divided_*
    divided_folders = glob.glob(f"{bench_folder}/{benchmark_name}/divided_*")

    # Merge contents from each divided_* folder into merged_divided
    for folder in divided_folders:
        for item in os.listdir(folder):
            s = os.path.join(folder, item)
            d = os.path.join(merged_dir, item)

            # If it's a directory, merge recursively
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                # If it's a file, copy it over
                shutil.copy2(s, d)

    # Delete the original divided_* folders
    for folder in divided_folders:
        shutil.rmtree(folder)

    shutil.rmtree(f"{merged_dir}/eq_folder")
    os.remove(f"{merged_dir}/cleaned_eq_statistics.txt")
    print("Merging complete, and original folders deleted.")

    # clean eq
    leaf_folder_list = find_leaf_folders(bench_folder + "/" + benchmark_name)
    leaf_folder_list = [os.path.dirname(folder) for folder in leaf_folder_list if os.path.basename(folder) == "eq"]

    for folder in leaf_folder_list:
        print("clean eq", folder)
        clean_eq_files(folder)

    # statistics
    get_clean_statistics(benchmark_name, leaf_folder_list)

    # collect cleaned files
    collect_cleaned_files(benchmark_name, leaf_folder_list)


if __name__ == '__main__':
    main()