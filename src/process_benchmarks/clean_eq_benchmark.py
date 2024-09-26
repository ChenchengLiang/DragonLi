
from src.solver.Constants import project_folder,bench_folder
from src.process_benchmarks.utils import clean_eq_files, get_clean_statistics, collect_cleaned_files
from src.solver.independent_utils import find_leaf_folders, remove_duplicates
import os
import glob
import shutil

def main():
    #clean and collect nest folders
    benchmark = "zaligvinder+smtlib"
    leaf_folder_list = find_leaf_folders(bench_folder + "/" + benchmark)
    #print(leaf_folder_list)
    leaf_folder_list=[os.path.dirname(folder) for folder in leaf_folder_list if os.path.basename(folder)=="eq"]
    #leaf_folder_list = remove_duplicates([os.path.dirname(folder) for folder in leaf_folder_list if "exceptions" not in folder])
    print(leaf_folder_list)

    #clean eq
    for folder in leaf_folder_list:
        print("clean eq", folder)
        clean_eq_files(folder)

    # statistics
    get_clean_statistics(benchmark, leaf_folder_list)

    # collect cleaned files
    collect_cleaned_files(benchmark,leaf_folder_list)


    #clean one folder
    # benchmark="smtlib-test/small"
    # folder=bench_folder+"/"+benchmark
    # clean_eq_files(folder)
    #
    # get_clean_statistics(benchmark, [folder])



if __name__ == '__main__':
    main()