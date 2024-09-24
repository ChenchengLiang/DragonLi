from src.process_benchmarks.utils import run_on_one_problem, get_clean_statistics, collect_cleaned_files, update_ostrich
from src.solver.Constants import project_folder,bench_folder
from src.solver.independent_utils import strip_file_name_suffix,time_it,find_leaf_folders
import glob
import os
import shutil
from src.process_benchmarks.utils import smt_to_eq_one_folder,clean_eq_files
def main():
    benchmark="2023-05-05_clean-1/except-1"
    leaf_folder_list=find_leaf_folders(bench_folder+"/"+benchmark)

    #move smt2 files to smt2 folder
    for folder in leaf_folder_list:
        print(folder)
        if os.path.exists(folder+"/smt2"):
            shutil.rmtree(folder+"/smt2")
        os.mkdir(folder+"/smt2")
        for smt2_file in glob.glob(folder+"/*.smt2"):
            shutil.move(smt2_file,folder+"/smt2")

    #transform to eq and clean eq
    update_ostrich()
    for folder in leaf_folder_list:
        print("smt2 to eq",folder)
        smt_to_eq_one_folder(folder)
        print("clean eq",folder)
        clean_eq_files(folder)
        get_clean_statistics(folder.replace(bench_folder+"/",""), [folder])

    #statistics
    get_clean_statistics(benchmark, leaf_folder_list)


    #collect cleaned files
    collect_cleaned_files(benchmark,leaf_folder_list)













if __name__ == '__main__':
    main()
