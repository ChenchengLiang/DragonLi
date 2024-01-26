from src.process_benchmarks.utils import run_on_one_problem
from src.solver.Constants import project_folder,bench_folder
from src.solver.independent_utils import strip_file_name_suffix,time_it,find_leaf_folders
import glob
import os
import shutil
from src.process_benchmarks.utils import smt_to_eq_one_folder,clean_eq_files
def main():
    leaf_folder_list=find_leaf_folders(bench_folder+"/"+"smtlib/test")
    #leaf_folder_list= [os.path.dirname(f) for f in leaf_folder_list if os.path.basename(f)=="smt2"]
    for folder in leaf_folder_list:
        print(folder)
        if os.path.exists(folder+"/smt2"):
            shutil.rmtree(folder+"/smt2")
        os.mkdir(folder+"/smt2")
        for smt2_file in glob.glob(folder+"/*.smt2"):
            shutil.move(smt2_file,folder+"/smt2")

    for folder in leaf_folder_list:
        print("smt2 to eq",folder)
        smt_to_eq_one_folder(folder)
        print("clean eq",folder)
        clean_eq_files(folder)





if __name__ == '__main__':
    main()
