from src.process_benchmarks.utils import run_on_one_problem
from src.solver.Constants import project_folder,bench_folder
from src.solver.independent_utils import strip_file_name_suffix,time_it,find_leaf_folders
import glob
import os
import shutil
from src.process_benchmarks.utils import smt_to_eq_one_folder,clean_eq_files
def main():
    benchmark="smtlib/2021-05-26_clean"
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
    for folder in leaf_folder_list:
        print("smt2 to eq",folder)
        smt_to_eq_one_folder(folder)
        print("clean eq",folder)
        clean_eq_files(folder)

    #statistics and collect cleaned files
    total_smt2_files=0
    smt2_to_eq_exception_others=0
    smt2_to_eq_exception_too_many_variables=0
    smt2_to_eq_exceptions_too_many_letters=0
    total_eq_files=0
    empty_eq_files=0
    duplicated_eqs=0
    no_terminals_eqs=0
    no_variables_eqs=0
    total_eq_cleaned_files=0
    total_eq_cleaned_folder=bench_folder+"/"+benchmark+"/total_cleaned_eq_folder"
    os.mkdir(total_eq_cleaned_folder)

    for folder in leaf_folder_list:
        total_smt2_files+=len(glob.glob(folder+"/smt2/*.smt2"))
        smt2_to_eq_exception_others+=len(glob.glob(folder+"/exceptions/others/*.smt2"))
        smt2_to_eq_exception_too_many_variables+=len(glob.glob(folder+"/exceptions/too_many_variables/*.smt2"))
        smt2_to_eq_exceptions_too_many_letters+=len(glob.glob(folder+"/exceptions/too_many_letters/*.smt2"))
        total_eq_files+=len(glob.glob(folder+"/eq/*.eq"))
        empty_eq_files+=len(glob.glob(folder+"/empty_eq/*.eq"))
        duplicated_eqs+=len(glob.glob(folder+"/duplicated_eqs/*.eq"))
        no_terminals_eqs+=len(glob.glob(folder+"/no_terminals/*.eq"))
        no_variables_eqs+=len(glob.glob(folder+"/no_variables/*.eq"))
        total_eq_cleaned_files+=len(glob.glob(folder+"/eq_cleaned/*.eq"))
        #collect all eq_cleaned files
        for eq_file in glob.glob(folder+"/eq_cleaned/*.eq"):
            shutil.copy(eq_file,total_eq_cleaned_folder)


    log_file=bench_folder+"/"+benchmark+"/cleaned_eq_statistics.txt"
    with open(log_file,"w") as f:
        f.write("total_smt2_files:"+str(total_smt2_files)+"\n")
        f.write("smt2_to_eq_exception_others:"+str(smt2_to_eq_exception_others)+"\n")
        f.write("smt2_to_eq_exception_too_many_variables:"+str(smt2_to_eq_exception_too_many_variables)+"\n")
        f.write("smt2_to_eq_exceptions_too_many_letters:"+str(smt2_to_eq_exceptions_too_many_letters)+"\n")
        f.write("total_eq_files:"+str(total_eq_files)+"\n")
        f.write("empty_eq_files:"+str(empty_eq_files)+"\n")
        f.write("duplicated_eqs:"+str(duplicated_eqs)+"\n")
        f.write("no_terminals_eqs:"+str(no_terminals_eqs)+"\n")
        f.write("no_variables_eqs:"+str(no_variables_eqs)+"\n")
        f.write("total_eq_cleaned_files:"+str(total_eq_cleaned_files)+"\n")









if __name__ == '__main__':
    main()
