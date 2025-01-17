import sys
import configparser

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
    '''
    unsatcores for proof tree is stored as .unsatcore files need use generate_train_data_from_solver_one_folder.py to generate train data
    one unsatcore is stored in the folder *_unsat_cores, use generate_train_data_from_unsatcores.py to generate train data
    :return:
    '''
    benchmark_name="unsatcores_Benchmark_C_train_10001_60000"
    divided_folder_list=[fo for fo in os.listdir(f"{bench_folder}/{benchmark_name}") if "divided" in os.path.basename(fo)]

    create_folder(f"{bench_folder}/{benchmark_name}/merged_for_proof_tree")
    merged_folder_for_proof_tree=create_folder(f"{bench_folder}/{benchmark_name}/merged_for_proof_tree/UNSAT")
    create_folder(f"{bench_folder}/{benchmark_name}/merged_unsatcores")
    merged_unsatcores_folder=create_folder(f"{bench_folder}/{benchmark_name}/merged_unsatcores/UNSAT")

    no_unsatcore_file_list=[]


    for divided_folder in divided_folder_list:

        for eq_file in glob.glob(f"{bench_folder}/{benchmark_name}/{divided_folder}/UNSAT/*.eq"):
            file_name=strip_file_name_suffix(eq_file)
            #merged_folder_for_proof_tree
            if os.path.exists(f"{file_name}.unsatcore"):
                with open(f"{file_name}.unsatcore", 'r') as f:
                    lines = f.readlines()
                if len(lines)>0:
                    copy_relative_files(prefix=file_name,target_folder=merged_folder_for_proof_tree)
                #copy_relative_files_and_directories(prefix=file_name,target_folder=merged_folder)
                #copy_relative_files(prefix=file_name,target_folder=merged_folder_for_proof_tree)
                else:
                    print(os.path.basename(eq_file),"unsatcore empty")
            else:
                print(os.path.basename(eq_file),"don't have .unsatcore")
            #merged_unsatcores_folder
            if os.path.exists(f"{file_name}_unsat_cores") and glob.glob(f"{file_name}_unsat_cores/*")!=[]:
                #copy_relative_files(prefix=file_name, target_folder=merged_unsatcores_folder)
                if not os.path.exists(f"{file_name}.unsatcore"):
                    # move smallest unsatecore to file .unsatcore
                    smallest_unsatcore_file=sorted(glob.glob(f"{file_name}_unsat_cores/*.eq"),reverse=True)[-1]
                    with open(smallest_unsatcore_file, 'r') as f:
                        lines = f.readlines()
                    if len(lines) > 0:
                        copy_relative_files(prefix=file_name, target_folder=merged_unsatcores_folder)
                        shutil.copy(smallest_unsatcore_file,f"{merged_unsatcores_folder}/{os.path.basename(file_name)}.unsatcore")
                else:
                    pass

            elif os.path.exists(f"{file_name}_unsatcore") and glob.glob(f"{file_name}_unsatcore/*")!=[]:
                copy_relative_files(prefix=file_name, target_folder=merged_unsatcores_folder)
                for file in glob.glob(f"{file_name}_unsatcore/{os.path.basename(file_name)}*"):
                    shutil.copy(file,merged_unsatcores_folder)
                    unsatcore_eq=f"{merged_unsatcores_folder}/{os.path.basename(file_name)}.current_unsatcore"
                    unsatcore_smt2=f"{merged_unsatcores_folder}/{os.path.basename(file_name)}.current_unsatcore.smt2"
                if os.path.exists(unsatcore_eq):
                    os.rename(unsatcore_eq,unsatcore_eq.replace(".current_unsatcore",".unsatcore"))
                if os.path.exists(unsatcore_smt2):
                    os.rename(unsatcore_smt2,unsatcore_smt2.replace(".current_unsatcore.smt2",".unsatcore.smt2"))

            else:
                no_unsatcore_file_list.append(eq_file)
                print(os.path.basename(eq_file),"don't have unsatcore")

    print("no_unsatcore_file_list",len(no_unsatcore_file_list))







if __name__ == '__main__':
    main()
