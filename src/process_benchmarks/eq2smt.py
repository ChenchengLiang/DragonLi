import glob
import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.Constants import bench_folder, project_folder
from src.process_benchmarks.eq2smt_utils import one_eq_file_to_smt2

import shutil

def main():
    #transform one benchmark with divived folders
    # folder="05_track/ALL"
    # folder_number = sum([1 for fo in os.listdir(bench_folder + "/"+folder) if "divided" in os.path.basename(fo)])
    # for i in range(folder_number):
    #     divided_folder_index=i+1
    #     for file_path in glob.glob(bench_folder + "/"+folder+"/divided_"+str(divided_folder_index)+"/*.eq"):
    #         one_eq_file_to_smt2(file_path)

    #transform one file
    exception_folder=bench_folder+"/kaluza/all_eq_cleaned/eq_to_smt2_exception"
    if os.path.exists(exception_folder):
        shutil.rmtree(exception_folder)
    os.mkdir(exception_folder)

    for file in glob.glob(bench_folder+"/kaluza/all_eq_cleaned/eqs/*.eq"):
        try:
            one_eq_file_to_smt2(file)
        except:
            shutil.move(file,exception_folder)


if __name__ == '__main__':
    main()
