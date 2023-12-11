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


def main():
    # #transform one benchmark with divived folders
    # folder="03_track_generated_eval_20000_21000/ALL"
    # folder_number = sum([1 for fo in os.listdir(bench_folder + "/"+folder) if "divided" in os.path.basename(fo)])
    # for i in range(folder_number):
    #     divided_folder_index=i+1
    #     for file_path in glob.glob(bench_folder + "/"+folder+"/divided_"+str(divided_folder_index)+"/*.eq"):
    #         one_eq_file_to_smt2(file_path)

    #transform one file
    one_eq_file_to_smt2("/home/cheli243/Desktop/CodeToGit/string-equation-solver/Woorpje_benchmarks/examples/multi_eqs/test3.eq")


if __name__ == '__main__':
    main()
