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
    for file_path in glob.glob(bench_folder + "/to_smt/*.eq"):
        one_eq_file_to_smt2(file_path)




if __name__ == '__main__':
    main()
