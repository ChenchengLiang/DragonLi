from src.process_benchmarks.utils import run_on_one_problem
from src.solver.Constants import project_folder,bench_folder
from src.solver.independent_utils import strip_file_name_suffix,time_it,find_leaf_folders
import glob
import os
import shutil
import subprocess
from tqdm import tqdm
from src.process_benchmarks.utils import smt_to_eq_one_folder
def main():

    folder=bench_folder+"/smtlib/2023-05-08/non-incremental/QF_S/2019-Jiang/slog"
    smt_to_eq_one_folder(folder)

if __name__ == '__main__':
    main()
