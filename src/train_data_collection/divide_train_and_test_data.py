import sys
import configparser

from src.process_benchmarks.eq2smt_utils import one_eq_file_to_smt2

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.Constants import project_folder, bench_folder
from src.train_data_collection.utils import dvivde_track_for_cluster
from src.solver.independent_utils import get_folders,strip_file_name_suffix
import os
import shutil
import random
import glob


def main():
    benchmark="smtlib/2023-05-05_clean/total_cleaned_eq_folder_without_woorpje"
    benchmark_folder = bench_folder + "/"+benchmark + "/ALL"

    #transform to smt2 file
    exception_folder=f"{bench_folder}/{benchmark}/eq_to_smt2_exception"
    os.mkdir(exception_folder)
    for file in glob.glob(benchmark_folder+"/*.eq"):
        try:
            one_eq_file_to_smt2(file)
        except:
            shutil.move(file,exception_folder)


    #divide to train and eval
    train_folder=bench_folder + "/"+benchmark + "_train"
    eval_folder=bench_folder + "/"+benchmark + "_eval"
    os.mkdir(train_folder)
    os.mkdir(eval_folder)
    train_all_folder=train_folder+"/ALL"
    eval_all_folder=eval_folder+"/ALL"
    os.mkdir(train_all_folder)
    os.mkdir(eval_all_folder)

    all_files=glob.glob(benchmark_folder+"/*.eq")
    random.seed(42)
    random.shuffle(all_files)
    random.shuffle(all_files)

    #divide to train and eval by 8:2
    train_files=all_files[:int(len(all_files)*0.8)]
    eval_files=all_files[int(len(all_files)*0.8):]


    for file in train_files:
        shutil.copy(file,train_all_folder)
        shutil.copy(strip_file_name_suffix(file)+".smt2",train_all_folder)

    for file in eval_files:
        shutil.copy(file,eval_all_folder)
        shutil.copy(strip_file_name_suffix(file) + ".smt2", eval_all_folder)

    # divide for clusters
    dvivde_track_for_cluster(train_folder, file_folder="ALL", chunk_size=50)
    dvivde_track_for_cluster(eval_folder, file_folder="ALL", chunk_size=50)



if __name__ == '__main__':
    main()






