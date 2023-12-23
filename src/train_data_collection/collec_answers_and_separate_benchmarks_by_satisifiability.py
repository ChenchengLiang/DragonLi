import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

from src.solver.Constants import project_folder,bench_folder,SAT,UNSAT,UNKNOWN
from src.solver.independent_utils import strip_file_name_suffix
import shutil
import glob

def main():

    #collect answers from divided folders
    benchmark="01_track_multi_word_equations_generated_train_1_40000_new"
    benchmark_folder = bench_folder + "/"+benchmark+"/ALL"

    folder_number = sum([1 for fo in os.listdir(benchmark_folder) if "divided" in os.path.basename(fo)])
    for i in range(folder_number):
        divided_folder_index = i + 1
        for a in glob.glob(benchmark_folder + "/divided_" + str(divided_folder_index) + "/*.answer"):
            # print(a)
            shutil.copy(a, benchmark_folder + "/ALL")


    #separate to SAT UNSAT UNKNOWN
    benchmark_folder=bench_folder+"/"+benchmark

    #create folders
    sat_folder=benchmark_folder+"/SAT"
    unsat_folder = benchmark_folder + "/UNSAT"
    unknown_folder = benchmark_folder + "/UNKNOWN"
    for folder in [sat_folder,unsat_folder,unknown_folder]:
        if os.path.exists(folder)==False:
            os.mkdir(folder)


    #separate files according to answers
    for file in glob.glob(benchmark_folder+"/ALL/ALL/*.eq"):
        file_name =  strip_file_name_suffix(file)
        #read .answer file
        answer_file=file_name+".answer"
        if os.path.exists(answer_file):
            with open(answer_file) as f:
                answer = f.read()
                if answer == SAT:
                    shutil.copy(file,sat_folder)
                    shutil.copy(answer_file,sat_folder)
                elif answer == UNSAT:
                    shutil.copy(file, unsat_folder)
                    shutil.copy(answer_file, unsat_folder)
                else:
                    shutil.copy(file, unknown_folder)
                    shutil.copy(answer_file, unknown_folder)
        else:
            print("error: answer file does not exist")
            exit(1)

    #remove original files
    # for file in glob.glob(benchmark_folder+"/*.eq"):
    #     if os.path.exists(file):
    #         shutil.copy(file,all_folder)
    #         os.remove(file)
    # for file in glob.glob(benchmark_folder + "/*.answer"):
    #     if os.path.exists(file):
    #         shutil.copy(file,all_folder)
    #         os.remove(file)








if __name__ == '__main__':
    main()