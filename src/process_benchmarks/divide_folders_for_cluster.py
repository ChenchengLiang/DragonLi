
import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

from src.solver.Constants import bench_folder
import glob
import os
import shutil

from src.solver.independent_utils import strip_file_name_suffix

def main():
    folder=bench_folder+"/01_track_generated_SAT_train/ALL"
    chunk_size=100


    folder_counter=0
    all_folder=folder+"/ALL"
    if not os.path.exists(all_folder):
        os.mkdir(all_folder)

    for file in glob.glob(folder+"/*"):
        shutil.move(file,all_folder)


    for i,eq_file in enumerate(glob.glob(all_folder+"/*.eq")):
        if i%chunk_size==0:
            folder_counter += 1
            divided_folder_name=folder+"/divided_"+str(folder_counter)
            os.mkdir(divided_folder_name)
        file_name=strip_file_name_suffix(eq_file)
        for f in glob.glob(file_name+".eq")+glob.glob(file_name+".answer"):
            shutil.copy(f, divided_folder_name)







if __name__ == '__main__':
    main()