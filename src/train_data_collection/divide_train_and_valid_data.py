import sys
import configparser
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
    # generate track
    track_name="divide_test"
    track_folder = bench_folder + "/"+track_name

    # divide to train and valid folder
    split_to_train_valid_set(source_folder=track_folder+"/ALL",train_folder=track_folder+"/train",valid_folder=track_folder+"/valid",valid_ratio=0.2)

    # divide train to multiple folders
    #dvivde_track_for_cluster(track_folder,file_folder="ALL", chunk_size=300)

    #divided_folder_list = get_folders(track_folder + "/ALL")



def split_to_train_valid_set(source_folder, train_folder, valid_folder, valid_ratio):
    """
    Splits files from the source folder into train and valid folders based on the specified ratio.

    :param source_folder: Path to the source folder containing files.
    :param train_folder: Path to the folder where training files will be moved.
    :param valid_folder: Path to the folder where validation files will be moved.
    :param valid_ratio: The ratio of files to be moved to the validation folder.
    """
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(valid_folder):
        os.makedirs(valid_folder)

    # Get all file names in the source folder
    files = glob.glob(source_folder+"/*.eq")
    random.shuffle(files)

    # Calculate split index
    split_index = int(len(files) * valid_ratio)

    # Split files
    valid_files = files[:split_index]
    train_files = files[split_index:]
    print(len(train_files),train_files[0])
    print(len(valid_files),valid_files[0])


    # Move files
    for file in valid_files:
        shutil.copy(file, os.path.join(valid_folder, os.path.basename(file)))
        answer_file = strip_file_name_suffix(file)+".answer"
        shutil.copy(answer_file, os.path.join(valid_folder,os.path.basename(answer_file)))
    for file in train_files:
        shutil.copy(file, os.path.join(train_folder, os.path.basename(file)))
        answer_file = strip_file_name_suffix(file)+".answer"
        shutil.copy(answer_file, os.path.join(train_folder,os.path.basename(answer_file)))


if __name__ == '__main__':
    main()
