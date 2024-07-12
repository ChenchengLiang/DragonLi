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
    track_name="01_track_multi_word_equations_generated_train_1_40000_for_rank_task_2"
    track_folder = bench_folder + "/"+track_name

    satisfiability="UNSAT"

    # divide to train and valid folder
    split_to_train_valid_set(source_folder=f"{track_folder}/{satisfiability}",train_folder=track_folder+"/train",valid_folder=track_folder+"/valid",valid_ratio=0.2)
    #handle valid folder
    os.mkdir(track_folder+"/valid_data")
    shutil.move(track_folder+"/valid",f"{track_folder}/valid_data/{satisfiability}")


    # divide train to multiple folders
    dvivde_track_for_cluster(track_folder,file_folder="train", chunk_size=20000)

    divided_folder_list = [train_folder for train_folder in get_folders(track_folder + "/train") if "divided" in train_folder]
    print(divided_folder_list)

    for divided_folder in divided_folder_list:
        os.mkdir(track_folder+"/"+divided_folder)
    for divided_folder in divided_folder_list:
        shutil.move(track_folder+"/train/"+divided_folder,f"{track_folder}/{divided_folder}/{satisfiability}")

    shutil.rmtree(track_folder+"/train")



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
    print("train_files",len(train_files))
    print("valid_files",len(valid_files))


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
