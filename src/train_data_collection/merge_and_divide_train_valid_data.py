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
import zipfile
import tempfile

def main():
    # generate track
    track_name="01_track_multi_word_equations_generated_train_1_40000_for_rank_task_UNSAT_data_extraction_test"
    track_folder = bench_folder + "/"+track_name

    satisfiability="UNSAT"

    divided_folder_list = [train_folder for train_folder in get_folders(track_folder) if "divided" in train_folder]


    #merge UNSAT file
    merged_eq_file_path=f"{track_folder}/{satisfiability}"
    if os.path.exists(merged_eq_file_path):
        shutil.rmtree(merged_eq_file_path)
    os.mkdir(merged_eq_file_path)
    for divided_folder in divided_folder_list:
        copy_files(f"{track_folder}/{divided_folder}/{satisfiability}",merged_eq_file_path)
    print("total files",len(glob.glob(f"{merged_eq_file_path}/*.eq")))


    #merge train.zip
    train_zip_list=[f"{track_folder}/{divided_folder}/train.zip" for divided_folder in divided_folder_list]
    output_zip_path=f"{track_folder}/train.zip"
    merge_zip_files(train_zip_list, output_zip_path)
    clean_zip_files(output_zip_path,merged_eq_file_path)

    #merge graph_n.zip
    for graph_index in [1,2,3,4,5]:
        graph_zip_list = [f"{track_folder}/{divided_folder}/graph_{graph_index}.zip" for divided_folder in divided_folder_list]
        output_zip_path = f"{track_folder}/graph_{graph_index}.zip"
        merge_zip_files(graph_zip_list, output_zip_path)

    graph_lists = []
    for graph_index in [1, 2, 3, 4, 5]:
        with zipfile.ZipFile(f"{track_folder}/graph_{graph_index}.zip", 'r') as zfile:
            # Get list of file names
            file_name_list = []
            for name in zfile.namelist():
                file_name = os.path.basename(name).split("@")[0]
                file_name_list.append(file_name)
            graph_lists.append(file_name_list)
    # find intersection
    if graph_lists:
        intersection = set(graph_lists[0].copy())
        for s in graph_lists[1:]:
            intersection.intersection_update(s)

    # remove graph files not in intersection
    for graph_index in [1, 2, 3, 4, 5]:
        remove_name_list = []
        with zipfile.ZipFile(f"{track_folder}/graph_{graph_index}.zip", 'r') as zfile:
            for name in zfile.namelist():
                file_name = os.path.basename(name).split("@")[0]
                if file_name not in intersection:
                    remove_name_list.append(remove_name_list)
        for remove_name in remove_name_list:
            remove_files_from_zip(f"{track_folder}/graph_{graph_index}.zip",
                                  f"graph_{graph_index}/{remove_name}")

    #remove train files not in intersection
    remove_name_list = []
    with zipfile.ZipFile(f"{track_folder}/train.zip", 'r') as zfile:
        for name in zfile.namelist():
            if "@" in name:
                file_name = os.path.basename(name).split("@")[0]
            else:
                file_name = os.path.basename(name).split(".eq")[0]
            if file_name not in intersection:
                remove_name_list.append(remove_name_list)
    for remove_name in remove_name_list:
        remove_files_from_zip(f"{track_folder}/train.zip",
                              f"train/{remove_name}")

    # remove eq files in UNSAT folder not in intersection
    for file in glob.glob(f"{track_folder}/UNSAT/*.eq"):
        file_name = os.path.basename(file).split(".eq")[0]
        if file_name not in intersection:
            os.remove(file)
            os.remove(file.replace(".eq", ".answer"))


    # divide to train and valid folder
    split_to_train_valid_with_zip(source_folder=f"{track_folder}",satisfiability=satisfiability,train_folder=track_folder+"/train",valid_folder=track_folder+"/valid",valid_ratio=0.2)


    #collect data
    os.mkdir(f"{track_folder}/extracted_data")
    for divided_folder in divided_folder_list:
        shutil.move(f"{track_folder}/{divided_folder}",f"{track_folder}/extracted_data")
    os.mkdir(f"{track_folder}/merged_data")
    shutil.move(f"{track_folder}/train.zip",f"{track_folder}/merged_data")
    for graph_index in [1,2,3,4,5]:
        shutil.move(f"{track_folder}/graph_{graph_index}.zip",f"{track_folder}/merged_data")
    shutil.move(f"{track_folder}/UNSAT",f"{track_folder}/merged_data")



    # # divide train to multiple chunks
    # dvivde_track_for_cluster(track_folder,file_folder="train", chunk_size=20000)
    #
    # divided_folder_list = [train_folder for train_folder in get_folders(track_folder + "/train") if "divided" in train_folder]
    # print(divided_folder_list)
    #
    # for divided_folder in divided_folder_list:
    #     os.mkdir(track_folder+"/"+divided_folder)
    # for divided_folder in divided_folder_list:
    #     shutil.move(track_folder+"/train/"+divided_folder,f"{track_folder}/{divided_folder}/{satisfiability}")
    #
    # shutil.rmtree(track_folder+"/train")


def copy_files(source_dir, destination_dir):
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print("Source directory does not exist.")
        return

    # Check if destination directory exists, if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # List all files (not directories) in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Copy each file to the destination directory
    for file in files:
        src_file = os.path.join(source_dir, file)
        dst_file = os.path.join(destination_dir, file)
        shutil.copy(src_file, dst_file)
        print(f"Copied {src_file} to {dst_file}")


def clean_zip_files(zip_path,merged_eq_file_path):
    all_data_list=[]
    for file in os.listdir(merged_eq_file_path):
        if file.endswith(".eq"):
            all_data_list.append("train/"+file.split(".eq")[0])
    with zipfile.ZipFile(zip_path, 'r') as zfile:
        json_data_list=[]
        for file in zfile.namelist():
            if file.endswith(".json"):
                json_prefix=file.split("@")[0]
                json_data_list.append(json_prefix)

        zfile.close()


    remove_data_list=[data for data in all_data_list if data not in json_data_list]

    for data in remove_data_list:
        remove_files_from_zip(zip_path, data)
        for file in glob.glob(f"{merged_eq_file_path}/{os.path.basename(data)}*"):
            os.remove(file)



def remove_files_from_zip(zip_path, prefix):
    # Create a temporary file
    temp_fd, temp_path = tempfile.mkstemp()
    os.close(temp_fd)  # Close the file descriptor

    # Create a new zip file excluding files with the specified prefix
    with zipfile.ZipFile(zip_path, 'r') as zfile, zipfile.ZipFile(temp_path, 'w') as temp_zip:
        # Iterate through each file in the original ZIP
        for item in zfile.infolist():
            if not item.filename.startswith(prefix):
                # Read the file data from the original ZIP
                data = zfile.read(item.filename)
                # Write the file to the new ZIP if it does not start with the prefix
                temp_zip.writestr(item, data)

    # Replace the old ZIP file with the new one
    os.remove(zip_path)  # Remove the original ZIP file
    os.rename(temp_path, zip_path)  # Rename the temporary ZIP file to the original file name


def merge_zip_files(zip_paths, output_zip_path):
    # Create a temporary directory to store the contents of all zip files
    temp_dir = tempfile.mkdtemp()

    # Process each zip file in the provided list
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, 'r') as zfile:
            # Extract all contents into the temporary directory
            zfile.extractall(temp_dir)

    # Create a new zip file that will contain the merged contents
    with zipfile.ZipFile(output_zip_path, 'w') as zfile:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                # Create the path to file
                file_path = os.path.join(root, file)
                # Create the archive name (relative path within the zip)
                arcname = os.path.relpath(file_path, temp_dir)
                # Add file to zip
                zfile.write(file_path, arcname)

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    print(f"Merged zip created at {output_zip_path}")

def split_to_train_valid_with_zip(source_folder, satisfiability,train_folder, valid_folder, valid_ratio):
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
    files = glob.glob(f"{source_folder}/{satisfiability}/*.eq")
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
        move_files_with_fold(source_folder, file, "valid", valid_folder)

    for file in train_files:
        move_files_with_fold(source_folder, file, "train", train_folder)

def move_files_with_fold(source_folder,file,fold,folder_name):
    file_name = os.path.basename(file)
    shutil.copy(file, os.path.join(folder_name, file_name))
    answer_file = file.replace(".eq", ".answer")
    shutil.copy(answer_file, os.path.join(folder_name, os.path.basename(answer_file)))

    file_name_before_at = file_name.split(".eq")[0]
    file_list = get_filenames_with_prefix(f"{source_folder}/train.zip", file_name_before_at, "train")
    copy_files_between_zips(f"{source_folder}/train.zip", f"{source_folder}/{fold}/train.zip", file_list)

    for graph_index in [1, 2, 3, 4, 5]:
        file_list = get_filenames_with_prefix(f"{source_folder}/graph_{graph_index}.zip", file_name_before_at,
                                              f"graph_{graph_index}")
        copy_files_between_zips(f"{source_folder}/graph_{graph_index}.zip",
                                f"{source_folder}/{fold}/graph_{graph_index}.zip", file_list)


def get_filenames_with_prefix(zip_path, prefix,folder_name):
    # List to store the names of files that match the prefix
    matching_filenames = []

    # Open the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zfile:
        # Get all file names in the ZIP file
        all_filenames = zfile.namelist()
        # Filter filenames that start with the specified prefix
        matching_filenames = [name for name in all_filenames if name.startswith(f"{folder_name}/{prefix}")]

    return matching_filenames
def copy_files_between_zips(source_zip_path, destination_zip_path, file_names):
    # Temporary storage for extracted files
    temp_directory = "temp_extracted_files"
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    # Open the source ZIP file
    with zipfile.ZipFile(source_zip_path, 'r') as source_zip:
        # Extract specific files
        for file_name in file_names:
            source_zip.extract(file_name, temp_directory)

    # Open the destination ZIP file in append mode
    with zipfile.ZipFile(destination_zip_path, 'a') as dest_zip:
        # Add files from the temporary directory to the destination ZIP
        for file_name in file_names:
            # Construct the path to the extracted file
            file_path = os.path.join(temp_directory, file_name)
            # Add the file to the ZIP
            dest_zip.write(file_path, file_name)

    # Clean up extracted files by removing them
    shutil.rmtree(temp_directory)
if __name__ == '__main__':
    main()
