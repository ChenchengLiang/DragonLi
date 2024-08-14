import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.Constants import project_folder, bench_folder
from src.train_data_collection.utils import dvivde_track_for_cluster
from src.solver.independent_utils import get_folders, strip_file_name_suffix
import os
import shutil
import random
import glob
import zipfile
import tempfile
from tqdm import tqdm

def main():
    # generate track
    track_name = "01_track_multi_word_equations_generated_train_1_40000_for_rank_task_UNSAT_data_extraction_test"
    track_folder = bench_folder + "/" + track_name

    satisfiability = "UNSAT"

    graph_indices = [1, 2, 3, 4, 5]

    divided_folder_list = [train_folder for train_folder in get_folders(track_folder) if "divided" in train_folder]

    # merge UNSAT file
    merged_eq_file_path = f"{track_folder}/{satisfiability}"
    if os.path.exists(merged_eq_file_path):
        shutil.rmtree(merged_eq_file_path)
    os.mkdir(merged_eq_file_path)
    for divided_folder in divided_folder_list:
        copy_files(f"{track_folder}/{divided_folder}/{satisfiability}", merged_eq_file_path)
    print("total files", len(glob.glob(f"{merged_eq_file_path}/*.eq")))

    # merge train.zip
    train_zip_list = [f"{track_folder}/{divided_folder}/train.zip" for divided_folder in divided_folder_list]
    output_zip_path = f"{track_folder}/train.zip"
    merge_zip_files(train_zip_list, output_zip_path)
    clean_zip_files(track_folder,output_zip_path, merged_eq_file_path)

    # merge graph_n.zip
    for graph_index in graph_indices:
        graph_zip_list = [f"{track_folder}/{divided_folder}/graph_{graph_index}.zip" for divided_folder in
                          divided_folder_list]
        output_zip_path = f"{track_folder}/graph_{graph_index}.zip"
        merge_zip_files(graph_zip_list, output_zip_path)

    # find intersection
    graph_lists = []
    for graph_index in graph_indices:
        with zipfile.ZipFile(f"{track_folder}/graph_{graph_index}.zip", 'r') as zfile:
            # Get list of file names
            file_name_list = []
            for name in zfile.namelist():
                file_name = os.path.basename(name).split("@")[0]
                file_name_list.append(file_name)
            graph_lists.append(file_name_list)

    if graph_lists:
        intersection = set(graph_lists[0].copy())
        for s in graph_lists[1:]:
            intersection.intersection_update(s)


    # remove graph files not in intersection
    print("remove graph files not in intersection")
    for graph_index in graph_indices:
        remove_name_list = []
        with zipfile.ZipFile(f"{track_folder}/graph_{graph_index}.zip", 'r') as zfile:
            for name in zfile.namelist():
                file_name = os.path.basename(name).split("@")[0]
                if file_name not in intersection:
                    remove_name_list.append(file_name)

        remove_name_list=list(set(remove_name_list))
        remove_multiple_files_from_zip(track_folder, remove_name_list)


    # remove train files not in intersection
    print("remove train files not in intersection")
    remove_name_list = []
    with zipfile.ZipFile(f"{track_folder}/train.zip", 'r') as zfile:
        for name in zfile.namelist():
            if "@" in name:
                file_name = os.path.basename(name).split("@")[0]
            else:
                file_name = os.path.basename(name).split(".eq")[0]
            if file_name not in intersection and ".answer" not in file_name:
                remove_name_list.append(file_name)

    remove_name_list=list(set(remove_name_list))


    remove_multiple_files_from_zip(track_folder, remove_name_list)

    # remove eq files in UNSAT folder not in intersection
    print("remove eq files in UNSAT folder not in intersection")
    for file in glob.glob(f"{track_folder}/UNSAT/*.eq"):
        file_name = os.path.basename(file).split(".eq")[0]
        if file_name not in intersection:
            os.remove(file)
            os.remove(file.replace(".eq", ".answer"))

    # divide to train and valid folder
    split_to_train_valid_with_zip(source_folder=f"{track_folder}", satisfiability=satisfiability,
                                  train_folder=track_folder + "/train", valid_folder=track_folder + "/valid",
                                  valid_ratio=0.2)

    # collect data
    os.mkdir(f"{track_folder}/extracted_data")
    for divided_folder in divided_folder_list:
        shutil.move(f"{track_folder}/{divided_folder}", f"{track_folder}/extracted_data")
    os.mkdir(f"{track_folder}/merged_data")
    shutil.move(f"{track_folder}/train.zip", f"{track_folder}/merged_data")
    for graph_index in graph_indices:
        shutil.move(f"{track_folder}/graph_{graph_index}.zip", f"{track_folder}/merged_data")
    shutil.move(f"{track_folder}/{satisfiability}", f"{track_folder}/merged_data")
    os.mkdir(f"{track_folder}/train/{satisfiability}")
    for file in glob.glob(f"{track_folder}/train/*"):
        if ".zip" not in file:
            shutil.move(file, f"{track_folder}/train/{satisfiability}")
    os.mkdir(f"{track_folder}/valid/{satisfiability}")
    for file in glob.glob(f"{track_folder}/valid/*"):
        if ".zip" not in file:
            shutil.move(file, f"{track_folder}/valid/{satisfiability}")

    # divide train to multiple chunks
    print("divide train to multiple chunks")

    chunk_size = 500
    folder_counter = 0
    for i, eq_file in enumerate(glob.glob(f"{track_folder}/train/{satisfiability}/*.eq")):
        if i % chunk_size == 0:
            folder_counter += 1
            divided_folder_name = f"{track_folder}/divided_{folder_counter}"
            os.mkdir(divided_folder_name)
            os.mkdir(f"{divided_folder_name}/{satisfiability}")
        file_name = strip_file_name_suffix(eq_file)
        # move files in UNSAT folder
        for f in glob.glob(file_name + ".eq") + glob.glob(file_name + ".answer") + glob.glob(file_name + ".smt2"):
            shutil.copy(f, f"{divided_folder_name}/{satisfiability}")

        # move files in train.zip
        file_list = get_filenames_with_prefix(f"{track_folder}/train/train.zip", os.path.basename(file_name), "train")
        copy_files_between_zips(f"{track_folder}/train/train.zip", f"{divided_folder_name}/train.zip", file_list)
        # move files in graph_n.zip
        for graph_index in graph_indices:
            file_list = get_filenames_with_prefix(f"{track_folder}/train/graph_{graph_index}.zip",
                                                  os.path.basename(file_name),
                                                  f"graph_{graph_index}")
            copy_files_between_zips(f"{track_folder}/train/graph_{graph_index}.zip",
                                    f"{divided_folder_name}/graph_{graph_index}.zip", file_list)

    # handle valid data
    #shutil.move(f"{track_folder}/valid", f"{track_folder}/valid_data")
    shutil.move(f"{track_folder}/valid", f"{track_folder}/divided_0")

    # remove middle files
    shutil.rmtree(f"{track_folder}/train")
    shutil.rmtree(f"{track_folder}/merged_data")


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
        #print(f"Copied {src_file} to {dst_file}")


def clean_zip_files(track_folder,zip_path, merged_eq_file_path):
    all_data_list = []
    for file in os.listdir(merged_eq_file_path):
        if file.endswith(".eq"):
            all_data_list.append("train/" + file.split(".eq")[0])
    with zipfile.ZipFile(zip_path, 'r') as zfile:
        json_data_list = []
        for file in zfile.namelist():
            if file.endswith(".json"):
                json_prefix = file.split("@")[0]
                json_data_list.append(json_prefix)

        zfile.close()

    remove_data_list = [data for data in all_data_list if data not in json_data_list]
    remove_data_list=list(set(remove_data_list))


    remove_multiple_files_from_zip(track_folder,remove_data_list)

    for data in tqdm(remove_data_list, desc="Removing files"):
        for file in glob.glob(f"{merged_eq_file_path}/{os.path.basename(data)}*"):
            os.remove(file)

def remove_multiple_files_from_zip(track_folder,file_list):
    unzip_file(f"{track_folder}/train.zip", f"{track_folder}/train")
    for file_name in file_list:
        file_list_in_zip=glob.glob(f"{track_folder}/train/{file_name}*")
        for file in file_list_in_zip:
            os.remove(file)

    zip_folder(f"{track_folder}/train", f"{track_folder}/train.zip")
    shutil.rmtree(f"{track_folder}/train")


def unzip_file(zip_path, extract_to):
    """Unzip a zip file to a specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def zip_folder(folder_path, output_zip_path):
    """Zip the contents of a folder, including subfolders."""
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Store files with relative paths (not absolute paths)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))
    print(f"Created zip {output_zip_path} from {folder_path}")

def remove_files_from_zip(zip_path, prefix):
    # Create a temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.zip')
    os.close(temp_fd)  # Close the file descriptor

    with zipfile.ZipFile(zip_path, 'r') as src_zip, zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as dest_zip:
        # Filter entries to avoid unnecessary reading
        entries = [item for item in src_zip.infolist() if not item.filename.startswith(prefix)]

        # Use tqdm for progress display
        for item in entries:
            # Open the source file and add it directly to the destination zip
            with src_zip.open(item) as source_file:
                dest_zip.writestr(item, source_file.read(), compress_type=zipfile.ZIP_DEFLATED)

    # Replace the old ZIP file with the new one
    os.replace(temp_path, zip_path)  # More atomic than remove + rename



def merge_zip_files(zip_paths, output_zip_path):
    # Create a temporary directory to store the contents of all zip files
    temp_dir = tempfile.mkdtemp()

    # Process each zip file in the provided list
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, 'r') as zfile:
            # Extract all contents into the temporary directory
            zfile.extractall(temp_dir)

    # Create a new zip file that will contain the merged contents
    with zipfile.ZipFile(output_zip_path, 'w',zipfile.ZIP_DEFLATED) as zfile:
        for root, dirs, files in os.walk(temp_dir):
            for file in tqdm(files,desc="Merging files"):
                # Create the path to file
                file_path = os.path.join(root, file)
                # Create the archive name (relative path within the zip)
                arcname = os.path.relpath(file_path, temp_dir)
                # Add file to zip
                zfile.write(file_path, arcname)

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    print(f"Merged zip created at {output_zip_path}")


def split_to_train_valid_with_zip(source_folder, satisfiability, train_folder, valid_folder, valid_ratio):
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
    print("train_files", len(train_files))
    print("valid_files", len(valid_files))

    # Move files
    for file in valid_files:
        move_files_with_fold(source_folder, file, "valid", valid_folder)

    for file in train_files:
        move_files_with_fold(source_folder, file, "train", train_folder)


def move_files_with_fold(source_folder, file, fold, folder_name):
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


def get_filenames_with_prefix(zip_path, prefix, folder_name):
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
    with zipfile.ZipFile(destination_zip_path, 'a', zipfile.ZIP_DEFLATED) as dest_zip:
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
