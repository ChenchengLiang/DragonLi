import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

from src.solver.Constants import project_folder, bench_folder
from src.train_data_collection.utils import dvivde_track_for_cluster
from src.solver.independent_utils import get_folders, strip_file_name_suffix,time_it,flatten_list
import os
import shutil
import random
import glob
import zipfile
import tempfile
from tqdm import tqdm


def main():

    track_name = "04_track_woorpje_train_unsatcores"
    track_folder = bench_folder + "/" + track_name

    satisfiability = "UNSAT"

    graph_indices = [1, 2, 3, 4, 5]
    chunk_size = 5000

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
    # for graph_index in graph_indices:
    #     graph_zip_list = [f"{track_folder}/{divided_folder}/graph_{graph_index}.zip" for divided_folder in
    #                       divided_folder_list]
    #     output_zip_path = f"{track_folder}/graph_{graph_index}.zip"
    #     merge_zip_files(graph_zip_list, output_zip_path)
    #
    # # find intersection
    # graph_lists = []
    # for graph_index in graph_indices:
    #     with zipfile.ZipFile(f"{track_folder}/graph_{graph_index}.zip", 'r') as zfile:
    #         # Get list of file names
    #         file_name_list = []
    #         for name in zfile.namelist():
    #             file_name = os.path.basename(name).split("@")[0]
    #             file_name_list.append(file_name)
    #         graph_lists.append(file_name_list)
    #
    # if graph_lists:
    #     intersection = set(graph_lists[0].copy())
    #     for s in graph_lists[1:]:
    #         intersection.intersection_update(s)
    #
    #
    # # remove graph files not in intersection
    # print("remove graph files not in intersection")
    # for graph_index in graph_indices:
    #     remove_name_list = []
    #     with zipfile.ZipFile(f"{track_folder}/graph_{graph_index}.zip", 'r') as zfile:
    #         for name in zfile.namelist():
    #             file_name = os.path.basename(name).split("@")[0]
    #             if file_name not in intersection:
    #                 remove_name_list.append(file_name)
    #
    #     remove_name_list=list(set(remove_name_list))
    #     remove_multiple_files_from_zip(track_folder, remove_name_list)
    #
    #
    # # remove train files not in intersection
    # print("remove train files not in intersection")
    # remove_name_list = []
    # with zipfile.ZipFile(f"{track_folder}/train.zip", 'r') as zfile:
    #     for name in zfile.namelist():
    #         if "@" in name:
    #             file_name = os.path.basename(name).split("@")[0]
    #         else:
    #             file_name = os.path.basename(name).split(".eq")[0]
    #         if file_name not in intersection and ".answer" not in file_name:
    #             remove_name_list.append(file_name)
    #
    # remove_name_list=list(set(remove_name_list))
    #
    #
    # remove_multiple_files_from_zip(track_folder, remove_name_list)
    #
    # # remove eq files in UNSAT folder not in intersection
    # print("remove eq files in UNSAT folder not in intersection")
    # for file in glob.glob(f"{track_folder}/UNSAT/*.eq"):
    #     file_name = os.path.basename(file).split(".eq")[0]
    #     if file_name not in intersection:
    #         os.remove(file)
    #         os.remove(file.replace(".eq", ".answer"))

    # collect data
    print("collect data extracted_data")
    os.mkdir(f"{track_folder}/extracted_data")
    for divided_folder in divided_folder_list:
        shutil.move(f"{track_folder}/{divided_folder}", f"{track_folder}/extracted_data")
    shutil.rmtree(f"{track_folder}/extracted_data")

    # divide to train and valid folder

    split_to_train_valid_with_zip(source_folder=f"{track_folder}", satisfiability=satisfiability,
                                  train_folder=track_folder + "/train", valid_folder=track_folder + "/valid",
                                  valid_ratio=0.2)

    # collect data
    print("collect data merged_data")
    os.mkdir(f"{track_folder}/merged_data")
    shutil.move(f"{track_folder}/train.zip", f"{track_folder}/merged_data")
    # for graph_index in graph_indices:
    #     shutil.move(f"{track_folder}/graph_{graph_index}.zip", f"{track_folder}/merged_data")
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
    folder_counter = 0
    total_file_list=[]
    current_file_list=[]
    file_list_blocks=[]
    divided_folder_name_list=[]
    unzip_file(f"{track_folder}/train/train.zip", f"{track_folder}/temp")

    for i, eq_file in enumerate(glob.glob(f"{track_folder}/train/{satisfiability}/*.eq")):
        if i % chunk_size == 0:
            folder_counter += 1
            divided_folder_name = f"{track_folder}/divided_{folder_counter}"
            os.mkdir(divided_folder_name)
            os.mkdir(f"{divided_folder_name}/{satisfiability}")
            divided_folder_name_list.append(divided_folder_name)
        file_name = strip_file_name_suffix(eq_file)
        # move files in UNSAT folder
        for f in glob.glob(file_name + ".eq") + glob.glob(file_name + ".answer") + glob.glob(file_name + ".smt2"):
            shutil.copy(f, f"{divided_folder_name}/{satisfiability}")

        # move files in train.zip
        base_file_name = os.path.basename(file_name)
        all_filenames = glob.glob(f"{track_folder}/temp/train/*")
        all_filenames = [os.path.basename(name) for name in all_filenames]
        matching_filenames = [name for name in all_filenames if name.startswith(f"{base_file_name}")]
        matching_filenames=[f"train/{name}" for name in matching_filenames]
        total_file_list.append(matching_filenames)
        current_file_list.append(matching_filenames)

    shutil.rmtree(f"{track_folder}/temp")

    folder_counter=0
    current_file_list=[]
    for i,file_list in enumerate(total_file_list):
        if i % chunk_size == 0 and folder_counter>=1:
            file_list_blocks.append(current_file_list)
            current_file_list=[]
        if i % chunk_size == 0:
            folder_counter += 1
        current_file_list.append(file_list)
    file_list_blocks.append(current_file_list)

    for divided_folder_name,file_list in zip(divided_folder_name_list,file_list_blocks):
        file_list=flatten_list(file_list)
        copy_files_between_zips(f"{track_folder}/train/train.zip", f"{divided_folder_name}/train.zip", file_list)

    # handle valid data
    shutil.move(f"{track_folder}/valid", f"{track_folder}/valid_data")
    #shutil.move(f"{track_folder}/valid", f"{track_folder}/divided_0")

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


    for data in tqdm(remove_data_list, desc="Removing files not generated trained data"):
        for file in glob.glob(f"{merged_eq_file_path}/{os.path.basename(data)}*"):
            os.remove(file)

@time_it
def remove_multiple_files_from_zip(track_folder,file_list):
    unzip_file(f"{track_folder}/train.zip", f"{track_folder}/train")
    total_file_list=[]
    for file_name in tqdm(file_list,desc="Removing files"):
        file_list_in_zip=glob.glob(f"{track_folder}/train/{file_name}*")
        total_file_list.append(file_list_in_zip)
    total_file_list=flatten_list(total_file_list)

    total_file_list=list(set(total_file_list))
    for file in total_file_list:
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

@time_it
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


    #valid
    unzip_file(f"{source_folder}/train.zip", f"{source_folder}/temp")

    total_file_list=[]
    for file in valid_files:
        file_name = os.path.basename(file)
        shutil.copy(file, os.path.join(valid_folder, file_name))
        #answer_file = file.replace(".eq", ".answer")
        #shutil.copy(answer_file, os.path.join(valid_folder, os.path.basename(answer_file)))
        file_name_before_at = file_name.split(".eq")[0]
        all_filenames=get_filenames(f"{source_folder}/temp/train")
        all_filenames = [os.path.basename(name) for name in all_filenames]
        matching_filenames = [name for name in all_filenames if name.startswith(f"{file_name_before_at}")]
        total_file_list.append(matching_filenames)
    total_file_list=flatten_list(total_file_list)
    total_file_list = [f"train/{name}" for name in total_file_list]
    copy_files_between_zips(f"{source_folder}/train.zip", f"{source_folder}/valid/train.zip", total_file_list)

    #train
    total_file_list=[]
    for file in train_files:
        file_name = os.path.basename(file)
        shutil.copy(file, os.path.join(train_folder, file_name))
        #answer_file = file.replace(".eq", ".answer")
        #shutil.copy(answer_file, os.path.join(train_folder, os.path.basename(answer_file)))
        file_name_before_at = file_name.split(".eq")[0]
        all_filenames = get_filenames(f"{source_folder}/temp/train")
        all_filenames=[os.path.basename(name) for name in all_filenames]
        matching_filenames = [name for name in all_filenames if name.startswith(f"{file_name_before_at}")]
        total_file_list.append(matching_filenames)
    total_file_list=flatten_list(total_file_list)
    total_file_list= [f"train/{name}" for name in total_file_list]
    copy_files_between_zips(f"{source_folder}/train.zip", f"{source_folder}/train/train.zip", total_file_list)

    shutil.rmtree(f"{source_folder}/temp")




@time_it
def copy_files_between_zips(source_zip_path, destination_zip_path, file_names):
    # Create temporary directories to extract the ZIPs
    source_temp_dir = tempfile.mkdtemp()
    destination_temp_dir = tempfile.mkdtemp()

    # Extract the source ZIP
    with zipfile.ZipFile(source_zip_path, 'r') as source_zip:
        source_zip.extractall(source_temp_dir)

    # Extract the destination ZIP
    if os.path.exists(destination_zip_path):
        with zipfile.ZipFile(destination_zip_path, 'r') as destination_zip:
            destination_zip.extractall(destination_temp_dir)
    else:
        pass

    # Copy specified files from the source to the destination
    for file_name in file_names:
        source_file_path = os.path.join(source_temp_dir, file_name)
        destination_file_path = os.path.join(destination_temp_dir, file_name)
        if os.path.exists(source_file_path):
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
            shutil.copy2(source_file_path, destination_file_path)
        else:
            print(f"Warning: {file_name} not found in {source_zip_path}")

    # Re-zip the destination directory
    with zipfile.ZipFile(destination_zip_path, 'w', zipfile.ZIP_DEFLATED) as dest_zip:
        for root, dirs, files in os.walk(destination_temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, destination_temp_dir)
                dest_zip.write(file_path, arcname)

    # Clean up temporary directories
    shutil.rmtree(source_temp_dir)
    shutil.rmtree(destination_temp_dir)


def get_filenames(folder_path):
    """Get all file names in the specified directory with a more efficient method."""
    files = []
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_file():  # Make sure it's a file
                files.append(entry.path)
    return files

if __name__ == '__main__':
    main()
