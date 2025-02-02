import configparser
import os
import sys

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

import random
import string
from src.solver.Constants import bench_folder
from copy import deepcopy
from src.solver.independent_utils import remove_duplicates, identify_available_capitals, strip_file_name_suffix
from src.train_data_collection.utils import dvivde_track_for_cluster
from src.train_data_collection.generate_track_utils import save_equations, generate_one_track_1, generate_conjunctive_track_03


def main():
    # generate track
    start_idx = 30001
    end_idx = 60000
    track_name = f"Benchmark_C_train_{start_idx}_{end_idx}"
    # track_name = f"01_track_multi_word_equations_eq_2_50_generated_train_{start_idx}_{end_idx}"
    # track_name = f"Benchmark_C_train_eq_1_100_{start_idx}_{end_idx}"
    #track_name = f"Benchmark_D_max_replace_length_bounded_16_train_{start_idx}_{end_idx}"  # generate_one_track_4_v4
    track_folder = bench_folder + "/" + track_name
    file_name_list = save_equations(start_idx, end_idx, track_folder, track_name, generate_conjunctive_track_03)
    #file_name_list=save_equations(start_idx, end_idx, track_folder, track_name, generate_one_track_4)
    # save_equations(start_idx, end_idx, track_folder, track_name, generate_one_track_4_v2)
    # save_equations(start_idx, end_idx, track_folder, track_name, generate_one_track_4_v3)
    # save_equations(start_idx, end_idx, track_folder, track_name, generate_one_track_1_v2)
    #save_equations(start_idx, end_idx, track_folder, track_name, generate_one_track_4_v4)

    #save_equations(start_idx, end_idx, track_folder, track_name, generate_one_track_4_v5)

    print("data generating finished")

    print("statistic files reading finished")

    # divide tracks
    dvivde_track_for_cluster(track_folder, chunk_size=50)

    print("done")



if __name__ == '__main__':
    main()
