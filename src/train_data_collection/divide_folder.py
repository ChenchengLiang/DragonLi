import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)


from src.solver.Constants import project_folder, bench_folder

from src.train_data_collection.utils import dvivde_track_for_cluster



def main():
    # generate track
    track_name="01_track_multi_word_equations_generated_train_1_40000_new"
    track_folder = bench_folder + "/"+track_name
    print(track_folder)

    # divide tracks
    dvivde_track_for_cluster(track_folder,file_folder="SAT_divided", chunk_size=300)


if __name__ == '__main__':
    main()
