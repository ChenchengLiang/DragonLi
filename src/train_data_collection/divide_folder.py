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
    track_name="03_track_generated_eval_1_20000_task_3_continuously_train_337_debug"
    track_folder = bench_folder + "/"+track_name
    print(track_folder)

    # divide tracks
    dvivde_track_for_cluster(track_folder,file_folder="ALL", chunk_size=50)


if __name__ == '__main__':
    main()
