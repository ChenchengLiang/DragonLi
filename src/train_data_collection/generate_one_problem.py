
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path', 'local')
sys.path.append(path)

import argparse


from src.train_data_collection.generate_track_utils import generate_one_track_4, generate_one_track_4_v2, \
    generate_one_track_4_v5, generate_conjunctive_track_03


def main():
    # parse argument
    arg_parser = argparse.ArgumentParser(description='Process command line arguments.')
    arg_parser.add_argument('benchmark', type=str, default=None,
                            help='benchmark name')
    args = arg_parser.parse_args()
    # Accessing the arguments
    benchmark = args.benchmark



    benchmark_generator_map={"A1":generate_one_track_4,"A2":generate_one_track_4_v5,
                             "B":generate_one_track_4_v2,"C":generate_conjunctive_track_03}
    file_name="webapp/user-input.eq"
    index=1
    record_track_info=False
    equation_str, variable_list, terminal_list, eq_list=benchmark_generator_map[benchmark](file_name,index,record_track_info=record_track_info)
    print(equation_str)

    with open(file_name, 'w') as file:
        file.write(equation_str)


if __name__ == '__main__':
    main()