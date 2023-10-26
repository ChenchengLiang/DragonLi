import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

import csv
import glob
import os
from src.solver.independent_utils import strip_file_name_suffix
from src.process_benchmarks.utils import summary_one_track
from src.solver.Constants import project_folder,bench_folder


def main():
    summary_folder = project_folder+"/src/process_benchmarks/summary"

    solver_param_list = [["this", ["fixed"]],
                         ["this", ["random"]],
                         ["this", ["gnn", "--graph_type graph_1"]],
                         ["this", ["gnn", "--graph_type graph_2"]],
                         # ["woorpje",[]],
                         # ["z3",[]],
                         # ["ostrich",[]],
                         # ["cvc5",[]],
                         ]



    benchmark_dict = {
        # "test_track": bench_folder + "/test",
        "example_track": bench_folder + "/examples",
        # "track_01": bench_folder + "/01_track",
        # "g_track_01_sat":bench_folder + "/01_track_generated/SAT",
        # "g_track_01_mixed": bench_folder + "/01_track_generated/mixed",
        # "g_track_01_eval":bench_folder + "/01_track_generated_eval_data",
        # "track_02": bench_folder + "/02_track",
        # "track_03": bench_folder + "/03_track",
        # "track_04": bench_folder + "/04_track",
        # "track_05": bench_folder + "/05_track",
    }


    #summary cross solvers
    # for track in ["track_01","track_02","track_03","g_track_01"]:
    #     summary_file_dict={"this":"this_"+track+"_summary.csv",
    #                        "woorpje":"woorpje_"+track+"_summary.csv",
    #                        # "z3":"z3_"+track+"_summary.csv",
    #                        # "ostrich":"ostrich_"+track+"_summary.csv",
    #                        # "cvc5":"cvc5_"+track+"_summary.csv"
    #                        }
    #
    #     summary_one_track(summary_folder,summary_file_dict,track)


    # summary one cross tracks
    # for track in benchmark_dict.keys():
    #     summary_file_dict = {}
    #     for solver_param in solver_param_list:
    #         k = solver_param[0]
    #         v = solver_param[1]
    #         v = [i.replace("--graph_type ", "") for i in v]
    #         parammeters_str = "_".join(v)
    #         summary_file_dict[k + ":" + parammeters_str] = k + "_" + parammeters_str + "_" + track + "_summary.csv"
    #     print(summary_file_dict)

        #summary_one_track(summary_folder, summary_file_dict, track)

    track="example_track"
    summary_file_dict={}
    for f in glob.glob(project_folder+"/src/process_benchmarks/summary/to_summary/*.csv"):
        f=f[f.rfind("/")+1:]
        solver=f[:f.find("_")]
        parameter_str=f[f.find("_")+1:f.find(track)-1]
        summary_file_dict[solver+":"+parameter_str]=solver+"_"+parameter_str+"_"+track+"_summary.csv"
    print(summary_file_dict)

    summary_one_track(summary_folder, summary_file_dict, track)







if __name__ == '__main__':
    main()