import csv
import os
from src.solver.independent_utils import strip_file_name_suffix
from src.process_benchmarks.utils import summary_one_track


def main():
    summary_folder = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/process_benchmarks/summary"

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

    #summary one cross tracks
    for track in ["track_01","track_02","track_03","g_track_01_eval"]:
        summary_file_dict={"this:fixed": "this_" + track + "_summary_fixed.csv",
                           "this:random":"this_"+track+"_summary_random.csv",
                           #"this:random_1": "this_" + track + "_summary_random_1.csv",
                           "this:graph_1": "this_" + track + "_summary_graph_1.csv",
                           "this:graph_2": "this_" + track + "_summary_graph_2.csv",
                           }

        summary_one_track(summary_folder,summary_file_dict,track)





if __name__ == '__main__':
    main()