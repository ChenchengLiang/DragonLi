import csv
import os
from src.solver.independent_utils import strip_file_name_suffix
from src.process_benchmarks.utils import summary_one_track


def main():
    summary_folder = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/process_benchmarks/summary"


    for track in ["track_01","track_02","track_03"]:
        summary_file_dict={"this":"this_"+track+"_summary.csv",
                           "woorpje":"woorpje_"+track+"_summary.csv",
                           "z3":"z3_"+track+"_summary.csv",
                           "ostrich":"ostrich_"+track+"_summary.csv",
                           "cvc5":"cvc5_"+track+"_summary.csv"}

        summary_one_track(summary_folder,summary_file_dict,track)






if __name__ == '__main__':
    main()