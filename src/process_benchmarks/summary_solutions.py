import csv
import os
from src.solver.independent_utils import strip_file_name_suffix
from src.process_benchmarks.utils import summary_one_track


def main():
    summary_folder = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/process_benchmarks/summary"

    solver_param_list = [["this", ["fixed"]],
                         ["this", ["random"]],
                         ["this", ["gnn", "--graph_type graph_1"]],
                         ["this", ["gnn", "--graph_type graph_2"]],
                         # ["woorpje",[]],
                         # ["z3",[]],
                         # ["ostrich",[]],
                         # ["cvc5",[]],
                         ]

    test_track = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/test"
    example_track = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/examples"
    track_01 = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track"
    g_track_01_sat = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track_generated/SAT"
    g_track_01_mixed = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track_generated/mixed"
    g_track_01_eval = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/01_track_generated_eval_data"
    track_02 = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/02_track"
    track_03 = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/03_track"
    track_04 = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/04_track"
    track_05 = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/05_track"

    benchmark_dict = {
        # "test_track":test_track,
        "example_track": example_track,
        # "track_01": track_01,
        # "g_track_01_sat":g_track_01_sat,
        # "g_track_01_mixed": g_track_01_mixed,
        # "g_track_01_eval":g_track_01_eval,
        # "track_02": track_02,
        # "track_03": track_03,
        # "track_04": track_04,
        # "track_05": track_05
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

    #summary one cross tracks
    # summary one cross tracks
    for track in benchmark_dict.keys():
        summary_file_dict = {}
        for solver_param in solver_param_list:
            k = solver_param[0]
            v = solver_param[1]
            v = [i.replace("--graph_type ", "") for i in v]
            parammeters_str = "_".join(v)
            summary_file_dict[k + ":" + parammeters_str] = k + "_" + parammeters_str + "_" + track + "_summary.csv"

        summary_one_track(summary_folder, summary_file_dict, track)





if __name__ == '__main__':
    main()