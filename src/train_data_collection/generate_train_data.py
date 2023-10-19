import os.path
from src.solver.independent_utils import strip_file_name_suffix,dump_to_json_with_format
from src.solver.Parser import Parser,EqParser
import json
import glob
def main():
    file_list = glob.glob("/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/03_track/woorpje/*.eq")

    for file_path in file_list:
        output_one_eq_graph(file_path,visualize=False)





def output_one_eq_graph(file_path,visualize=False):

    parser_type = EqParser()
    parser = Parser(parser_type)
    parsed_content = parser.parse(file_path)
    #print("parsed_content:", parsed_content)

    answer_file = strip_file_name_suffix(file_path) + ".answer"
    with open(answer_file, 'r') as file:
        answer = file.read()

    for eq in parsed_content["equation_list"]:
        if visualize==True:
            # visualize
            eq.visualize_graph(file_path)
        # get gnn format
        nodes, edges = eq.get_graph_1()
        satisfiability = answer
        graph_dict = eq.graph_to_gnn_format(nodes, edges, satisfiability)
        #print(graph_dict)
        # Dumping the dictionary to a JSON file
        json_file=strip_file_name_suffix(file_path) + ".json"
        dump_to_json_with_format(graph_dict, json_file)





if __name__ == '__main__':
    main()