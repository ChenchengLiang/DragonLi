import fnmatch
import os
import sys
import zipfile

from src.solver.independent_utils import color_print

os.environ["DGLBACKEND"] = "pytorch"
import torch
import dgl
from dgl.data import DGLDataset
import pandas as pd
import glob
import json
from src.solver.Constants import SAT,UNKNOWN,UNSAT,RED,COLORRESET
from typing import Dict, List
from tqdm import tqdm
import time



def get_one_dgl_graph_concatenated_with_other_graphs():
    pass


class WordEquationDatasetMultiClassificationRankTask(DGLDataset):
    def __init__(self, graph_folder="", data_fold="train", node_type=3, graphs_from_memory:List[Dict]=[], label_size=2):
        self._data_fold = data_fold
        self._graph_folder = graph_folder
        self._graphs_from_memory = graphs_from_memory
        self._label_size = label_size
        self._node_type = node_type
        super().__init__(name="WordEquation")

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def process(self):
        self.graphs = []
        self.labels = []
        self.node_embedding_dim = self._node_type
        graph_generator = self.get_graph_list_from_folder() if len(
            self._graphs_from_memory) == 0 else self._graphs_from_memory

        for graphs_to_rank in tqdm(graph_generator, desc="Processing graphs"):
            graph_list_to_rank=[]
            rank_labels=[]
            for index,g in graphs_to_rank.items():
                if isinstance(g,dict):
                    dgl_graph, label = get_one_dgl_graph_concatenated_with_other_graphs(g) #todo design this function
                    graph_list_to_rank.append(dgl_graph)
                    rank_labels.append(label)


            self.graphs.append(graph_list_to_rank)
            self.labels.append(rank_labels)


        self.labels = torch.Tensor(self.labels)


    def get_graph_list_from_folder(self):
        '''
                graph_1 = {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2], "edges": [[1, 2], [2, 3], [3, 0]],
                           "edge_types": [1, 1, 1],
                           "label": -1,"satisfiability": UNKNOWN}
                graph_1 = {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2], "edges": [[1, 2], [2, 3], [3, 0]],
                           "edge_types": [1, 1, 1],
                           "label": 1,"satisfiability": SAT}
                graph_2 = {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2], "edges": [[1, 2], [2, 3], [3, 0]],
                           "edge_types": [1, 1, 1],
                           "label": 0,"satisfiability": UNSAT}
                graphs = [graph_0,graph_1, graph_2]
                '''
        zip_file=self._graph_folder+".zip"
        if os.path.exists(zip_file): #read from zip file
            with zipfile.ZipFile(zip_file,'r') as zip_file_content:
                for graph_file in zip_file_content.namelist():
                    if fnmatch.fnmatch(graph_file,"*.graph.json"):
                        with zip_file_content.open(graph_file) as json_file:
                            loaded_dict = json.load(json_file)
                            loaded_dict["file_path"] = self._graph_folder+"/"+os.path.basename(graph_file)
                        yield loaded_dict
        elif os.path.exists(self._graph_folder):
            graph_file_list = glob.glob(self._graph_folder + "/*.graph.json")
            for graph_file in graph_file_list:
                with open(graph_file, 'r') as f:
                    loaded_dict = json.load(f)
                    loaded_dict["file_path"] = graph_file
                yield loaded_dict
        else:
            print(f"folder not existed: {self._graph_folder}")


    def statistics(self):

        start_time = time.time()
        print("-- get statistics --")
        sat_label_number = 0
        unsat_label_number = 0
        unknown_label_number = 0
        split_number = 0
        max_node_number=0
        graph_number=0
        total_graph_node=0
        min_node_number=sys.maxsize
        multi_classification_label_list=[]
        for graphs in self.get_graph_list_from_folder():
            split_number += 1
            multi_classification_label:List=[]
            for index, g in graphs.items():
                if isinstance(g, dict):
                    total_graph_node+=len(g["nodes"])
                    graph_number+=1
                    if max_node_number<len(g["nodes"]):
                        max_node_number=len(g["nodes"])
                    if min_node_number>len(g["nodes"]):
                        min_node_number=len(g["nodes"])
                    if g["label"]!=-1:
                        multi_classification_label.append(g["label"])
                    if g["satisfiability"] == SAT:
                        sat_label_number += 1
                    elif g["satisfiability"] == UNSAT:
                        unsat_label_number += 1
                    else:
                        unknown_label_number += 1
            multi_classification_label_list.append(multi_classification_label)


        # Initialize a counter for each category
        category_count = {0: 0, 1: 0, 2: 0}

        # Count each category
        for label in multi_classification_label_list:
            category = label.index(1) # return 1's index
            category_count[category] += 1


        result_str = (f"label size: {self._label_size}, split_number: {split_number}, \n "
                      f"sat_label_number: {sat_label_number}, unsat_label_number: {unsat_label_number}, unknown_label_number: {unknown_label_number} \n")
        result_str+=f"labe distribution: {category_count.__str__()} \n"
        result_str+= f"dominate accuracy: {max(category_count.values())/sum(category_count.values())} \n"
        result_str+= f"max node number: {max_node_number} \n"
        result_str+= f"min node number: {min_node_number} \n"
        result_str+= f"average node number: {total_graph_node/graph_number}"
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"-- get statistics finished, used time (s) {elapsed_time} --")
        print(result_str)
        return result_str