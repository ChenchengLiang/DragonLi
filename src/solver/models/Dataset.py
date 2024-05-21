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


def get_one_dgl_graph(g):
    edges_src, edges_dst = get_edge_src_and_dst_list(g["edges"])
    num_nodes = torch.tensor(pd.DataFrame(g["nodes"]).to_numpy().shape[0])

    dgl_graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)

    dgl_graph.ndata["feat"] = torch.from_numpy(pd.DataFrame(g["node_types"]).to_numpy())
    # dgl_graph.ndata["label"] = node_labels #node label
    dgl_graph.edata["weight"] = torch.from_numpy(pd.DataFrame(g["edge_types"]).to_numpy())
    dgl_graph = dgl.add_self_loop(dgl_graph)


    return dgl_graph,g["label"]

def get_edge_src_and_dst_list(edges):
    edges_src = []
    edges_dst = []
    for e in edges:
        edges_src.append(e[0])
        edges_dst.append(e[1])
    return torch.from_numpy(pd.DataFrame(edges_src).to_numpy().flatten()), torch.from_numpy(pd.DataFrame(edges_dst).to_numpy().flatten())

class WordEquationDatasetBinaryClassification(DGLDataset):
    def __init__(self,graph_folder="",data_fold="train",node_type=3,graphs_from_memory=[],label_size=1):
        self._data_fold = data_fold
        self._graph_folder = graph_folder
        self._graphs_from_memory = graphs_from_memory
        self._node_type=node_type
        self._label_size=label_size
        super().__init__(name="WordEquation")

    def process(self):
        self.graphs = []
        self.labels = []

        graph_generator = self.get_graph_list_from_folder() if len(
            self._graphs_from_memory) == 0 else self._graphs_from_memory

        for g in graph_generator:
            dgl_graph,label=get_one_dgl_graph(g)
            self.graphs.append(dgl_graph)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)


    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def list_to_pandas(self, graphs):
        for g in graphs:
            for k in g:
                if isinstance(g[k], list):
                    g[k] = pd.DataFrame(g[k])

    def statistics(self):
        sat_label_number = 0
        unsat_label_number = 0
        unknown_label_number = 0
        for g in self.get_graph_list_from_folder():
            if g["label"] == 1:
                sat_label_number += 1
            elif g["label"] == 0:
                unsat_label_number += 1
            else:
                unknown_label_number += 1
        result_str = f"sat_label_number: {sat_label_number}, unsat_label_number: {unsat_label_number}, unknown_label_number: {unknown_label_number} \n"
        print(result_str)
        return result_str


    def get_graph_list_from_folder(self):
        '''
        graph_1 = {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2], "edges": [[1, 2], [2, 3], [3, 0]],
                   "edge_types": [1, 1, 1],
                   "label": 1}
        graph_2 = {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2], "edges": [[1, 2], [2, 3], [3, 0]],
                   "edge_types": [1, 1, 1],
                   "label": 1}
        graphs = [graph_1, graph_2]
        '''
        zip_file = self._graph_folder + ".zip"
        if os.path.exists(zip_file):  # read from zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_file_content:
                for graph_file in zip_file_content.namelist():
                    if fnmatch.fnmatch(graph_file, "*.graph.json"):
                        with zip_file_content.open(graph_file) as json_file:
                            loaded_dict = json.load(json_file)
                            loaded_dict["file_path"] = self._graph_folder + "/" + os.path.basename(graph_file)
                        if self._data_fold == "train":
                            if loaded_dict["label"] != -1:
                                yield loaded_dict
                        else:
                            yield loaded_dict
        elif os.path.exists(self._graph_folder):
            graph_file_list = glob.glob(self._graph_folder + "/*.graph.json")
            for graph_file in graph_file_list:
                with open(graph_file, 'r') as f:
                    loaded_dict = json.load(f)
                    loaded_dict["file_path"] = graph_file
                if self._data_fold == "train":
                    if loaded_dict["label"] != -1:
                        yield loaded_dict
                else:
                    yield loaded_dict
        else:
            print(f"folde not existed: {self._graph_folder}")



class WordEquationDatasetMultiClassification(DGLDataset):
    def __init__(self, graph_folder="", data_fold="train", node_type=3, graphs_from_memory:List[Dict]=[], label_size=3):
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

        for split_graphs in tqdm(graph_generator, desc="Processing graphs"):
            split_graph_list=[]
            split_graph_labels=[]
            for index,g in split_graphs.items():
                if isinstance(g,dict):
                    dgl_graph, label = get_one_dgl_graph(g)
                    split_graph_list.append(dgl_graph)
                    if label!=-1:
                        split_graph_labels.append(label)
            #print(len(split_graph_list)==self._label_size, len(split_graph_list),self._label_size)

            if len(split_graph_list)==self._label_size+1: # filter out the data with wrong label size for different classificaiton model
                self.graphs.append(split_graph_list)
                self.labels.append(split_graph_labels)

        # Convert the label list to tensor for saving.
        # if self._label_size == 2:
        #     self.labels = [1 if label == [1,0] else 0 for label in self.labels]
        #     self.labels = torch.LongTensor(self.labels)
        #
        # else:
        #     self.labels = torch.Tensor(self.labels)
        self.labels = torch.Tensor(self.labels)

        #print(self.labels)


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


class WordEquationDatasetMultiClassificationLazy(WordEquationDatasetMultiClassification):
    def __init__(self, graph_folder="", data_fold="train", node_type=3, graphs_from_memory=[], label_size=3):
        self._data_fold = data_fold
        self._graph_folder = graph_folder
        self._graphs_from_memory = graphs_from_memory
        self._label_size = label_size
        self._node_type = node_type
        super().__init__(graph_folder=graph_folder,data_fold=data_fold,node_type=node_type,graphs_from_memory=graphs_from_memory,label_size=label_size)

    def __getitem__(self, i):
        # Load and process the graph when it's accessed
        zip_file = self._graph_folder + ".zip"
        graph_name = self.graph_file_names[i]
        with zipfile.ZipFile(zip_file, 'r') as zip_file_content:
            split_graphs = self.load_graph(graph_name,zip_file_content)
        split_graph_list = []
        split_graph_labels = []
        for index, g in split_graphs.items():
            if isinstance(g, dict):
                dgl_graph, label = get_one_dgl_graph(g)
                split_graph_list.append(dgl_graph)
                split_graph_labels.append(label)

        # Convert the label list to tensor for saving.
        processed_label=None
        if self._label_size == 2:
            if split_graph_labels == [1, 0]:
                processed_label = torch.LongTensor([1]).squeeze()

            elif split_graph_labels == [0, 1]:
                processed_label = torch.LongTensor([0]).squeeze()
            else:
                color_print(text=f"label error: label={processed_label}", color="red")
                print(split_graphs)

        elif self._label_size == 3:
            processed_label = torch.Tensor(split_graph_labels)
        else:
            color_print(text=f"label size error: label_size={self._label_size}", color="red")


        return split_graph_list, processed_label

    def __len__(self):
        return len(self.graph_file_names)

    def load_graph(self, graph_file,zip_file_content):
        zip_file = self._graph_folder + ".zip"
        #with zipfile.ZipFile(zip_file, 'r') as zip_file_content:
        with zip_file_content.open(graph_file) as json_file:
            loaded_dict = json.load(json_file)
            loaded_dict["file_path"] = self._graph_folder + "/" + os.path.basename(graph_file)


        return loaded_dict

    def process(self):
        # Instead of loading all graphs, just store the file paths
        self.graph_file_names = []

        zip_file = self._graph_folder + ".zip"
        with zipfile.ZipFile(zip_file, 'r') as zip_file_content:
            for graph_file in tqdm(zip_file_content.namelist(), desc="Processing zip_file_content.namelist()"):
                if fnmatch.fnmatch(graph_file, "*.graph.json"):
                    split_graphs = self.load_graph(graph_file,zip_file_content)

                    if self._label_size+1==len(split_graphs): #self._label_size+1 because split_graphs has a additional file_path key
                        self.graph_file_names.append(graph_file)




class WordEquationDatasetMultiModels(WordEquationDatasetBinaryClassification):
    def __init__(self,graph_folder="",data_fold="train",node_type=3,graphs_from_memory=[],label_size=2):
        self._data_fold = data_fold
        self._graph_folder = graph_folder
        self._graphs_from_memory = graphs_from_memory
        self._label_size=label_size
        self._node_type=node_type
        super().__init__(graph_folder=graph_folder,data_fold=data_fold,node_type=node_type,graphs_from_memory=graphs_from_memory)

    def process(self):
        self.graphs = []
        self.labels = []
        self.node_embedding_dim = self._node_type
        graph_generator = self.get_graph_list_from_folder() if len(
            self._graphs_from_memory) == 0 else self._graphs_from_memory

        for split_graphs in graph_generator:
            split_graph_list=[]
            split_graph_labels=[]
            for index,g in split_graphs.items():
                if isinstance(g,dict):
                    dgl_graph, label = get_one_dgl_graph(g)
                    split_graph_list.append(dgl_graph)
                    split_graph_labels.append(label)

            self.graphs.append(split_graph_list)
            self.labels.append(split_graph_labels)

        # Convert the label list to tensor for saving.
        self.labels = torch.Tensor(self.labels)




    def get_graph_list_from_folder(self):
        zip_file=self._graph_folder+".zip"
        if os.path.exists(zip_file): #read from zip file
            with zipfile.ZipFile(zip_file,'r') as zip_file_content:
                for graph_file in zip_file_content.namelist():
                    if fnmatch.fnmatch(graph_file,"*.graph.json"):
                        with zip_file_content.open(graph_file) as json_file:
                            loaded_dict = json.load(json_file)
                            loaded_dict["file_path"] = self._graph_folder+"/"+os.path.basename(graph_file)
                        if self._data_fold == "train":
                            if len(loaded_dict) == self._label_size + 1:
                                yield loaded_dict
                        else:
                            yield loaded_dict
        elif os.path.exists(self._graph_folder):
            graph_file_list = glob.glob(self._graph_folder + "/*.graph.json")
            for graph_file in graph_file_list:
                with open(graph_file, 'r') as f:
                    loaded_dict = json.load(f)
                    loaded_dict["file_path"] = graph_file
                if self._data_fold == "train":
                    if len(loaded_dict) == self._label_size + 1:
                        yield loaded_dict
                else:
                    yield loaded_dict
        else:
            print(f"folde not existed: {self._graph_folder}")



    def statistics(self):
        sat_label_number = 0
        unsat_label_number = 0
        unknown_label_number = 0
        split_number = 0
        multi_classification_label_list=[]
        for graphs in self.get_graph_list_from_folder():
            split_number += 1
            multi_classification_label:List=[]
            for index, g in graphs.items():
                if isinstance(g, dict):
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


        result_str = f"label size: {self._label_size}, split_number: {split_number}, sat_label_number: {sat_label_number}, unsat_label_number: {unsat_label_number}, unknown_label_number: {unknown_label_number} \n"
        result_str+=f"labe distribution: {category_count.__str__()} \n"
        result_str+= f"dominate accuracy: {max(category_count.values())/sum(category_count.values())}"
        print(result_str)
        return result_str
