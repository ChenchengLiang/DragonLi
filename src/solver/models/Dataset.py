import fnmatch
import os
import zipfile

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from dgl.data import DGLDataset
import pandas as pd
import glob
import json
from src.solver.Constants import SAT,UNKNOWN,UNSAT
from typing import Dict, List

def get_one_dgl_graph(g):
    edges_src, edges_dst = get_edge_src_and_dst_list(g["edges"])
    num_nodes = pd.DataFrame(g["nodes"]).to_numpy().shape[0]

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
    return pd.DataFrame(edges_src).to_numpy().flatten(), pd.DataFrame(edges_dst).to_numpy().flatten()

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
    def __init__(self, graph_folder="", data_fold="train", node_type=3, graphs_from_memory=[], label_size=3):
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

        for split_graphs in graph_generator:
            split_graph_list=[]
            split_graph_labels=[]
            for index,g in split_graphs.items():
                if isinstance(g,dict):
                    dgl_graph, label = get_one_dgl_graph(g)
                    split_graph_list.append(dgl_graph)
                    split_graph_labels.append(label)
            #print(len(split_graph_list)==self._label_size, len(split_graph_list),self._label_size)
            if len(split_graph_list)==self._label_size:
                self.graphs.append(split_graph_list)
                self.labels.append(split_graph_labels)

        # Convert the label list to tensor for saving.
        if self._label_size == 2:
            self.labels = [1 if label == [1,0] else 0 for label in self.labels]
            self.labels = torch.LongTensor(self.labels)
        else:
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
        sat_label_number = 0
        unsat_label_number = 0
        unknown_label_number = 0
        split_number = 0
        max_node_number=0
        multi_classification_label_list=[]
        for graphs in self.get_graph_list_from_folder():
            split_number += 1
            multi_classification_label:List=[]
            for index, g in graphs.items():
                if isinstance(g, dict):
                    if max_node_number<len(g["nodes"]):
                        max_node_number=len(g["nodes"])
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
        result_str+= f"dominate accuracy: {max(category_count.values())/sum(category_count.values())} \n"
        result_str+= f"max node number: {max_node_number}"
        print(result_str)
        return result_str


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

class SyntheticDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="synthetic")

    def process(self):
        edges = pd.read_csv("./graph_edges.csv")
        properties = pd.read_csv("./graph_properties.csv")
        self.graphs = []
        self.labels = []

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row["graph_id"]] = row["label"]
            num_nodes_dict[row["graph_id"]] = row["num_nodes"]

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby("graph_id")

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id["src"].to_numpy()
            dst = edges_of_id["dst"].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class KarateClubDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="karate_club")

    def process(self):
        nodes_data = pd.read_csv("./members.csv")
        edges_data = pd.read_csv("./interactions.csv")
        node_features = torch.from_numpy(nodes_data["Age"].to_numpy())
        node_labels = torch.from_numpy(
            nodes_data["Club"].astype("category").cat.codes.to_numpy()
        )
        edge_features = torch.from_numpy(edges_data["Weight"].to_numpy())
        edges_src = torch.from_numpy(edges_data["Src"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["Dst"].to_numpy())

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=nodes_data.shape[0]
        )
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels
        self.graph.edata["weight"] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train: n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
