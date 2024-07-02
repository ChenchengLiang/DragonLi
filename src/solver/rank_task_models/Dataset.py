import fnmatch
import os
import sys
import zipfile

import mlflow

from src.solver.models.train_util import get_data_distribution, custom_collate_fn, custom_collate_fn_rank_task_2

os.environ["DGLBACKEND"] = "pytorch"
import torch
import dgl
from dgl.data import DGLDataset
import pandas as pd
import glob
import json
from src.solver.Constants import SAT, UNKNOWN, UNSAT, RED, COLORRESET, bench_folder, project_folder
from typing import Dict, List
from tqdm import tqdm
import time
from src.solver.models.Dataset import get_one_dgl_graph
from src.solver.independent_utils import time_it, load_from_pickle_within_zip, color_print, keep_first_one
from src.solver.independent_utils import color_print, initialize_one_hot_category_count
import pytorch_lightning as pl
from dgl.dataloading import GraphDataLoader


def get_one_dgl_graph_concatenated_with_other_graphs():
    pass




class DGLDataModule(pl.LightningDataModule):
    def __init__(self, parameters, batch_size, num_workers):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.parameters = parameters

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_ds = read_dataset_from_zip(self.parameters,
                                              os.path.basename(self.parameters["current_train_folder"]))
        self.val_ds = read_dataset_from_zip(self.parameters, "valid_data")
        dataset = {"train": self.train_ds, "valid": self.val_ds}

        data_distribution_str=get_data_distribution(dataset, self.parameters)
        self.parameters["data_distribution_str"]=data_distribution_str

    def train_dataloader(self):
        return GraphDataLoader(self.train_ds, batch_size=self.parameters["batch_size"], drop_last=False,
                               collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return GraphDataLoader(self.val_ds, batch_size=self.parameters["batch_size"], drop_last=False,
                               collate_fn=custom_collate_fn, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()


class DGLDataModuleRank0(DGLDataModule):
    def __init__(self, parameters, batch_size, num_workers):
        super().__init__(parameters, batch_size, num_workers)

    def train_dataloader(self):
        return GraphDataLoader(self.train_ds, batch_size=self.parameters["batch_size"], drop_last=False, shuffle=False)

    def val_dataloader(self):
        return GraphDataLoader(self.val_ds, batch_size=self.parameters["batch_size"], drop_last=False, shuffle=False)

class DGLDataModuleRank2(DGLDataModule):
    def __init__(self, parameters, batch_size, num_workers):
        super().__init__(parameters, batch_size, num_workers)

    def train_dataloader(self):
        return GraphDataLoader(self.train_ds, batch_size=self.parameters["batch_size"],
                               drop_last=False,shuffle=False,collate_fn=custom_collate_fn_rank_task_2)

    def val_dataloader(self):
        return GraphDataLoader(self.val_ds, batch_size=self.parameters["batch_size"],
                               drop_last=False, shuffle=False,collate_fn=custom_collate_fn_rank_task_2)


class WordEquationDatasetMultiClassificationRankTask(DGLDataset):
    def __init__(self, graph_folder="", graphs_from_memory: List[Dict] = [],
                 label_size=2):
        self._graph_folder = graph_folder
        self._graphs_from_memory = graphs_from_memory
        self._label_size = label_size
        super().__init__(name="WordEquation")

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def process(self):
        self.graphs = []
        self.labels = []

        graph_generator = self.get_graph_list_from_folder() if len(
            self._graphs_from_memory) == 0 else self._graphs_from_memory

        for graphs_to_rank in tqdm(graph_generator, desc="Processing graphs"):
            # read G and labels
            G_list = []
            label_list = []
            for index, g in graphs_to_rank.items():
                if isinstance(g, dict):
                    dgl_graph, label = get_one_dgl_graph(g)
                    G_list.append(dgl_graph)
                    label_list.append(label)

            # form each graph and label for training
            for index, g in enumerate(G_list):
                one_train_data = [g] + G_list
                rank_label = [1, 0] if label_list[index] == 1 else [0, 1]
                self.graphs.append(one_train_data)
                self.labels.append(rank_label)

        self.labels = torch.Tensor(self.labels)

    def get_graph_list_from_folder(self):
        '''
                graph_1 = {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2], "edges": [[1, 2], [2, 3], [3, 0]],
                           "edge_types": [1, 1, 1],
                           "label": 0,"satisfiability": UNKNOWN}
                graph_2 = {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2], "edges": [[1, 2], [2, 3], [3, 0]],
                           "edge_types": [1, 1, 1],
                           "label": 1,"satisfiability": SAT}
                graph_3 = {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2], "edges": [[1, 2], [2, 3], [3, 0]],
                           "edge_types": [1, 1, 1],
                           "label": 0,"satisfiability": UNSAT}
                graphs = [graph_0,graph_1, graph_2]
                '''
        zip_file = self._graph_folder + ".zip"
        if os.path.exists(zip_file):  # read from zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_file_content:
                for graph_file in zip_file_content.namelist():
                    if fnmatch.fnmatch(graph_file, "*.graph.json"):
                        with zip_file_content.open(graph_file) as json_file:
                            loaded_dict = json.load(json_file)
                            loaded_dict["file_path"] = self._graph_folder + "/" + os.path.basename(graph_file)
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

    @time_it
    def statistics(self):

        print("-- get statistics --")
        sat_label_number = 0
        unsat_label_number = 0
        unknown_label_number = 0
        split_number = 0
        max_node_number = 0
        graph_number = 0
        total_graph_node = 0
        min_node_number = sys.maxsize
        category_count = initialize_one_hot_category_count(self._label_size)
        for label in self.labels:
            category_count[tuple(label.numpy())] += 1

        for graphs in self.get_graph_list_from_folder():
            split_number += 1

            for index, g in graphs.items():
                if isinstance(g, dict):
                    total_graph_node += len(g["nodes"])
                    graph_number += 1

                    if max_node_number < len(g["nodes"]):
                        max_node_number = len(g["nodes"])
                    if min_node_number > len(g["nodes"]):
                        min_node_number = len(g["nodes"])
                    if g["satisfiability"] == SAT:
                        sat_label_number += 1
                    elif g["satisfiability"] == UNSAT:
                        unsat_label_number += 1
                    else:
                        unknown_label_number += 1

        result_str = (f"label size: {self._label_size}, split_number: {split_number}, \n"
                      f"sat_label_number: {sat_label_number}, unsat_label_number: {unsat_label_number}, unknown_label_number: {unknown_label_number} \n")
        result_str += f"label distribution: {category_count.__str__()} \n"
        result_str += f"dominate accuracy: {max(category_count.values()) / sum(category_count.values())} \n"
        result_str += f"max node number: {max_node_number} \n"
        result_str += f"min node number: {min_node_number} \n"
        result_str += f"average node number: {total_graph_node / graph_number}"

        print(result_str)
        return result_str


class WordEquationDatasetMultiClassificationRankTask0(WordEquationDatasetMultiClassificationRankTask):
    def __init__(self, graph_folder="", graphs_from_memory: List[Dict] = [],
                 label_size=2):
        super().__init__(graph_folder=graph_folder, graphs_from_memory=graphs_from_memory, label_size=label_size)

    def process(self):
        self.graphs = []
        self.labels = []

        graph_generator = self.get_graph_list_from_folder() if len(
            self._graphs_from_memory) == 0 else self._graphs_from_memory

        for graphs_to_rank in tqdm(graph_generator, desc="Processing graphs"): #graphs_to_rank represent a list of eq graphs
            for index, g in graphs_to_rank.items():
                if isinstance(g, dict):
                    dgl_graph, label = get_one_dgl_graph(g)
                    rank_label=[1,0] if label == 1 else [0,1]
                    self.graphs.append(dgl_graph)
                    self.labels.append(rank_label)

        self.labels = torch.Tensor(self.labels)



class WordEquationDatasetMultiClassificationRankTask2(WordEquationDatasetMultiClassificationRankTask):
    def __init__(self, graph_folder="", graphs_from_memory: List[Dict] = [],
                 label_size=2):
        super().__init__(graph_folder=graph_folder, graphs_from_memory=graphs_from_memory, label_size=label_size)
        self._label_size = label_size

    def process(self):
        self.graphs = []
        self.labels = []

        graph_generator = self.get_graph_list_from_folder() if len(
            self._graphs_from_memory) == 0 else self._graphs_from_memory
        empty_graph_dict = {"nodes": [0], "node_types": [5], "edges": [[0,0]],
                           "edge_types": [1],
                           "label": 0,"satisfiability": SAT}

        for graphs_to_rank in tqdm(graph_generator, desc="Processing graphs"): #graphs_to_rank represent a list of eq graphs
            one_data_graph_list = []
            one_data_label_list = []
            # prepare one data
            for index, g in graphs_to_rank.items():
                if isinstance(g, dict):
                    dgl_graph, label = get_one_dgl_graph(g)
                    one_data_graph_list.append(dgl_graph)
                    one_data_label_list.append(label)
            # pad to the same size
            while len(one_data_graph_list)<self._label_size:
                dgl_graph, label = get_one_dgl_graph(empty_graph_dict)
                one_data_graph_list.append(dgl_graph)
                one_data_label_list.append(label)
            #trim to the same size
            if len(one_data_graph_list)>self._label_size:
                one_data_graph_list=one_data_graph_list[:self._label_size]
                one_data_label_list=one_data_label_list[:self._label_size]

            #ensure sum(label)==1
            one_data_label_list=keep_first_one(one_data_label_list)
            if sum(one_data_label_list)!=1:
                color_print(f"graph number: {len(one_data_graph_list)}","yellow")
                color_print(f"label number: {len(one_data_label_list)}","yellow")
                color_print(f"sum labels: {sum(one_data_label_list)}","yellow")
                color_print(str(one_data_label_list),"yellow")

            # form one for training
            self.graphs.append(one_data_graph_list)
            self.labels.append(one_data_label_list)


        self.labels = torch.Tensor(self.labels)



def read_dataset_from_zip(parameters, data_folder, get_data_statistics=True):
    pickle_folder = os.path.join(bench_folder, parameters["benchmark_folder"], data_folder)
    graph_type = parameters["graph_type"]

    # Filenames for the ZIP files
    zip_file = os.path.join(pickle_folder, f"dataset_{graph_type}.pkl.zip")
    if os.path.exists(zip_file):
        print("-" * 10, "load dataset from zipped pickle:", data_folder, "-" * 10)
        # Names of the pickle files inside ZIP archives
        pickle_name = f"dataset_{graph_type}.pkl"
        # Load the datasets directly from ZIP files
        dataset = load_from_pickle_within_zip(zip_file, pickle_name)
        if get_data_statistics == True:
            dataset_statistics = dataset.statistics()
            # with open(f"{project_folder}/mlruns/{parameters['experiment_id']}/{parameters['run_id']}/artifacts/{data_folder}_dataset_statistics.txt", 'w') as file:
            #     file.write(dataset_statistics)
            parameters["dataset_statistics_str"]=dataset_statistics
            #mlflow.log_text(dataset_statistics, artifact_file=f"{data_folder}_dataset_statistics.txt")
        else:
            dataset_statistics = ""
    else:
        color_print(f"Error: ZIP file not found: {zip_file}", RED)
        dataset = None
        dataset_statistics = ""

    return dataset
