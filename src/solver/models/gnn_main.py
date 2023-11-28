import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import urllib.request
import pandas as pd
from Dataset import KarateClubDataset,SyntheticDataset,WordEquationDatasetBinaryClassification
from src.solver.Constants import project_folder,bench_folder

def main():


    # dataset = KarateClubDataset()
    # graph = dataset[0]
    # print(graph)

    # dataset = SyntheticDataset()
    # graph, label = dataset[0]
    # print(graph, label)

    graph_folder=bench_folder+"/test"
    train_valid_dataset = WordEquationDatasetBinaryClassification(graph_folder=graph_folder)
    graph, label = train_valid_dataset[0]
    print("train_valid_dataset[0]",graph, label)




if __name__ == '__main__':
    main()