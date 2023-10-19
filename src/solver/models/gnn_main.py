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
from Dataset import KarateClubDataset,SyntheticDataset,WordEquationDataset
from train import train
def main():


    # dataset = KarateClubDataset()
    # graph = dataset[0]
    # print(graph)

    # dataset = SyntheticDataset()
    # graph, label = dataset[0]
    # print(graph, label)

    graph_folder="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/test"
    dataset = WordEquationDataset(graph_folder=graph_folder)
    graph, label = dataset[0]
    print(graph, label)

    train(dataset)




if __name__ == '__main__':
    main()