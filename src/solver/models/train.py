
import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models import GCN
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def train(dataset):


    # Create the model with given dimensions
    model = GCN(dataset.node_embedding_dim, 16, dataset.gclasses)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_examples = len(dataset)
    num_train = int(num_examples * 0.8)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=1, drop_last=False
    )
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=1, drop_last=False
    )

    for epoch in range(20):
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, batched_graph.ndata["feat"].float())
            loss = F.cross_entropy(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    num_correct = 0
    num_tests = 0
    for batched_graph, labels in test_dataloader:
        pred = model(batched_graph, batched_graph.ndata["feat"].float())
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)

    print("Test accuracy:", num_correct / num_tests)