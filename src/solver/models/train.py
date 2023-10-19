
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


def train(dataset,model_save_path="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/models/model.pth"):
    num_examples = len(dataset)

    # Split the dataset into 80% training and 20% validation
    num_train = int(num_examples * 0.8)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    valid_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=1, drop_last=False
    )
    valid_dataloader = GraphDataLoader(
        dataset, sampler=valid_sampler, batch_size=1, drop_last=False
    )

    # Create the model with given dimensions
    model = GCN(dataset.node_embedding_dim, 16, dataset.gclasses)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_valid_loss = float('inf')  # Initialize with a high value

    for epoch in range(20):
        # Training Phase
        model.train()
        train_loss = 0.0
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, batched_graph.ndata["feat"].float())
            loss = F.cross_entropy(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation Phase
        model.eval()
        valid_loss = 0.0
        num_correct = 0
        num_valids = 0
        with torch.no_grad():
            for batched_graph, labels in valid_dataloader:
                pred = model(batched_graph, batched_graph.ndata["feat"].float())
                loss = F.cross_entropy(pred, labels)
                valid_loss += loss.item()
                num_correct += (pred.argmax(1) == labels).sum().item()
                num_valids += len(labels)

        avg_valid_loss = valid_loss / len(valid_dataloader)
        valid_accuracy = num_correct / num_valids

        # Check if the current validation loss is lower than the best known
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            print(
                f"Epoch {epoch + 1:05d} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}","save model")
            # Save the model with the best validation loss
            torch.save(model, model_save_path)

        # Print the losses once every five epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1:05d} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}")


    # Save the entire model
    torch.save(model, model_save_path)
    return model


