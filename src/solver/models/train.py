
import os
import sys
sys.path.append("/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs")
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models import GCNWithNFFNN,GATWithNFFNN,GINWithNFFNN
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Dict
from collections import Counter
from src.solver.Constants import project_folder
from Dataset import WordEquationDataset

def main():
    configurations=[]
    for graph_type in ["graph_1","graph_2"]:
        for model_type in ["GCN","GAT","GIN"]:
            configurations.append({
                "graph_type": graph_type, "model_type": model_type, "num_epochs": 50, "learning_rate": 0.001,
                "save_criterion": "valid_accuracy", "batch_size": 20, "gnn_hidden_dim": 32,
                "gnn_layer_num": 2, "num_heads": 2, "ffnn_hidden_dim": 32, "ffnn_layer_num": 2
            })



    for config in configurations:
        train_one_model(config)


def train_one_model(parameters):
    print("-" * 10, parameters["graph_type"], "-" * 10)

    graph_folder = os.path.join(project_folder, "Woorpje_benchmarks", "example_train", parameters["graph_type"])
    train_valid_dataset = WordEquationDataset(graph_folder=graph_folder)
    train_valid_dataset.statistics()

    model = None
    if parameters["model_type"] == "GCN":
        model = GCNWithNFFNN(input_feature_dim=train_valid_dataset.node_embedding_dim,
                             gnn_hidden_dim=parameters["gnn_hidden_dim"],
                             gnn_layer_num=parameters["gnn_layer_num"],
                             ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                             ffnn_layer_num=parameters["ffnn_layer_num"])
    elif parameters["model_type"] == "GAT":
        model = GATWithNFFNN(input_feature_dim=train_valid_dataset.node_embedding_dim,
                             gnn_hidden_dim=parameters["gnn_hidden_dim"],
                             gnn_layer_num=parameters["gnn_layer_num"],
                             num_heads=parameters["num_heads"],
                             ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                             ffnn_layer_num=parameters["ffnn_layer_num"])
    elif parameters["model_type"] == "GIN":
        model = GINWithNFFNN(input_feature_dim=train_valid_dataset.node_embedding_dim,
                                gnn_hidden_dim=parameters["gnn_hidden_dim"],
                                gnn_layer_num=parameters["gnn_layer_num"],
                             ffnn_layer_num=parameters["ffnn_layer_num"],
                             ffnn_hidden_dim=parameters["ffnn_hidden_dim"])

    else:
        raise ValueError("Unsupported model type")

    save_path = os.path.join(project_folder, "models", f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")
    parameters["model_save_path"] = save_path

    trained_model = train(train_valid_dataset, GNN_model=model, parameters=parameters)


def train(dataset,GNN_model,parameters:Dict):
    train_dataloader, valid_dataloader  = create_data_loaders(dataset, parameters)

    # Create the model with given dimensions
    model = GNN_model
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    loss_function = nn.BCELoss()  # Initialize the loss function

    best_model = None
    best_valid_loss = float('inf')  # Initialize with a high value
    best_valid_accuracy = float('-inf')  # Initialize with a low value

    for epoch in range(parameters["num_epochs"]):
        # Training Phase
        model.train()
        train_loss = 0.0
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, batched_graph.ndata["feat"].float())

            # Convert labels to float for BCELoss
            labels = labels.float()

            loss = loss_function(pred.squeeze(), labels)
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

                # Convert labels to float for BCELoss
                labels_float = labels.float()

                loss = loss_function(pred.squeeze(), labels_float)
                valid_loss += loss.item()

                # Compute accuracy for binary classification
                predicted_labels = (pred > 0.5).float()
                num_correct += (predicted_labels.squeeze() == labels_float).sum().item()
                num_valids += len(labels)

        avg_valid_loss = valid_loss / len(valid_dataloader)
        valid_accuracy = num_correct / num_valids

        # Save based on specified criterion
        if parameters["save_criterion"] == "valid_loss" and avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            print(
                f"Epoch {epoch + 1:05d} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}",
                ", Save model for lowest validation loss")
            best_model = model
            torch.save(best_model, parameters["model_save_path"])

        elif parameters["save_criterion"] == "valid_accuracy" and valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            print(
                f"Epoch {epoch + 1:05d} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}",
                ", Save model for highest validation accuracy")
            best_model = model
            torch.save(best_model, parameters["model_save_path"])

        # Print the losses once every ten epochs
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch + 1:05d} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}")

    return best_model


def create_data_loaders(dataset, parameters):
    # Set seed for reproducibility for shuffling
    torch.manual_seed(42)

    num_examples = len(dataset)
    # Shuffle indices
    indices = torch.randperm(num_examples)

    # Reset randomness to ensure only the shuffling was deterministic
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(torch.initial_seed())  # Set seed to a new random value

    # Split the indices into 80% training and 20% validation
    num_train = int(num_examples * 0.8)
    train_indices = indices[:num_train]
    valid_indices = indices[num_train:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    # Count for training data
    train_labels = [int(dataset[i][1].item()) for i in train_indices]
    train_label_distribution = Counter(train_labels)

    # Count for validation data
    valid_labels = [int(dataset[i][1].item()) for i in valid_indices]
    valid_label_distribution = Counter(valid_labels)

    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=parameters["batch_size"], drop_last=False)
    valid_dataloader = GraphDataLoader(dataset, sampler=valid_sampler, batch_size=parameters["batch_size"], drop_last=False)

    print("Training label distribution:", train_label_distribution, "Base accuracy", max(train_label_distribution.values()) / sum(train_label_distribution.values()))
    print("Validation label distribution:", valid_label_distribution, "Base accuracy", max(valid_label_distribution.values()) / sum(valid_label_distribution.values()))

    return train_dataloader, valid_dataloader



if __name__ == '__main__':
    main()


