
import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models import GCNWithNFFNN,GATWithNFFNN
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Dict

from Dataset import WordEquationDataset

def main():


    graph_folder="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/03_track"
    train_valid_dataset = WordEquationDataset(graph_folder=graph_folder)
    train_valid_dataset.statistics()
    graph, label = train_valid_dataset[0]
    print("train_valid_dataset[0]",graph, label)



    save_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/models/model.pth"
    parameters ={"model_save_path":save_path,"num_epochs":200,"learning_rate":0.001,"batch_size":10,"gnn_hidden_dim":64,
                 "gnn_layer_num":2,"num_heads":2,"ffnn_hidden_dim":64,"ffnn_layer_num":2}


    GCN_model = GCNWithNFFNN(input_feature_dim=train_valid_dataset.node_embedding_dim, gnn_hidden_dim=parameters["gnn_hidden_dim"],
                             gnn_layer_num=parameters["gnn_layer_num"], ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                             ffnn_layer_num=parameters["ffnn_layer_num"])
    GAT_model = GATWithNFFNN(input_feature_dim=train_valid_dataset.node_embedding_dim, gnn_hidden_dim=parameters["gnn_hidden_dim"],
                             gnn_layer_num=parameters["gnn_layer_num"], num_heads=parameters["num_heads"],
                             ffnn_hidden_dim=parameters["ffnn_hidden_dim"], ffnn_layer_num=parameters["ffnn_layer_num"])


    trained_model=train(train_valid_dataset,GNN_model=GAT_model,parameters=parameters)



def train(dataset,GNN_model,parameters:Dict):
    num_examples = len(dataset)

    # Split the dataset into 80% training and 20% validation
    num_train = int(num_examples * 0.8)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    valid_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=parameters["batch_size"], drop_last=False)
    valid_dataloader = GraphDataLoader(dataset, sampler=valid_sampler, batch_size=parameters["batch_size"], drop_last=False)

    # Create the model with given dimensions
    model = GNN_model
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    loss_function = nn.BCELoss() # Initialize the loss function
    best_valid_loss = float('inf')  # Initialize with a high value
    best_model=None


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

        # Check if the current validation loss is lower than the best known
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            print(
                f"Epoch {epoch + 1:05d} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}",
                ", Save model for lowest validation loss")
            # Save the model with the best validation loss
            best_model = model
            #torch.save(best_model.state_dict(), parameters["model_save_path"])
            torch.save(best_model,parameters["model_save_path"])

        # Print the losses once every five epochs
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1:05d} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}")

    return best_model


if __name__ == '__main__':
    main()


