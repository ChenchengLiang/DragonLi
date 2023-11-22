import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.solver.models.Models import GCNWithNFFNN, GATWithNFFNN, GINWithNFFNN, GCNWithGAPFFNN, MultiGNNs,GraphClassifier,SharedGNN,Classifier
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Dict, Callable
from collections import Counter
from src.solver.Constants import project_folder
from src.solver.independent_utils import get_memory_usage
from Dataset import WordEquationDataset,WordEquationDatasetMultiModels
import mlflow
import time
import random


def train_multiple_models(parameters, benchmark_folder):
    print("-" * 10, "train", "-" * 10)
    print("parameters:", parameters)
    # benchmark_folder = config['Path']['woorpje_benchmarks']

    print("load dataset")
    graph_folder = os.path.join(benchmark_folder, parameters["benchmark"], parameters["graph_type"])
    node_type=3
    dataset_2=WordEquationDatasetMultiModels(graph_folder=graph_folder,node_type=node_type,label_size=2)
    dataset_3=WordEquationDatasetMultiModels(graph_folder=graph_folder,node_type=node_type,label_size=3)
    # Shared GNN module
    shared_gnn = SharedGNN(input_feature_dim=node_type, gnn_hidden_dim=parameters["gnn_hidden_dim"], gnn_layer_num=parameters["gnn_layer_num"],gnn_dropout_rate=parameters["gnn_dropout_rate"])

    # Classifiers
    classifier_2 = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                              ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=2,
                              ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
    classifier_3 = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"], ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=3,ffnn_dropout_rate=parameters["ffnn_dropout_rate"])


    # GraphClassifiers
    model_2 = GraphClassifier(shared_gnn, classifier_2)
    model_3 = GraphClassifier(shared_gnn, classifier_3)


    loss_function = nn.CrossEntropyLoss()


def train_one_model(parameters, benchmark_folder):
    print("-" * 10, "train", "-" * 10)
    print("parameters:", parameters)
    # benchmark_folder = config['Path']['woorpje_benchmarks']

    print("load dataset")
    node_type=3
    graph_folder = os.path.join(benchmark_folder, parameters["benchmark"], parameters["graph_type"])
    train_valid_dataset = WordEquationDataset(graph_folder=graph_folder,node_type=node_type)
    dataset_statistics=train_valid_dataset.statistics()
    mlflow.log_text(dataset_statistics , artifact_file="dataset_statistics.txt")

    model = None
    if parameters["model_type"] == "GCN":
        model = GCNWithNFFNN(input_feature_dim=node_type,
                             gnn_hidden_dim=parameters["gnn_hidden_dim"],
                             gnn_layer_num=parameters["gnn_layer_num"], gnn_dropout_rate=parameters["gnn_dropout_rate"],
                             ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                             ffnn_layer_num=parameters["ffnn_layer_num"],
                             ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
        loss_function = nn.BCELoss()
    elif parameters["model_type"] == "GAT":
        model = GATWithNFFNN(input_feature_dim=node_type,
                             gnn_hidden_dim=parameters["gnn_hidden_dim"],
                             gnn_layer_num=parameters["gnn_layer_num"], gnn_dropout_rate=parameters["gnn_dropout_rate"],
                             num_heads=parameters["num_heads"],
                             ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                             ffnn_layer_num=parameters["ffnn_layer_num"],
                             ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
        loss_function = nn.BCELoss()
    elif parameters["model_type"] == "GIN":
        model = GINWithNFFNN(input_feature_dim=node_type,
                             gnn_hidden_dim=parameters["gnn_hidden_dim"],
                             gnn_layer_num=parameters["gnn_layer_num"], gnn_dropout_rate=parameters["gnn_dropout_rate"],
                             ffnn_layer_num=parameters["ffnn_layer_num"],
                             ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                             ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
        loss_function = nn.BCELoss()
    elif parameters["model_type"] == "GCNwithGAP":
        model = GCNWithGAPFFNN(input_feature_dim=node_type,
                               gnn_hidden_dim=parameters["gnn_hidden_dim"],
                               gnn_layer_num=parameters["gnn_layer_num"],
                               gnn_dropout_rate=parameters["gnn_dropout_rate"],
                               ffnn_layer_num=parameters["ffnn_layer_num"],
                               ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                               ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
        loss_function = nn.BCELoss()
    elif parameters["model_type"] == "MultiGNNs":
        model = MultiGNNs(input_feature_dim=node_type,
                          gnn_hidden_dim=parameters["gnn_hidden_dim"],
                          gnn_layer_num=parameters["gnn_layer_num"], gnn_dropout_rate=parameters["gnn_dropout_rate"],
                          ffnn_layer_num=parameters["ffnn_layer_num"],
                          ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                          ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
        loss_function = nn.BCELoss()

    else:
        raise ValueError("Unsupported model type")

    save_path = os.path.join(project_folder, "Models",
                             f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")
    parameters["model_save_path"] = save_path

    best_model, metrics = train_binary_classification(train_valid_dataset, GNN_model=model, parameters=parameters, loss_function=loss_function)

    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(best_model, "model")


def train_multi_classification():
    pass



def train_binary_classification(dataset, GNN_model, parameters: Dict, loss_function:Callable):
    train_dataloader, valid_dataloader = create_data_loaders(dataset, parameters)

    # Create the model with given dimensions
    model = GNN_model
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    #loss_function = nn.BCELoss()  # Initialize the loss function

    best_model = None
    best_valid_loss = float('inf')  # Initialize with a high value
    best_valid_accuracy = float('-inf')  # Initialize with a low value

    epoch_info_log = ""

    for epoch in range(parameters["num_epochs"]):
        #time.sleep(10)
        # Training Phase
        model.train()
        train_loss = 0.0
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph)

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
                pred = model(batched_graph)

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
            best_model,epoch_info_log=add_log_and_save_model(parameters, epoch, model, avg_train_loss, avg_valid_loss, valid_accuracy,
                                   epoch_info_log)

        elif parameters["save_criterion"] == "valid_accuracy" and valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_model,epoch_info_log=add_log_and_save_model(parameters, epoch, model, avg_train_loss, avg_valid_loss, valid_accuracy,
                                   epoch_info_log)

        # Print the losses once every ten epochs
        if epoch % 20 == 0:
            current_epoch_info = f"Epoch {epoch + 1:05d} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}"
            print(current_epoch_info)
            epoch_info_log = epoch_info_log + "\n" + current_epoch_info
            mlflow.log_text(epoch_info_log, artifact_file="model_log.txt")
        metrics = {"train_loss": avg_train_loss, "valid_loss": avg_valid_loss,
                   "best_valid_accuracy": best_valid_accuracy, "valid_accuracy": valid_accuracy, "epoch": epoch}
        mlflow.log_metrics(metrics, step=epoch)

    # Return the trained model and the best metrics
    best_metrics = {"best_valid_loss": best_valid_loss, "best_valid_accuracy": best_valid_accuracy}
    return best_model, best_metrics

def add_log_and_save_model(parameters,epoch,model,avg_train_loss,avg_valid_loss,valid_accuracy,epoch_info_log):
    current_epoch_info = f"Epoch {epoch + 1:05d} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}, Save model for highest validation accuracy"
    print(current_epoch_info)
    best_model = model
    best_model_path = parameters["model_save_path"].replace(".pth", "_" + parameters["run_id"] + ".pth")
    torch.save(best_model, best_model_path)
    mlflow.log_artifact(best_model_path)
    os.remove(best_model_path)
    epoch_info_log = epoch_info_log + "\n" + current_epoch_info
    mlflow.log_text(epoch_info_log, artifact_file="model_log.txt")
    return best_model,epoch_info_log


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


    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=parameters["batch_size"],
                                       drop_last=False)
    valid_dataloader = GraphDataLoader(dataset, sampler=valid_sampler, batch_size=parameters["batch_size"],
                                       drop_last=False)


    for batched_graph, labels in train_dataloader:
        print("debug")
        print(batched_graph)

    if parameters["dataset_task"]=="one_graph":
        # Count for training data
        train_labels = [int(dataset[i][1].item()) for i in train_indices]
        train_label_distribution = Counter(train_labels)

        # Count for validation data
        valid_labels = [int(dataset[i][1].item()) for i in valid_indices]
        valid_label_distribution = Counter(valid_labels)

        train_distribution_str = "Training label distribution: " + str(
            train_label_distribution) + "\nBase accuracy: " + str(
            max(train_label_distribution.values()) / sum(train_label_distribution.values()))
        valid_distribution_str = "Validation label distribution: " + str(
            valid_label_distribution) + "\nBase accuracy: " + str(
            max(valid_label_distribution.values()) / sum(valid_label_distribution.values()))
    else:
        train_distribution_str="multi_graphs"
        valid_distribution_str="multi_graphs"

    print(train_distribution_str)
    print(valid_distribution_str)

    mlflow.log_text(train_distribution_str + "\n" + valid_distribution_str, artifact_file="data_distribution.txt")

    return train_dataloader, valid_dataloader



def random_dataloaders(loader1, loader2):
    iterators = [iter(loader1), iter(loader2)]
    loaders_exhausted = [False, False]

    while not all(loaders_exhausted):
        loader_index = random.choice([i for i, exhausted in enumerate(loaders_exhausted) if not exhausted])

        try:
            yield next(iterators[loader_index]), loader_index + 1
        except StopIteration:
            loaders_exhausted[loader_index] = True
