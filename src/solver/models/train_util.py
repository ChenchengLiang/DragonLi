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
from Dataset import WordEquationDatasetBinaryClassification,WordEquationDatasetMultiModels,WordEquationDatasetMultiClassification
import mlflow
import time
import random
import numpy as np


def train_multiple_models(parameters, benchmark_folder):
    print("-" * 10, "train", "-" * 10)
    print("parameters:", parameters)
    # benchmark_folder = config['Path']['woorpje_benchmarks']

    print("load dataset")
    graph_folder = os.path.join(benchmark_folder, parameters["benchmark"], parameters["graph_type"])
    node_type=3
    dataset_2=WordEquationDatasetMultiModels(graph_folder=graph_folder,node_type=node_type,label_size=2)
    dataset_3=WordEquationDatasetMultiModels(graph_folder=graph_folder,node_type=node_type,label_size=3)


    dataset_statistics = dataset_2.statistics()
    mlflow.log_text(dataset_statistics, artifact_file="dataset_2_statistics.txt")
    dataset_statistics = dataset_3.statistics()
    mlflow.log_text(dataset_statistics, artifact_file="dataset_3_statistics.txt")


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


    parameters["model_save_path"] = os.path.join(project_folder, "Models",
                             f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")




    best_models, metrics = train_binary_and_multi_classification(dataset_list=[dataset_2, dataset_3], GNN_model_list=[model_2, model_3], parameters=parameters)

    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(best_models[0], "model_2")
    mlflow.pytorch.log_model(best_models[1], "model_3")


def train_one_model(parameters, benchmark_folder):
    print("-" * 10, "train", "-" * 10)
    print("parameters:", parameters)
    # benchmark_folder = config['Path']['woorpje_benchmarks']

    print("load dataset")
    node_type=3
    graph_folder = os.path.join(benchmark_folder, parameters["benchmark"], parameters["graph_type"])
    train_valid_dataset = WordEquationDatasetBinaryClassification(graph_folder=graph_folder, node_type=node_type)
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
    elif parameters["model_type"] == "GAT":
        model = GATWithNFFNN(input_feature_dim=node_type,
                             gnn_hidden_dim=parameters["gnn_hidden_dim"],
                             gnn_layer_num=parameters["gnn_layer_num"], gnn_dropout_rate=parameters["gnn_dropout_rate"],
                             num_heads=parameters["num_heads"],
                             ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                             ffnn_layer_num=parameters["ffnn_layer_num"],
                             ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
    elif parameters["model_type"] == "GIN":
        model = GINWithNFFNN(input_feature_dim=node_type,
                             gnn_hidden_dim=parameters["gnn_hidden_dim"],
                             gnn_layer_num=parameters["gnn_layer_num"], gnn_dropout_rate=parameters["gnn_dropout_rate"],
                             ffnn_layer_num=parameters["ffnn_layer_num"],
                             ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                             ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
    elif parameters["model_type"] == "GCNwithGAP":
        model = GCNWithGAPFFNN(input_feature_dim=node_type,
                               gnn_hidden_dim=parameters["gnn_hidden_dim"],
                               gnn_layer_num=parameters["gnn_layer_num"],
                               gnn_dropout_rate=parameters["gnn_dropout_rate"],
                               ffnn_layer_num=parameters["ffnn_layer_num"],
                               ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                               ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
    elif parameters["model_type"] == "MultiGNNs":
        model = MultiGNNs(input_feature_dim=node_type,
                          gnn_hidden_dim=parameters["gnn_hidden_dim"],
                          gnn_layer_num=parameters["gnn_layer_num"], gnn_dropout_rate=parameters["gnn_dropout_rate"],
                          ffnn_layer_num=parameters["ffnn_layer_num"],
                          ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                          ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
    else:
        raise ValueError("Unsupported model type")

    save_path = os.path.join(project_folder, "Models",
                             f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")
    parameters["model_save_path"] = save_path

    best_model, metrics = train_binary_classification(train_valid_dataset, model=model, parameters=parameters)

    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(best_model, "model")


def train_binary_and_multi_classification(dataset_list, GNN_model_list, parameters):
    train_sampler_1, valid_sampler_1, _, _ = get_samplers(dataset_list[0])
    train_sampler_2, valid_sampler_2, _, _ = get_samplers(dataset_list[1])

    train_dataloader_1 = GraphDataLoader(dataset_list[0], sampler=train_sampler_1, batch_size=parameters["batch_size"],
                                         drop_last=False)
    valid_dataloader_1 = GraphDataLoader(dataset_list[0], sampler=valid_sampler_1, batch_size=parameters["batch_size"],
                                         drop_last=False)

    train_dataloader_2 = GraphDataLoader(dataset_list[1], sampler=train_sampler_2, batch_size=parameters["batch_size"],
                                         drop_last=False)
    valid_dataloader_2 = GraphDataLoader(dataset_list[1], sampler=valid_sampler_2, batch_size=parameters["batch_size"],
                                         drop_last=False)

    best_models = [None] * len(GNN_model_list)
    best_valid_losses = [float('inf')] * len(GNN_model_list)
    best_valid_accuracies = [float('-inf')] * len(GNN_model_list)

    loss_function = nn.CrossEntropyLoss()
    epoch_info_log = ""
    for epoch in range(parameters["num_epochs"]):
        model_index=epoch%2

        # Training Phase
        for model in GNN_model_list:
            model.train()
        train_dataloaders = [train_dataloader_1, train_dataloader_2]
        train_data_loader=train_dataloaders[model_index]

        train_loss=0.0
        num_train=0

        for batched_graph, labels in train_data_loader:

            model = GNN_model_list[model_index]
            optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])

            pred = model(batched_graph)

            loss = loss_function(pred.squeeze(), labels)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_train+=1
        avg_train_loss = train_loss / num_train

        # Validation Phase
        for model in GNN_model_list:
            model.eval()
        valid_dataloaders = [valid_dataloader_1, valid_dataloader_2]
        valid_data_loader=valid_dataloaders[model_index]
        valid_loss = 0.0
        num_correct = 0
        num_valids = 0

        for batched_graph, labels in valid_data_loader:

            model = GNN_model_list[model_index]

            with torch.no_grad():
                pred = model(batched_graph)
                predicted_labels=pred.squeeze()
                valid_loss += loss_function(predicted_labels, labels)

                # Compute accuracy
                _,predicted_labels= torch.max(predicted_labels, 1)
                _,labels=torch.max(labels,1)

                num_correct += (predicted_labels == labels).sum().item()
                num_valids += len(labels)

        avg_valid_loss = valid_loss / num_valids
        valid_accuracy = num_correct / num_valids

        # Check and save the best model based on validation loss or accuracy
        if parameters["save_criterion"] == "valid_loss" and avg_valid_loss < best_valid_losses[model_index ]:
            best_valid_losses[model_index] = avg_valid_loss
            best_model, epoch_info_log = add_log_and_save_model(parameters, epoch, model, avg_train_loss,avg_valid_loss, valid_accuracy,epoch_info_log,model_index=model_index+2)
            best_models[model_index] = best_model  # Deep copy if needed


        elif parameters["save_criterion"] == "valid_accuracy" and valid_accuracy > best_valid_accuracies[model_index]:
            best_valid_accuracies[model_index] = valid_accuracy
            best_model, epoch_info_log = add_log_and_save_model(parameters, epoch, model, avg_train_loss, avg_valid_loss, valid_accuracy,epoch_info_log,model_index=model_index+2)
            best_models[model_index] = best_model  # Deep copy if needed>


        # Print the losses once every ten epochs
        if epoch % 20 == 0:
            current_epoch_info = f"Epoch {epoch + 1:05d} | Model {model_index+2} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}"
            print(current_epoch_info)
            epoch_info_log = epoch_info_log + "\n" + current_epoch_info
            mlflow.log_text(epoch_info_log, artifact_file="model_log.txt")
        metrics = {"train_loss": avg_train_loss, "valid_loss": avg_valid_loss,
                   "best_valid_accuracy_model_2": best_valid_accuracies[0],
                   "best_valid_accuracy_model_3": best_valid_accuracies[1], "valid_accuracy": valid_accuracy, "epoch": epoch}
        mlflow.log_metrics(metrics, step=epoch)

    # Return the trained model and the best metrics
    best_metrics = {"best_valid_loss_model_2": best_valid_losses[0],"best_valid_loss_model_3": best_valid_losses[1],
                    "best_valid_accuracy_model_2": best_valid_accuracies[0],"best_valid_accuracy_model_3": best_valid_accuracies[1]}

    return best_models, best_metrics


def train_multiple_models_separately(parameters, benchmark_folder):
    print("-" * 10, "train", "-" * 10)
    print("parameters:", parameters)
    # benchmark_folder = config['Path']['woorpje_benchmarks']

    print("load dataset")
    graph_folder = os.path.join(benchmark_folder, parameters["benchmark"], parameters["graph_type"])
    node_type = 3
    dataset_2 = WordEquationDatasetMultiClassification(graph_folder=graph_folder, node_type=node_type,label_size=2)
    dataset_3 = WordEquationDatasetMultiClassification(graph_folder=graph_folder, node_type=node_type, label_size=3)

    dataset_statistics = dataset_2.statistics()
    mlflow.log_text(dataset_statistics, artifact_file="dataset_2_statistics.txt")
    dataset_statistics = dataset_3.statistics()
    mlflow.log_text(dataset_statistics, artifact_file="dataset_3_statistics.txt")

    # Shared GNN module
    shared_gnn = SharedGNN(input_feature_dim=node_type, gnn_hidden_dim=parameters["gnn_hidden_dim"],
                           gnn_layer_num=parameters["gnn_layer_num"], gnn_dropout_rate=parameters["gnn_dropout_rate"])

    # Classifiers
    classifier_2 = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                              ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=1,
                              ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
    classifier_3 = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                              ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=3,
                              ffnn_dropout_rate=parameters["ffnn_dropout_rate"])

    # GraphClassifiers
    model_2 = GraphClassifier(shared_gnn, classifier_2)
    model_3 = GraphClassifier(shared_gnn, classifier_3)

    parameters["model_save_path"] = os.path.join(project_folder, "Models",
                                                 f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")

    best_model_2, metrics_2 = train_binary_classification(dataset_2, model=model_2, parameters=parameters)
    best_model_3, metrics_3 = train_multi_classification(dataset_3, model=model_3, parameters=parameters)

    metrics = {**metrics_2, **metrics_3}
    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(best_model_2, "model_2")
    mlflow.pytorch.log_model(best_model_3, "model_3")


def train_multi_classification(dataset, model, parameters: Dict):
    train_dataloader, valid_dataloader = create_data_loaders(dataset, parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])

    best_model = None
    best_valid_loss = float('inf')
    best_valid_accuracy = float('-inf')

    loss_function = nn.CrossEntropyLoss()  # Change to CrossEntropyLoss for multi-class
    epoch_info_log = ""

    for epoch in range(parameters["num_epochs"]):
        model.train()
        train_loss = 0.0
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph).squeeze()  # Squeeze to remove the extra dimension

            loss = loss_function(pred, labels)  # labels are not squeezed
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        valid_loss = 0.0
        num_correct = 0
        num_valids = 0
        with torch.no_grad():
            for batched_graph, labels in valid_dataloader:
                pred = model(batched_graph).squeeze()

                loss = loss_function(pred, labels)
                valid_loss += loss.item()

                # Accuracy calculation for multi-class
                predicted_labels = torch.argmax(pred, dim=1)
                true_labels = torch.argmax(labels, dim=1)
                num_correct += (predicted_labels == true_labels).sum().item()
                num_valids += len(predicted_labels)

        avg_valid_loss = valid_loss / len(valid_dataloader)
        valid_accuracy = num_correct / num_valids

        best_model,best_valid_loss, best_valid_accuracy = log_and_save_best_model(parameters, epoch, model, "binary", dataset._label_size,
                                                                       avg_train_loss, avg_valid_loss, valid_accuracy,
                                                                       best_valid_loss, best_valid_accuracy,
                                                                       epoch_info_log)
    # Return the trained model and the best metrics
    best_metrics = {"best_valid_loss_multi_class": best_valid_loss, "best_valid_accuracy_multi_class": best_valid_accuracy}
    return best_model, best_metrics





def train_binary_classification(dataset, model, parameters: Dict):
    train_dataloader, valid_dataloader = create_data_loaders(dataset, parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    #loss_function = nn.BCELoss()  # Initialize the loss function

    best_model = None
    best_valid_loss = float('inf')  # Initialize with a high value
    best_valid_accuracy = float('-inf')  # Initialize with a low value

    loss_function = nn.BCELoss()
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
        best_model,best_valid_loss,best_valid_accuracy=log_and_save_best_model(parameters, epoch, model,"binary",2, avg_train_loss, avg_valid_loss, valid_accuracy,
                                best_valid_loss, best_valid_accuracy, epoch_info_log)

    # Return the trained model and the best metrics
    best_metrics = {"best_valid_loss_binary": best_valid_loss, "best_valid_accuracy_binary": best_valid_accuracy}
    return best_model, best_metrics

def log_and_save_best_model(parameters, epoch, model ,model_type,label_size,avg_train_loss, avg_valid_loss, valid_accuracy,best_valid_loss,best_valid_accuracy, epoch_info_log):

    if parameters["save_criterion"] == "valid_loss" and avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        best_model, epoch_info_log = add_log_and_save_model(parameters, epoch, model, avg_train_loss, avg_valid_loss,
                                                            valid_accuracy,
                                                            epoch_info_log,model_index=label_size)

    elif parameters["save_criterion"] == "valid_accuracy" and valid_accuracy > best_valid_accuracy:
        best_valid_accuracy = valid_accuracy
        best_model, epoch_info_log = add_log_and_save_model(parameters, epoch, model, avg_train_loss, avg_valid_loss,
                                                            valid_accuracy,
                                                            epoch_info_log,model_index=label_size)

    # Print the losses once every ten epochs
    if epoch % 20 == 0:
        current_epoch_info = f"Model: {model_type} | Epoch: {epoch + 1:05d} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}"
        print(current_epoch_info)
        epoch_info_log = epoch_info_log + "\n" + current_epoch_info
        mlflow.log_text(epoch_info_log, artifact_file="model_log.txt")
    metrics = {f"train_loss_{model_type}": avg_train_loss, f"valid_loss_{model_type}": avg_valid_loss,
               f"best_valid_accuracy_{model_type}": best_valid_accuracy, f"valid_accuracy_{model_type}": valid_accuracy, "epoch": epoch}
    mlflow.log_metrics(metrics, step=epoch)
    return best_model,best_valid_loss,best_valid_accuracy

def add_log_and_save_model(parameters,epoch,model,avg_train_loss,avg_valid_loss,valid_accuracy,epoch_info_log,model_index=0):
    current_epoch_info = f"Epoch {epoch + 1:05d} | Model {model_index} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}, Save model for highest validation accuracy"
    print(current_epoch_info)
    best_model = model
    best_model_path = parameters["model_save_path"].replace(".pth", "_" + parameters["run_id"] + ".pth").replace("model_",f"model_{model_index}_")
    torch.save(best_model, best_model_path)
    mlflow.log_artifact(best_model_path)
    os.remove(best_model_path)
    epoch_info_log = epoch_info_log + "\n" + current_epoch_info
    mlflow.log_text(epoch_info_log, artifact_file="model_log.txt")
    return best_model,epoch_info_log

def get_samplers(dataset):
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
    return train_sampler,valid_sampler,train_indices,valid_indices

def create_data_loaders(dataset, parameters):
    train_sampler,valid_sampler,train_indices,valid_indices=get_samplers(dataset)

    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=parameters["batch_size"],
                                       drop_last=False)
    valid_dataloader = GraphDataLoader(dataset, sampler=valid_sampler, batch_size=parameters["batch_size"],
                                       drop_last=False)

    # Check if the dataset is for binary classification or multi-class classification
    first_label = dataset[0][1]
    is_binary_classification = len(first_label.shape) == 0 or (
                len(first_label.shape) == 1 and first_label.shape[0] == 1)

    # Process labels based on the classification type
    if is_binary_classification:
        # Binary classification
        train_labels = [int(dataset[i][1].item()) for i in train_indices]
        valid_labels = [int(dataset[i][1].item()) for i in valid_indices]
    else:
        # Multi-class classification
        train_labels = one_hot_to_class_indices([dataset[i][1].numpy() for i in train_indices])
        valid_labels = one_hot_to_class_indices([dataset[i][1].numpy() for i in valid_indices])

    # Calculate label distributions
    train_label_distribution = Counter(train_labels)
    valid_label_distribution = Counter(valid_labels)

    # Calculate distribution strings
    train_distribution_str = "Training label distribution: " + str(
        train_label_distribution) + "\nBase accuracy: " + str(
        max(train_label_distribution.values()) / sum(train_label_distribution.values()))
    valid_distribution_str = "Validation label distribution: " + str(
        valid_label_distribution) + "\nBase accuracy: " + str(
        max(valid_label_distribution.values()) / sum(valid_label_distribution.values()))
    print("-"*10)
    print(train_distribution_str)
    print(valid_distribution_str)

    mlflow.log_text(train_distribution_str + "\n" + valid_distribution_str, artifact_file="data_distribution.txt")

    return train_dataloader, valid_dataloader


def one_hot_to_class_indices(one_hot_labels):
    return [np.argmax(label_vector) for label_vector in one_hot_labels]
