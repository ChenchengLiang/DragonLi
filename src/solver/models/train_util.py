import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.solver.models.Models import GCNWithNFFNN, GATWithNFFNN, GINWithNFFNN, GCNWithGAPFFNN, MultiGNNs, \
    GraphClassifier, SharedGNN, Classifier
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Dict, Callable,Union
from collections import Counter
from src.solver.Constants import project_folder, mlflow_folder, checkpoint_folder
from src.solver.independent_utils import get_memory_usage, load_from_pickle, save_to_pickle, \
    load_from_pickle_within_zip, compress_to_zip, time_it,color_print
from Dataset import WordEquationDatasetBinaryClassification, WordEquationDatasetMultiModels, \
    WordEquationDatasetMultiClassification, WordEquationDatasetMultiClassificationLazy
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
    node_type = parameters["node_type"]
    dataset_2 = WordEquationDatasetMultiModels(graph_folder=graph_folder, node_type=node_type, label_size=2)
    dataset_3 = WordEquationDatasetMultiModels(graph_folder=graph_folder, node_type=node_type, label_size=3)

    dataset_statistics = dataset_2.statistics()
    mlflow.log_text(dataset_statistics, artifact_file="dataset_2_statistics.txt")
    dataset_statistics = dataset_3.statistics()
    mlflow.log_text(dataset_statistics, artifact_file="dataset_3_statistics.txt")

    # Shared GNN module
    shared_gnn = SharedGNN(input_feature_dim=node_type, gnn_hidden_dim=parameters["gnn_hidden_dim"],
                           gnn_layer_num=parameters["gnn_layer_num"], gnn_dropout_rate=parameters["gnn_dropout_rate"])

    # Classifiers
    classifier_2 = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                              ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=2,
                              ffnn_dropout_rate=parameters["ffnn_dropout_rate"])
    classifier_3 = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                              ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=3,
                              ffnn_dropout_rate=parameters["ffnn_dropout_rate"])

    # GraphClassifiers
    model_2 = GraphClassifier(shared_gnn, classifier_2)
    model_3 = GraphClassifier(shared_gnn, classifier_3)

    parameters["model_save_path"] = os.path.join(project_folder, "Models",
                                                 f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")

    best_models, metrics = train_binary_and_multi_classification(dataset_list=[dataset_2, dataset_3],
                                                                 GNN_model_list=[model_2, model_3],
                                                                 parameters=parameters)

    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(best_models[0], "model_2")
    mlflow.pytorch.log_model(best_models[1], "model_3")


def train_one_model(parameters, benchmark_folder):
    print("-" * 10, "train", "-" * 10)
    print("parameters:", parameters)
    # benchmark_folder = config['Path']['woorpje_benchmarks']

    print("load dataset")
    node_type = parameters["node_type"]
    graph_folder = os.path.join(benchmark_folder, parameters["benchmark"], parameters["graph_type"])
    train_valid_dataset = WordEquationDatasetBinaryClassification(graph_folder=graph_folder, node_type=node_type)
    dataset_statistics = train_valid_dataset.statistics()
    mlflow.log_text(dataset_statistics, artifact_file="dataset_statistics.txt")

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
        model_index = epoch % 2

        # Training Phase
        for model in GNN_model_list:
            model.train()
        train_dataloaders = [train_dataloader_1, train_dataloader_2]
        train_data_loader = train_dataloaders[model_index]

        train_loss = 0.0
        num_train = 0

        for batched_graph, labels in train_data_loader:
            model = GNN_model_list[model_index]
            optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])

            pred = model(batched_graph)

            loss = loss_function(pred.squeeze(), labels)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_train += 1
        avg_train_loss = train_loss / num_train

        # Validation Phase
        for model in GNN_model_list:
            model.eval()
        valid_dataloaders = [valid_dataloader_1, valid_dataloader_2]
        valid_data_loader = valid_dataloaders[model_index]
        valid_loss = 0.0
        num_correct = 0
        num_valids = 0

        for batched_graph, labels in valid_data_loader:
            model = GNN_model_list[model_index]

            with torch.no_grad():
                pred = model(batched_graph)
                predicted_labels = pred.squeeze()
                valid_loss += loss_function(predicted_labels, labels)

                # Compute accuracy
                _, predicted_labels = torch.max(predicted_labels, 1)
                _, labels = torch.max(labels, 1)

                num_correct += (predicted_labels == labels).sum().item()
                num_valids += len(labels)

        avg_valid_loss = valid_loss / num_valids
        valid_accuracy = num_correct / num_valids

        # Check and save the best model based on validation loss or accuracy
        if parameters["save_criterion"] == "valid_loss" and avg_valid_loss < best_valid_losses[model_index]:
            best_valid_losses[model_index] = avg_valid_loss
            best_model, epoch_info_log = add_log_and_save_model(parameters, epoch, model, avg_train_loss,
                                                                avg_valid_loss, valid_accuracy, epoch_info_log,
                                                                model_index=model_index + 2)
            best_models[model_index] = best_model  # Deep copy if needed


        elif parameters["save_criterion"] == "valid_accuracy" and valid_accuracy > best_valid_accuracies[model_index]:
            best_valid_accuracies[model_index] = valid_accuracy
            best_model, epoch_info_log = add_log_and_save_model(parameters, epoch, model, avg_train_loss,
                                                                avg_valid_loss, valid_accuracy, epoch_info_log,
                                                                model_index=model_index + 2)
            best_models[model_index] = best_model  # Deep copy if needed>

        # Print the losses once every ten epochs
        if epoch % 20 == 0:
            current_epoch_info = f"Epoch {epoch + 1:05d} | Model {model_index + 2} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}"
            print(current_epoch_info)
            epoch_info_log = epoch_info_log + "\n" + current_epoch_info
            mlflow.log_text(epoch_info_log, artifact_file="model_log.txt")
        metrics = {"train_loss": avg_train_loss, "valid_loss": avg_valid_loss,
                   "best_valid_accuracy_model_2": best_valid_accuracies[0],
                   "best_valid_accuracy_model_3": best_valid_accuracies[1], "valid_accuracy": valid_accuracy,
                   "epoch": epoch}
        mlflow.log_metrics(metrics, step=epoch)

    # Return the trained model and the best metrics
    best_metrics = {"best_valid_loss_model_2": best_valid_losses[0], "best_valid_loss_model_3": best_valid_losses[1],
                    "best_valid_accuracy_model_2": best_valid_accuracies[0],
                    "best_valid_accuracy_model_3": best_valid_accuracies[1]}

    return best_models, best_metrics


def initialize_model_structure(parameters):
    # todo expand GNN categories
    if parameters["model_type"] == "GCNSplit":
        shared_gnn = SharedGNN(input_feature_dim=parameters["node_type"], gnn_hidden_dim=parameters["gnn_hidden_dim"],
                               gnn_layer_num=parameters["gnn_layer_num"],
                               gnn_dropout_rate=parameters["gnn_dropout_rate"], embedding_type="GCN")
    elif parameters["model_type"] == "GINSplit":
        shared_gnn = SharedGNN(input_feature_dim=parameters["node_type"], gnn_hidden_dim=parameters["gnn_hidden_dim"],
                               gnn_layer_num=parameters["gnn_layer_num"],
                               gnn_dropout_rate=parameters["gnn_dropout_rate"], embedding_type="GIN")
    else:
        raise ValueError("Unsupported model type")
    # # Shared GNN module
    # shared_gnn = SharedGNN(input_feature_dim=node_type, gnn_hidden_dim=parameters["gnn_hidden_dim"],
    #                        gnn_layer_num=parameters["gnn_layer_num"], gnn_dropout_rate=parameters["gnn_dropout_rate"])

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

    return model_2, model_3
    #return shared_gnn, classifier_2, classifier_3, model_2, model_3


def train_multiple_models_separately(parameters, benchmark_folder):
    print("parameters:", parameters)
    # benchmark_folder = config['Path']['woorpje_benchmarks']

    model_2, model_3 = initialize_model_structure(parameters)

    parameters["model_save_path"] = os.path.join(project_folder, "Models",
                                                 f"model_{parameters['graph_type']}_{parameters['model_type']}.pth")
    dataset_2 = load_train_and_valid_dataset(parameters,benchmark_folder, 2)
    best_model_2, metrics_2 = train_binary_classification(dataset_2, model=model_2, parameters=parameters)
    dataset_3 = load_train_and_valid_dataset(parameters,benchmark_folder,  3)
    best_model_3, metrics_3 = train_multi_classification(dataset_3, model=model_3, parameters=parameters)

    metrics = {**metrics_2, **metrics_3}
    mlflow.log_metrics(metrics)
    # mlflow.pytorch.log_model(best_model_2, "model_2")
    # mlflow.pytorch.log_model(best_model_3, "model_3")

    print("-" * 10, "train finished", "-" * 10)


def load_train_and_valid_dataset(parameters, benchmark_folder, label_size):
    if os.path.exists(os.path.join(benchmark_folder,parameters["benchmark"], "valid_data")):
        color_print(text=f"load from two folders", color="yellow")
        train_data_folder = parameters["current_train_folder"]
        train_data=load_one_dataset(parameters, benchmark_folder, label_size,train_data_folder)
        valid_data_folder = os.path.join(parameters["benchmark"],"valid_data")
        valid_data=load_one_dataset(parameters, benchmark_folder, label_size, valid_data_folder)
        return {"train":train_data,"valid":valid_data}
    else:
        color_print(text=f"load from one folder", color="yellow")
        data_folder = parameters["current_train_folder"]
        return load_one_dataset(parameters, benchmark_folder, label_size, data_folder)

@time_it
def load_one_dataset(parameters, benchmark_folder,label_size,data_folder):
    graph_folder = os.path.join(benchmark_folder, data_folder, parameters["graph_type"])
    bench_folder = os.path.join(benchmark_folder, data_folder)
    node_type = parameters["node_type"]
    graph_type = parameters["graph_type"]

    start_time = time.time()
    # Filenames for the ZIP files
    zip_file = os.path.join(bench_folder, f"dataset_{label_size}_{graph_type}.pkl.zip")

    if os.path.exists(zip_file):
        print("-" * 10, "load dataset from zipped pickle:", data_folder, "-" * 10)
        # Names of the pickle files inside ZIP archives
        pickle_name = f"dataset_{label_size}_{graph_type}.pkl"
        # Load the datasets directly from ZIP files
        dataset = load_from_pickle_within_zip(zip_file, pickle_name)

    else:
        print("-" * 10, "load dataset from zipped file:", data_folder, "-" * 10)
        dataset = WordEquationDatasetMultiClassification(graph_folder=graph_folder, node_type=node_type,
                                                         label_size=label_size)
        # dataset = WordEquationDatasetMultiClassificationLazy(graph_folder=graph_folder, node_type=node_type,label_size=label_size)
        # pickle_file_2 = os.path.join(bench_folder, f"dataset_2_{graph_type}.pkl")
        # save_to_pickle(dataset_2, pickle_file_2)
        # compress_to_zip(pickle_file_2)

    parameters["label_size"] = dataset._label_size
    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print("-" * 10, "load dataset finished", "use time (s):", str(elapsed_time), "-" * 10)

    dataset_statistics = dataset.statistics()
    mlflow.log_text(dataset_statistics, artifact_file=f"{data_folder}_dataset_{label_size}_statistics.txt")
    return dataset

def training_phase(model,train_dataloader,loss_function,optimizer):
    # Training Phase
    model.train()
    train_loss = 0.0
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph)  # Squeeze to remove the extra dimension
        pred_final, labels = squeeze_labels(pred, labels)
        loss = loss_function(pred_final, labels)  # labels are not squeezed
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)
    return model,avg_train_loss

def validation_phase(model,valid_dataloader,loss_function,model_type):
    model.eval()
    valid_loss = 0.0
    num_correct = 0
    num_valids = 0
    with torch.no_grad():
        for batched_graph, labels in valid_dataloader:
            pred = model(batched_graph)
            pred_final, labels = squeeze_labels(pred, labels)

            loss = loss_function(pred_final, labels)
            valid_loss += loss.item()

            # Compute accuracy for binary classification
            num_correct, num_valids = compute_num_correct(pred_final, num_correct, num_valids, labels,
                                                          model_type=model_type)

    avg_valid_loss = valid_loss / len(valid_dataloader)
    valid_accuracy = num_correct / num_valids
    return model,avg_valid_loss,valid_accuracy




def train_binary_classification(dataset, model, parameters: Dict):
    print("-" * 10, "train_binary_classification", "-" * 10)

    model_type="binary_classification"
    train_dataloader, valid_dataloader, optimizer, loss_function, best_model, best_valid_loss, best_valid_accuracy, epoch_info_log, check_point_model_path = initialize_train_objects(
        dataset, parameters, model,model_type=model_type)

    model, optimizer, start_epoch, best_valid_loss, best_valid_accuracy = load_checkpoint(model, optimizer, parameters,
                                                                                          filename=check_point_model_path)

    for index, epoch in enumerate(range(start_epoch, parameters["num_epochs"]+1)):
        # Training Phase
        model, avg_train_loss = training_phase(model, train_dataloader, loss_function, optimizer)

        # Validation Phase
        model,avg_valid_loss,valid_accuracy=validation_phase(model, valid_dataloader, loss_function, model_type)

        # Save based on specified criterion
        best_model, best_valid_loss, best_valid_accuracy, epoch_info_log = log_and_save_best_model(parameters, epoch,
                                                                                                   best_model, model,
                                                                                                   "binary", 2,
                                                                                                   avg_train_loss,
                                                                                                   avg_valid_loss,
                                                                                                   valid_accuracy,
                                                                                                   best_valid_loss,
                                                                                                   best_valid_accuracy,
                                                                                                   epoch_info_log,index)
        if index == parameters["train_step"]-1 or epoch >= parameters["num_epochs"]:
            save_checkpoint(model, optimizer, epoch, best_valid_loss, best_valid_accuracy, parameters,
                            filename=check_point_model_path)
            break

    # Return the trained model and the best metrics
    best_metrics = {"best_valid_loss_binary": best_valid_loss, "best_valid_accuracy_binary": best_valid_accuracy}
    return best_model, best_metrics

def train_multi_classification(dataset, model, parameters: Dict):
    print("-" * 10, "train_multi_classification", "-" * 10)
    model_type="multi_classification"
    train_dataloader, valid_dataloader, optimizer, loss_function, best_model, best_valid_loss, best_valid_accuracy, epoch_info_log, check_point_model_path = initialize_train_objects(
        dataset, parameters, model,model_type=model_type)

    model, optimizer, start_epoch, best_valid_loss, best_valid_accuracy = load_checkpoint(model, optimizer, parameters,
                                                                                          filename=check_point_model_path)

    for index, epoch in enumerate(range(start_epoch, parameters["num_epochs"]+1)):
        # Training Phase
        model,avg_train_loss=training_phase(model,train_dataloader,loss_function,optimizer)

        # Validation Phase
        model, avg_valid_loss, valid_accuracy = validation_phase(model, valid_dataloader, loss_function,model_type)

        # Save based on specified criterion
        best_model, best_valid_loss, best_valid_accuracy, epoch_info_log = log_and_save_best_model(parameters, epoch,
                                                                                                   best_model, model,
                                                                                                   "multi_class",
                                                                                                   parameters["label_size"],
                                                                                                   avg_train_loss,
                                                                                                   avg_valid_loss,
                                                                                                   valid_accuracy,
                                                                                                   best_valid_loss,
                                                                                                   best_valid_accuracy,
                                                                                                   epoch_info_log,index)
        if index == parameters["train_step"]-1 or epoch >= parameters["num_epochs"]:
            save_checkpoint(model, optimizer, epoch, best_valid_loss, best_valid_accuracy, parameters,
                            filename=check_point_model_path)
            break

    # Return the trained model and the best metrics
    best_metrics = {"best_valid_loss_multi_class": best_valid_loss,
                    "best_valid_accuracy_multi_class": best_valid_accuracy}
    return best_model, best_metrics

def initialize_train_objects(dataset, parameters, model, model_type="binary_classification"):
    train_dataloader, valid_dataloader = create_data_loaders(dataset, parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    if model_type == "binary_classification":
        loss_function = nn.BCELoss()  # Initialize the loss function
        check_point_model_path = parameters["run_id"] + "_checkpoint_model_2.pth"
    elif model_type == "multi_classification":
        loss_function = nn.CrossEntropyLoss()
        check_point_model_path = parameters["run_id"] + "_checkpoint_model_3.pth"

    best_model = None
    best_valid_loss = float('inf')  # Initialize with a high value
    best_valid_accuracy = float('-inf')  # Initialize with a low value

    epoch_info_log = ""

    return train_dataloader, valid_dataloader, optimizer, loss_function, best_model, best_valid_loss, best_valid_accuracy, epoch_info_log, check_point_model_path

def compute_num_correct(pred_final,num_correct,num_valids,labels,model_type="binary_classification"):

    if model_type=="binary_classification":
        predicted_labels = (pred_final > 0.5).float()
        num_correct += (predicted_labels == labels).sum().item()
        num_valids += len(labels)
    elif model_type=="multi_classification":
        predicted_labels = torch.argmax(pred_final, dim=1)
        true_labels = torch.argmax(labels, dim=1)
        num_correct += (predicted_labels == true_labels).sum().item()
        num_valids += len(predicted_labels)
    return num_correct,num_valids


def save_checkpoint(model, optimizer, epoch, best_valid_loss, best_valid_accuracy, parameters,
                    filename='model_checkpoint.pth'):
    if epoch >= parameters["num_epochs"]:
        epoch=0

    checkpoint = {
        'epoch': epoch + 1,  # next epoch
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_valid_loss': best_valid_loss,
        'best_valid_accuracy': best_valid_accuracy,
        # Add other metrics or variables necessary for resuming training
    }
    run_id = parameters["run_id"]
    checkpoint_path = f"{checkpoint_folder}/{run_id}_{filename}"
    torch.save(checkpoint, checkpoint_path)
    mlflow.log_artifact(checkpoint_path)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, parameters, filename='model_checkpoint.pth'):
    try:
        run_id = parameters["run_id"]
        checkpoint_path = f"{checkpoint_folder}/{run_id}_{filename}"
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_valid_loss = checkpoint['best_valid_loss']
        best_valid_accuracy = checkpoint['best_valid_accuracy']
        print(
            f"Resuming from epoch {start_epoch} with best validation loss {best_valid_loss}, best_valid_accuracy {best_valid_accuracy}, current_train_folder {parameters['current_train_folder']}")
        return model, optimizer, start_epoch, best_valid_loss, best_valid_accuracy
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
        return model, optimizer, 0, float('inf'), float('-inf')  # Assuming you want to start with high loss


def squeeze_labels(pred, labels):
    # Convert labels to float for BCELoss
    labels = labels.float()
    pred_squeezed = torch.squeeze(pred)
    if len(labels) == 1:
        pred_final = torch.unsqueeze(pred_squeezed, 0)
    else:
        pred_final = pred_squeezed
    return pred_final, labels


def log_and_save_best_model(parameters, epoch, best_model, model, model_type, label_size, avg_train_loss,
                            avg_valid_loss, valid_accuracy, best_valid_loss, best_valid_accuracy, epoch_info_log,index):
    if parameters["save_criterion"] == "valid_loss" and avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        best_model, epoch_info_log = add_log_and_save_model(parameters, epoch, model, avg_train_loss, avg_valid_loss,
                                                            valid_accuracy,
                                                            epoch_info_log, model_index=label_size)

    elif parameters["save_criterion"] == "valid_accuracy" and valid_accuracy > best_valid_accuracy:
        best_valid_accuracy = valid_accuracy
        best_model, epoch_info_log = add_log_and_save_model(parameters, epoch, model, avg_train_loss, avg_valid_loss,
                                                            valid_accuracy,
                                                            epoch_info_log, model_index=label_size)

    # Print the losses once every 5 epochs
    if epoch % 2 == 0:
        current_epoch_info = f"Model: {model_type} | Epoch: {epoch:05d} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}"
        print(current_epoch_info)
        epoch_info_log = epoch_info_log + "\n" + current_epoch_info
        mlflow.log_text(epoch_info_log, artifact_file=f"model_log_{label_size}.txt")
    if "divided" in parameters["current_train_folder"]:
        current_folder_number=int(os.path.basename(parameters["current_train_folder"]).split("_")[1])
    else:
        current_folder_number=0

    if "train_data_folder_epoch_map" in parameters:
        total_epoch = sum(parameters["train_data_folder_epoch_map"].values())+index
    else:
        total_epoch = epoch

    metrics = {f"train_loss_{model_type}": avg_train_loss, f"valid_loss_{model_type}": avg_valid_loss,
               f"best_valid_accuracy_{model_type}": best_valid_accuracy, f"valid_accuracy_{model_type}": valid_accuracy,
               "epoch": epoch,"current_folder":current_folder_number,"total_epoch":total_epoch}
    mlflow.log_metrics(metrics)
    return best_model, best_valid_loss, best_valid_accuracy, epoch_info_log


def add_log_and_save_model(parameters, epoch, model, avg_train_loss, avg_valid_loss, valid_accuracy, epoch_info_log,
                           model_index=0):
    current_epoch_info = f"Epoch {epoch :05d} | Model {model_index} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.4f}, Save model for highest validation accuracy"
    print(current_epoch_info)
    best_model = model

    #rounded_valid_accuracy = round(valid_accuracy, 4)
    best_model_path = parameters["model_save_path"].replace(".pth", "_" + parameters["run_id"] + ".pth").replace(
        "model_", f"model_{model_index}_")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    torch.save(best_model, best_model_path)

    mlflow.log_artifact(best_model_path)
    os.remove(best_model_path)
    epoch_info_log = epoch_info_log + "\n" + current_epoch_info
    mlflow.log_text(epoch_info_log, artifact_file=f"model_log_{model_index}.txt")
    return best_model, epoch_info_log


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
    return train_sampler, valid_sampler, train_indices, valid_indices


def create_data_loaders(dataset:Union[WordEquationDatasetMultiClassification,Dict], parameters: Dict):
    if isinstance(dataset,WordEquationDatasetMultiClassification):
        train_dataloader, valid_dataloader = data_loader_1(dataset, parameters)
    elif isinstance(dataset,Dict):
        train_dataloader, valid_dataloader = data_loader_2(dataset, parameters)

    return train_dataloader, valid_dataloader


@time_it
def data_loader_1(dataset, parameters):
    train_sampler, valid_sampler, train_indices, valid_indices = get_samplers(dataset)

    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=parameters["batch_size"],
                                       drop_last=False)
    # valid_dataloader=train_dataloader
    valid_dataloader = GraphDataLoader(dataset, sampler=valid_sampler, batch_size=parameters["batch_size"],
                                       drop_last=False)

    # Check if the dataset is for binary classification or multi-class classification
    first_label = dataset[0][1]
    label_size = dataset._label_size
    print("print(dataset[0])")
    print(dataset[0])

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

    get_distribution_strings(label_size,train_labels,valid_labels)

    return train_dataloader, valid_dataloader


@time_it
def data_loader_2(dataset, parameters):
    train_dataloader = GraphDataLoader(dataset["train"], batch_size=parameters["batch_size"], drop_last=False)
    valid_dataloader = GraphDataLoader(dataset["valid"], batch_size=parameters["batch_size"], drop_last=False)

    first_label = dataset["train"][0][1]
    label_size =dataset['train']._label_size
    print("print(dataset[0])")
    print(dataset["train"][0])
    is_binary_classification = len(first_label.shape) == 0 or (
            len(first_label.shape) == 1 and first_label.shape[0] == 1)

    # Process labels based on the classification type
    if is_binary_classification:
        # Binary classification
        train_labels=[int(row[1].item()) for row in dataset["train"]]
        valid_labels=[int(row[1].item()) for row in dataset["valid"]]

    else:
        # Multi-class classification
        train_labels=one_hot_to_class_indices([row[1].numpy() for row in dataset["train"]])
        valid_labels=one_hot_to_class_indices([row[1].numpy() for row in dataset["valid"]])

    get_distribution_strings(label_size,train_labels,valid_labels)

    return train_dataloader, valid_dataloader

@time_it
def get_distribution_strings(label_size,train_labels,valid_labels):
    # Calculate label distributions
    train_label_distribution = Counter(train_labels)
    valid_label_distribution = Counter(valid_labels)

    train_distribution_str = "Training label distribution: " + str(
        train_label_distribution) + "\nBase accuracy: " + str(
        max(train_label_distribution.values()) / sum(train_label_distribution.values()))
    valid_distribution_str = "Validation label distribution: " + str(
        valid_label_distribution) + "\nBase accuracy: " + str(
        max(valid_label_distribution.values()) / sum(valid_label_distribution.values()))
    print("-" * 10)
    print(train_distribution_str)
    print(valid_distribution_str)

    mlflow.log_text(train_distribution_str + "\n" + valid_distribution_str,
                    artifact_file=f"data_distribution_{label_size}.txt")


def one_hot_to_class_indices(one_hot_labels):
    return [np.argmax(label_vector) for label_vector in one_hot_labels]


def check_run_exists(run_id):
    try:
        # Attempt to retrieve the run using the run_id
        run = mlflow.get_run(run_id)
        return True
    except Exception as e:
        # If an exception occurs, it likely means the run_id does not exist
        print(f"Run ID not found: {e}")
        return False

def check_experiment_exists(experiment_id):
    try:
        # Attempt to retrieve the experiment using the experiment_id
        experiment = mlflow.get_experiment(experiment_id)
        return True if experiment else False
    except Exception as e:
        # If an exception occurs, it likely means the experiment_id does not exist
        print(f"Experiment ID not found: {e}")
        return False