import os

import mlflow

from src.solver.Constants import bench_folder, RED
from src.solver.independent_utils import load_from_pickle_within_zip, color_print
from src.solver.models.Models import Classifier, GNNRankTask1, GraphClassifier

def initialize_model(parameters):
    classifier_2 = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                              ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=2,
                              ffnn_dropout_rate=parameters["ffnn_dropout_rate"], parent_node=False)

    # Decide on the GNN type based on parameters
    embedding_type = "GCN" if parameters["model_type"] == "GCNSplit" else "GIN"
    if parameters["model_type"] not in ["GCNSplit", "GINSplit"]:
        raise ValueError("Unsupported model type")

    gnn_model = GNNRankTask1(
        input_feature_dim=parameters["node_type"],
        gnn_hidden_dim=parameters["gnn_hidden_dim"],
        gnn_layer_num=parameters["gnn_layer_num"],
        gnn_dropout_rate=parameters["gnn_dropout_rate"],
        embedding_type=embedding_type
    )
    # Initialize GraphClassifiers with the respective GNN models
    model = GraphClassifier(gnn_model, classifier_2)
    return model


def read_dataset_from_zip(parameters,data_folder):

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
        dataset_statistics = dataset.statistics()
    else:
        color_print(f"Error: ZIP file not found: {zip_file}", RED)
        dataset = None
        dataset_statistics = ""

    mlflow.log_text(dataset_statistics, artifact_file=f"{data_folder}_dataset_statistics.txt")
    return dataset

