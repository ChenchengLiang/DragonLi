from src.solver.models.Models import Classifier, GNNRankTask1, GraphClassifier, GraphClassifierLightning, \
    GNNRankTask1BatchProcess, GNNRankTask0, GNNRankTask2, GNNRankTask0HashTable, ClassifierMultiFilter
from pytorch_lightning.callbacks import EarlyStopping, Callback

from src.solver.rank_task_models.Dataset import DGLDataModuleRank0, DGLDataModuleRank1, DGLDataModuleRank2


def get_dm(parameters):
    if parameters["rank_task"] == 0:
        dm = DGLDataModuleRank0(parameters, parameters["batch_size"], num_workers=4)
    elif parameters["rank_task"] == 1:
        dm = DGLDataModuleRank1(parameters, parameters["batch_size"], num_workers=4)
    elif parameters["rank_task"] == 2:
        dm = DGLDataModuleRank2(parameters, parameters["batch_size"], num_workers=4)
    else:
        raise ValueError("rank_task should be 0,1,2")
    return dm


def get_gnn_and_classifier(parameters):
    # Decide on the GNN type based on parameters
    embedding_type = "GCN" if parameters["model_type"] == "GCNSplit" else "GIN"
    if parameters["model_type"] not in ["GCNSplit", "GINSplit"]:
        raise ValueError("Unsupported model type")

    if parameters["rank_task"] == 0:
        # gnn_model = GNNRankTask0(
        #     input_feature_dim=parameters["node_type"],
        #     gnn_hidden_dim=parameters["gnn_hidden_dim"],
        #     gnn_layer_num=parameters["gnn_layer_num"],
        #     gnn_dropout_rate=parameters["gnn_dropout_rate"],
        #     embedding_type=embedding_type
        # )
        gnn_model = GNNRankTask0HashTable(
            input_feature_dim=parameters["node_type"],
            gnn_hidden_dim=parameters["gnn_hidden_dim"],
            gnn_layer_num=parameters["gnn_layer_num"],
            gnn_dropout_rate=parameters["gnn_dropout_rate"],
            embedding_type=embedding_type
        )

        # classifier = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
        #                           ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=2,
        #                           ffnn_dropout_rate=parameters["ffnn_dropout_rate"],
        #                           first_layer_ffnn_hidden_dim_factor=1)
        classifier = ClassifierMultiFilter(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                                           ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=2,
                                           ffnn_dropout_rate=parameters["ffnn_dropout_rate"],
                                           first_layer_ffnn_hidden_dim_factor=1,
                                           num_filters=parameters["classifier_num_filter"],
                                           pool_type=parameters["classifier_pool_type"])

    elif parameters["rank_task"] == 1:
        gnn_model = GNNRankTask1BatchProcess(
            input_feature_dim=parameters["node_type"],
            gnn_hidden_dim=parameters["gnn_hidden_dim"],
            gnn_layer_num=parameters["gnn_layer_num"],
            gnn_dropout_rate=parameters["gnn_dropout_rate"],
            embedding_type=embedding_type
        )
        # classifier = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
        #                         ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=2,
        #                         ffnn_dropout_rate=parameters["ffnn_dropout_rate"], first_layer_ffnn_hidden_dim_factor=2)
        classifier = ClassifierMultiFilter(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                                           ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=2,
                                           ffnn_dropout_rate=parameters["ffnn_dropout_rate"],
                                           first_layer_ffnn_hidden_dim_factor=2,
                                           num_filters=parameters["classifier_num_filter"],
                                           pool_type=parameters["classifier_pool_type"])
    elif parameters["rank_task"] == 2:
        gnn_model = GNNRankTask2(
            input_feature_dim=parameters["node_type"],
            gnn_hidden_dim=parameters["gnn_hidden_dim"],
            gnn_layer_num=parameters["gnn_layer_num"],
            gnn_dropout_rate=parameters["gnn_dropout_rate"],
            embedding_type=embedding_type,
            pooling_type=parameters["pooling_type"]
        )
        if parameters["pooling_type"] == "mean":
            first_layer_ffnn_hidden_dim_factor = 1
        elif parameters["pooling_type"] == "concat":
            first_layer_ffnn_hidden_dim_factor = parameters["label_size"]
        else:
            raise ValueError("Unsupported pooling type")
        # classifier = Classifier(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
        #                         ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=parameters["label_size"],
        #                         ffnn_dropout_rate=parameters["ffnn_dropout_rate"],
        #                         first_layer_ffnn_hidden_dim_factor=first_layer_ffnn_hidden_dim_factor)

        classifier = ClassifierMultiFilter(ffnn_hidden_dim=parameters["ffnn_hidden_dim"],
                                           ffnn_layer_num=parameters["ffnn_layer_num"], output_dim=parameters["label_size"],
                                           ffnn_dropout_rate=parameters["ffnn_dropout_rate"],
                                           first_layer_ffnn_hidden_dim_factor=first_layer_ffnn_hidden_dim_factor,
                                           num_filters=parameters["classifier_num_filter"],
                                           pool_type=parameters["classifier_pool_type"])
    else:
        raise ValueError("Unsupported rank task")

    return gnn_model, classifier


def initialize_model_lightning(parameters):
    gnn_model, classifier_2 = get_gnn_and_classifier(parameters)
    model = GraphClassifierLightning(gnn_model, classifier_2, model_parameters=parameters)

    return model


def initialize_model(parameters):
    gnn_model, classifier_2 = get_gnn_and_classifier(parameters)
    # Initialize GraphClassifiers with the respective GNN models
    model = GraphClassifier(gnn_model, classifier_2)

    return model


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("\n----- Starting to train -----\n")

    def on_train_end(self, trainer, pl_module):
        print("\n----- Training is done -----\n")
