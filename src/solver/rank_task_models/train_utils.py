from src.solver.models.Models import Classifier, GNNRankTask1, GraphClassifier, GraphClassifierLightning


def get_gnn_and_classifier(parameters):
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

    return gnn_model, classifier_2


def initialize_model_lightning(parameters):
    gnn_model, classifier_2=get_gnn_and_classifier(parameters)
    model =  GraphClassifierLightning(gnn_model, classifier_2,model_parameters=parameters)

    return model
def initialize_model(parameters):
    gnn_model, classifier_2=get_gnn_and_classifier(parameters)
    # Initialize GraphClassifiers with the respective GNN models
    model = GraphClassifier(gnn_model, classifier_2)


    return model


