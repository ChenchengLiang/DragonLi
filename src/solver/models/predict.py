import torch
from src.solver.Constants import int_label_to_satisfiability
from Dataset import WordEquationDataset
from dgl.dataloading import GraphDataLoader
def main():
    # Load the evaluation dataset
    graph_folder = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/eval"
    evaluation_dataset = WordEquationDataset(graph_folder=graph_folder,data_fold="eval")
    evaluation_dataset.statistics()
    graph, label = evaluation_dataset[0]
    print("evaluation_dataset[0]", graph, label)

    evaluation_dataloader = GraphDataLoader(evaluation_dataset,  batch_size=1, drop_last=False)

    # Load the model
    model_path = "/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/models/model.pth"
    model = load_model(model_path)

    # Evaluate the model
    for (batched_graph, labels), graph in zip(evaluation_dataloader,evaluation_dataset.get_graph_list_from_folder()):
        pred = model(batched_graph, batched_graph.ndata["feat"].float())
        binary_labels = pred.argmax(dim=1)
        print("graph", graph["file_path"])
        print("number of nodes", len(graph["nodes"]))
        print("number of edges", len(graph["edges"]))
        print("pred", pred)
        print("binary_labels",binary_labels.item())
        # create the prediction file
        prediction_file = graph["file_path"].replace(".json","") + ".prediction"
        with open(prediction_file, 'w') as file:
            file.write(int_label_to_satisfiability[binary_labels.item()])











def load_model(model_path) :
    loaded_model = torch.load(model_path)
    loaded_model.eval()  # Set the model to evaluation mode
    return loaded_model


if __name__ == '__main__':
    main()