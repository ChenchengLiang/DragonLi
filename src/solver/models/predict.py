from src.solver.Constants import int_label_to_satisfiability
from Dataset import WordEquationDataset
from dgl.dataloading import GraphDataLoader
from src.solver.models.utils import load_model

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
    for (batched_graph, labels), graph in zip(evaluation_dataloader, evaluation_dataset.get_graph_list_from_folder()):
        pred = model(batched_graph, batched_graph.ndata["feat"].float())

        # Interpret the output as a binary label
        binary_label = (pred > 0.5).long().squeeze().item()

        print("graph", graph["file_path"])
        print("number of nodes", len(graph["nodes"]))
        print("number of edges", len(graph["edges"]))
        print("pred", pred.item())
        print("binary_label", binary_label)

        # Create the prediction file
        prediction_file = graph["file_path"].replace(".json", "") + ".prediction"
        with open(prediction_file, 'w') as file:
            file.write(int_label_to_satisfiability[binary_label])











if __name__ == '__main__':
    main()