import os
import sys
import configparser

# Read path from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
path = config.get('Path','local')
sys.path.append(path)

from src.solver.Constants import int_label_to_satisfiability,project_folder,bench_folder
from Dataset import WordEquationDataset
from dgl.dataloading import GraphDataLoader
from src.solver.models.utils import load_model,load_model_from_mlflow
import mlflow

def main():
    graph_type="graph_1"
    benchmark="example_predict"

    # Load the evaluation dataset
    graph_folder = bench_folder+"/"+benchmark+"/"+graph_type
    evaluation_dataset = WordEquationDataset(graph_folder=graph_folder,data_fold="eval")
    evaluation_dataset.statistics()
    graph, label = evaluation_dataset[0]
    print("evaluation_dataset[0]", graph, label)

    evaluation_dataloader = GraphDataLoader(evaluation_dataset,  batch_size=1, drop_last=False)


    # Load the model from path
    #model_path = project_folder+"/Models/model_"+graph_type+".pth"
    #model = load_model(model_path)
    #load the model from mlflow
    experiment_id="856005721390468951"
    run_id="feb2e17e68bb4310bb3c539c672fd166"
    model = load_model_from_mlflow(experiment_id,run_id)



    # Evaluate the model
    for (batched_graph, label), graph in zip(evaluation_dataloader, evaluation_dataset.get_graph_list_from_folder()):
        pred = model(batched_graph, batched_graph.ndata["feat"].float())

        # Interpret the output as a binary label
        binary_label = (pred > 0.5).long().squeeze().item()

        print("graph", graph["file_path"])
        print("number of nodes", len(graph["nodes"]))
        print("number of edges", len(graph["edges"]))
        print("pred", pred.item())
        print("binary_label", binary_label)
        print("original label", label.item())

        # Create the prediction file
        prediction_file = graph["file_path"].replace(".json", "") + ".prediction"
        with open(prediction_file, 'w') as file:
            file.write(int_label_to_satisfiability[binary_label])











if __name__ == '__main__':
    main()