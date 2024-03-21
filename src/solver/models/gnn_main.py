import os

os.environ["DGLBACKEND"] = "pytorch"
from Dataset import WordEquationDatasetBinaryClassification,KarateClubDataset,SyntheticDataset,WordEquationDatasetMultiClassification
from src.solver.Constants import bench_folder
from dgl.dataloading import GraphDataLoader
def main():


    # dataset = KarateClubDataset()
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[1])


    # dataset = SyntheticDataset()
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[1])
    #
    # print("-----")
    # data_loader = GraphDataLoader(dataset, batch_size=100, shuffle=True)
    # for batched_graph, labels in data_loader:
    #     print(batched_graph, labels)



    graph_folder=bench_folder+"/debug-train/graph_1"
    dataset = WordEquationDatasetMultiClassification(graph_folder=graph_folder, node_type=4,
                                                     label_size=2)
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])

    print("-----")
    data_loader = GraphDataLoader(dataset, batch_size=1000, shuffle=True)
    for batched_graph, labels in data_loader:
        print(batched_graph)
        print(len(labels))




if __name__ == '__main__':
    main()