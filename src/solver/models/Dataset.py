import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from dgl.data import DGLDataset
import pandas as pd
import glob
import json

class WordEquationDataset(DGLDataset):
    def __init__(self,graph_folder,data_fold="train"):
        self.data_fold = data_fold
        self.graph_folder = graph_folder
        super().__init__(name="WordEquation")


    def process(self):
        self.graphs = []
        self.labels = []
        self.node_embedding_dim = 1
        self.gclasses = 2


        graph_generator=self.get_graph_list_from_folder()

        for g in graph_generator:
            edges_src, edges_dst = self.get_edge_src_and_dst_list(g["edges"])
            num_nodes = pd.DataFrame(g["nodes"]).to_numpy().shape[0]

            dgl_graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
            dgl_graph.ndata["feat"] = torch.from_numpy(pd.DataFrame(g["node_types"]).to_numpy())
            # dgl_graph.ndata["label"] = node_labels #node label
            dgl_graph.edata["weight"] = torch.from_numpy(pd.DataFrame(g["edge_types"]).to_numpy())
            dgl_graph = dgl.add_self_loop(dgl_graph)

            self.graphs.append(dgl_graph)
            self.labels.append(g["label"])

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def get_edge_src_and_dst_list(self, edges):
        edges_src = []
        edges_dst = []
        for e in edges:
            edges_src.append(e[0])
            edges_dst.append(e[1])
        return pd.DataFrame(edges_src).to_numpy().flatten(), pd.DataFrame(edges_dst).to_numpy().flatten()

    def list_to_pandas(self, graphs):
        for g in graphs:
            for k in g:
                if isinstance(g[k], list):
                    g[k] = pd.DataFrame(g[k])

    def statistics(self):
        sat_label_number=0
        unsat_label_number=0
        unknown_label_number=0
        for g in self.get_graph_list_from_folder():
            if g["label"]==1:
                sat_label_number+=1
            elif g["label"]==0:
                unsat_label_number+=1
            else:
                unknown_label_number+=1

        print("sat_label_number",sat_label_number,"unsat_label_number",unsat_label_number,"unknown_label_number",unknown_label_number)


    def get_graph_list_from_folder(self):
        '''
        graph_1 = {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2], "edges": [[1, 2], [2, 3], [3, 0]],
                   "edge_types": [1, 1, 1],
                   "label": 1}
        graph_2 = {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2], "edges": [[1, 2], [2, 3], [3, 0]],
                   "edge_types": [1, 1, 1],
                   "label": 1}
        graphs = [graph_1, graph_2]
        '''
        graph_file_list = glob.glob(self.graph_folder + "/*.json")

        for graph_file in graph_file_list:
            with open(graph_file, 'r') as f:
                loaded_dict = json.load(f)
            if self.data_fold == "train":
                if loaded_dict["label"] !=-1:
                    yield loaded_dict
            else:
                yield loaded_dict



class SyntheticDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="synthetic")

    def process(self):
        edges = pd.read_csv("./graph_edges.csv")
        properties = pd.read_csv("./graph_properties.csv")
        self.graphs = []
        self.labels = []

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row["graph_id"]] = row["label"]
            num_nodes_dict[row["graph_id"]] = row["num_nodes"]

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby("graph_id")

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id["src"].to_numpy()
            dst = edges_of_id["dst"].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class KarateClubDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="karate_club")

    def process(self):
        nodes_data = pd.read_csv("./members.csv")
        edges_data = pd.read_csv("./interactions.csv")
        node_features = torch.from_numpy(nodes_data["Age"].to_numpy())
        node_labels = torch.from_numpy(
            nodes_data["Club"].astype("category").cat.codes.to_numpy()
        )
        edge_features = torch.from_numpy(edges_data["Weight"].to_numpy())
        edges_src = torch.from_numpy(edges_data["Src"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["Dst"].to_numpy())

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=nodes_data.shape[0]
        )
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels
        self.graph.edata["weight"] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train: n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
