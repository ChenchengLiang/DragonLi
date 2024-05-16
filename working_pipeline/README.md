# boosting-string-equation-solving-by-GNNs (local process for rank task)

Woking pipeline:

1. Run generate_train_data_from_solver.py to generate train data
Inputs: task name, folder name 
The folder must have at least two folders: divided_i/SAT, valid_data/SAT
Outputs: divided_i/train.zip, valid_data/train.zip

2. Run draw_graphs_for_train_data.py to draw graphs for train data
Inputs: task name, folder name, graph type
Can change visualize=True in draw_func() to visualize the graph
Outputs: divided_i/graph_i.zip, valid_data/graph_i.zip

3. Run store_dataset_to_pickle.py to store train data and graphs to pickle
Inputs: task name, graph type, node_type
Outputs: divided_i/dataset_graph_1.pkl.zip, valid_data/dataset_graph_1.pkl.zip

4. Run local_train.py to train the model
Inputs: benchmark_folder, node_type, and some other parameters can be set in the code
Outputs: Models/model_2_graph_1_GCNSplit.pth
Train can be monitored by mlflow at http://127.0.0.1:5000

5. Run main.py to verify results