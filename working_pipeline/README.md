# boosting-string-equation-solving-by-GNNs 

Woking pipeline (local process for rank task):

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


Woking pipeline (cluster process for rank task):

1. Alvis:
Set parameters in src/train_data_collection/generate_train_data_from_solver_one_folder.py
sh word_equation_submit_generate_train_data_parallel.sh benchmark

2. Alvis:
Set parameters in src/train_data_collection/draw_graphs_for_train_data.py
sh word_equation_submit_draw_graphs_for_train_data.sh

3. Alvis:
Set parameters in src/train_data_collection/store_dataset_to_pickle.py
sh word_equation_submit_store_dataset_to_pickle.sh

4. Alvis:
Set parameters in src/solver/model/generate_configurations.py
sh word_equation_submit_initialize_configurations.sh

5. Alvis:
sh word_equation_submit_multiple_train_continuously.sh




