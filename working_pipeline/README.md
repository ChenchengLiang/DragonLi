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

3. run clean_train_data.py to unify the labels
Inputs: benchmark, graph type

4. Run store_dataset_to_pickle.py to store train data and graphs to pickle
Inputs: task name, graph type, node_type
Outputs: divided_i/dataset_graph_1.pkl.zip, valid_data/dataset_graph_1.pkl.zip

5. Run local_train.py to train the model
Inputs: benchmark_folder, node_type, and some other parameters can be set in the code
Outputs: Models/model_2_graph_1_GCNSplit.pth
Train can be monitored by mlflow at http://127.0.0.1:5000

6. Run main.py to verify results


Woking pipeline (cluster process for rank task):

1. Divide the benchmark into n=50 chunk size
Local: Set parameters in src/train_data_collection/divide_folder.py and run it

2. Generate train data:
Local: Set parameters in src/train_data_collection/generate_train_data_from_solver_one_folder.py
Alvis/UPPMAX: sh word_equation_submit_generate_train_data_parallel.sh benchmark


3. Unify the labels:
Local: Set parameters in src/train_data_collection/clean_train_data.py
Alvis/UPPMAX: sh word_equation_submit_clean_train_data.sh

4. Collect data
Local: call divided_train_and_valid_data.py to separate train and valid data. Divide the train data to chunks


5. Draw graphs for train data:
Local: Set parameters in src/train_data_collection/draw_graphs_for_train_data_one_folder.py
Alvis/UPPMAX: sh word_equation_submit_draw_graphs_for_train_data_parallel.sh benchmark


6. Store to pickle data:
Local: Set parameters in src/train_data_collection/store_dataset_to_pickle_one_folder.py
Alvis: sh word_equation_submit_store_dataset_to_pickle_parallel.sh benchmark

7. Initialize train configurations:
Local: Set parameters in src/solver/model/generate_configurations.py
Alvis: sh word_equation_submit_initialize_configurations.sh

8. Train
Alvis: sh word_equation_submit_multiple_train_continuously.sh

9. Select models send to UPPMAX
Local: in /home/cheli243/Desktop/CodeToGit/string-equation-solver/cluster-mlruns run sh get_models_by_run_id.sh run_id
Send to Uppmax path /home/cheli243/boosting-string-equation-solving-by-GNNs/Models


10. Evaluate:
Local: configurate evaluation parameters in generate_eval_configurations.py
Uppmax: sh word_equation_submit_multiple_eval.sh #evaluate track with good models
Uppmax: sh send_summary_back.sh

11. Summary evaluation:
Local: sh merge_and_summary.sh $benchmark_name # this will call merge_summary_folders.py and summary_solutions.py





