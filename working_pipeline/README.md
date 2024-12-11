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
1. Get answers for the benchmark
Local: set parameters in generate_eval_configurations.py
UPPMAX: sh word_equation_submit_multiple_eval.sh

2. Collect the answers and separate satisfiability
Local: run collec_answers_and_separate_benchmarks_by_satisifiability.py

3. Divide the benchmark into n=50 chunk size
Local: Set parameters in src/train_data_collection/divide_folder.py and run it

4. Generate train data:
Local: Set parameters in src/train_data_collection/generate_train_data_from_solver_one_folder.py
Alvis/UPPMAX: sh word_equation_submit_generate_train_data_parallel.sh benchmark


5. Unify the labels:
Local: Set parameters in src/train_data_collection/clean_train_data.py
Alvis/UPPMAX: sh word_equation_submit_clean_train_data.sh

6. Collect data
Local: call divided_train_and_valid_data.py to separate train and valid data. Divide the train data to chunks


7. Draw graphs for train data:
Local: Set parameters in src/train_data_collection/draw_graphs_for_train_data_one_folder.py
Alvis/UPPMAX: sh word_equation_submit_draw_graphs_for_train_data_parallel.sh benchmark


8. Store to pickle data:
Alvis: sh word_equation_submit_store_dataset_to_pickle_parallel.sh benchmark rank_task (an integer)

9. Initialize train configurations:
Local: Set parameters in src/solver/model/generate_configurations.py
Alvis: sh word_equation_submit_initialize_configurations.sh

10. Train
Alvis: sh word_equation_submit_multiple_train_continuously.sh

11. Select models send to UPPMAX
Local: in /home/cheli243/Desktop/CodeToGit/string-equation-solver/cluster-mlruns run sh get_models_by_run_id.sh run_id
Send to Uppmax path /home/cheli243/boosting-string-equation-solving-by-GNNs/Models


12. Evaluate:
Local: configurate evaluation parameters in generate_eval_configurations.py
Uppmax: sh word_equation_submit_multiple_eval.sh #evaluate track with good models
Uppmax: sh send_summary_back.sh

13. Summary evaluation:
Local: sh merge_and_summary.sh $benchmark_name # this will call merge_summary_folders.py and summary_solutions.py


Extract unsatcore pipeline (cluster process for rank task):
1. Get satisfiability sheet between this:category and other solvers
Local: configurate evaluation parameters in generate_eval_configurations.py
Uppmax: sh word_equation_submit_multiple_eval.sh
Uppmax: sh send_summary_back.sh

2. Identify problems can be used to extract unsatcores
Local: Set parameter and run identify_new_trainable_data.py

3. Divide problem set for cluster
Local: set parameter and run divided_folder.py
divide data in folder merged_new_trainable_data
The folder should looks like benchmark/z3/UNSAT/*.eq, benchmark/ostrich/UNSAT/*.eq,..., benchmark divided_i/UNSAT/*.eq

4. Extract unsatcores
Uppmax: sh word_equation_submit_generate_unsatcore_parallel.sh benchmark

5. Merge extracted unsatcores and divided to extract training data
Local: set parameter and run collect_unsatcores.py
   get two sets, one for extract from proof tree, one from first unsatcore
Local & Uppmax: set parameter and run divided_folder.py for the proof tree then use generate_train_data_from_solver_one_folder.py to generate data 
Local: set parameter and run generate_train_data_from_unsatcores.py to generate data from first unsatcore



Eval unsatcore pipeline:
1. Local: set parameter and run output_ranked_eq.py to get .predicted_unsatcore
2. Local: send GNN models to UPPMAX /home/cheli243/boosting-string-equation-solving-by-GNNs/mlruns
   Divide the benchmark into n=5 chunk size, the final directory should be benchmark/divided_i/*.eq
3. Local: change parameter benchmark_model in eval_GNN_ranked_eq.py
   Uppmax: sh word_equation_submit_eval_unsatcore_parallel.sh benchmark_name

4.Local: set parameter "get_unsatcore_func" in word_equation_submit_generate_unsatcore_from_all_solvers_parallel.sh
By setting differernt parameters it can get minimum or non-minimum unsatcore
   Uppmax: sh word_equation_submit_generate_unsatcore_from_all_solvers_parallel.sh benchmark_name
   This will run get_unsatcore_from_all_solvers.py in cluster

5. Local: collect_eval_unsatcore.py to get the final result