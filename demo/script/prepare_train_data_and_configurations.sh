#!/bin/sh

benchmark="train_data"

# rank_task can be change to 0, 1, or 2. They corresponding to the Task 1, 2, and 3 in the paper respectively.
rank_task="1"
 
cd ../..

# Label data according to MUS
apptainer exec --nv container/eval_image.sif python3 src/train_data_collection/generate_train_data_from_unsatcores.py $benchmark

# Transform labeled data to graph representations.
# Data in divided_1 for training. data in valid_data for validation
apptainer exec --nv container/eval_image.sif python3 src/train_data_collection/draw_graphs_for_train_data_one_folder.py graph_1 $benchmark divided_1
apptainer exec --nv container/eval_image.sif python3 src/train_data_collection/draw_graphs_for_train_data_one_folder.py graph_1 $benchmark valid_data

# Store training data in pikle format to make loading data faster
apptainer exec --nv container/eval_image.sif python3 src/train_data_collection/store_dataset_to_pickle_one_folder.py "graph_1" "train_data" "divided_1" $rank_task
apptainer exec --nv container/eval_image.sif python3 src/train_data_collection/store_dataset_to_pickle_one_folder.py "graph_1" "train_data" "valid_data" $rank_task

# Generate configuration for training
# server port
port="12000"
apptainer exec --nv container/eval_image.sif python3 src/solver/models/generate_configurations.py $rank_task $benchmark $port

