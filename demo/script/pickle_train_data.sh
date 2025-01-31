#!/bin/sh

 
cd ../..


apptainer exec --nv container/eval_image.sif python3 src/train_data_collection/store_dataset_to_pickle_one_folder.py "graph_1" "train_data" "divided_1" "1"

apptainer exec --nv container/eval_image.sif python3 src/train_data_collection/store_dataset_to_pickle_one_folder.py "graph_1" "train_data" "valid_data" "1"
