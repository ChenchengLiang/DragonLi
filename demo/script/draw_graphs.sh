#!/bin/sh

 
cd ../..


apptainer exec --nv container/eval_image.sif python3 src/train_data_collection/draw_graphs_for_train_data_one_folder.py graph_1 train_data divided_1

apptainer exec --nv container/eval_image.sif python3 src/train_data_collection/draw_graphs_for_train_data_one_folder.py graph_1 train_data valid_data

