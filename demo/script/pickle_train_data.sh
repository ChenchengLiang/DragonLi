#!/bin/sh

 
cd ../..


apptainer exec container/alvis_word_equation_recipe-A100.sif python3 src/train_data_collection/store_dataset_to_pickle_one_folder.py "graph_1" $2 $3 $4
