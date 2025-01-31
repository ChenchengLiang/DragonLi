#!/bin/bash

cd ../../

config_index=0
#each time train 10 epochs
train_n_times=1

for i in $(seq 0 $train_n_times)
do


apptainer exec --nv container/eval_image.sif python3 src/solver/rank_task_models/cluster_train_lightning.py --configuration_file Models/configurations/config_$config_index.json

done
