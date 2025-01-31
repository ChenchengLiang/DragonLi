#!/bin/sh

 
cd ../..
# rank_task can be change to 0, 1, or 2. They corresponding to the Task 1, 2, and 3 in the paper respectively.
rank_task="1"

benchmark="train_data"
port="12000"

apptainer exec --nv container/eval_image.sif python3 src/solver/models/generate_configurations.py $rank_task $benchmark $port

