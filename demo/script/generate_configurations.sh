#!/bin/sh

 
cd ../..
tank_task="1"
benchmark="train_data"
port="12000"

apptainer exec --nv container/eval_image.sif python3 src/solver/models/generate_configurations.py $rank_task $benchmark $port

