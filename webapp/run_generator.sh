#!/bin/bash

cd ..

apptainer exec --nv container/eval_image.sif python3 src/train_data_collection/generate_one_problem.py "A1"


