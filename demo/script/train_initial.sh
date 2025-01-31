#!/bin/bash

cd ../../

config_index=0

apptainer exec --nv container/eval_image.sif python3 /src/solver/rank_task_models/initialize_run_id_for_configurations_lightining.py --configuration_file Models/configurations/config_$config_index.json

