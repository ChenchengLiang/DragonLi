#!/bin/bash

cd ..
pwd
eq_file="webapp/user-input.eq"
rank_task="1"
order_equations_method="category_gnn_first_n_iterations"
model_path="Models/model_2_graph_1_GCNSplit.pth"

apptainer exec --nv container/eval_image.sif python3 src/process_benchmarks/main_parameter.py $eq_file "fixed" --graph_type "graph_1" --gnn_model_path $model_path --gnn_task "rank_task" --rank_task $rank_task --termination_condition "termination_condition_0" --order_equations_method $order_equations_method --algorithm "SplitEquations"

